"""Teacher client for CoT label generation.

Talks to any OpenAI-compatible `/v1/chat/completions` endpoint, which covers both
a locally served Qwen2.5-VL-72B (`scripts/serve_teacher.sh`) and a hosted API.

Differences from MindDriver's version of this step (gen_data/api_call_mutil.py),
which are the reasons this is rewritten rather than reused:

  - it shelled out to a bash script that base64'd ten images per call, one sample
    at a time, and rewrote the entire growing result json after *every* sample
    (quadratic I/O over 23k samples);
  - a hardcoded bearer token was committed to the repo;
  - no retry, no backoff, no resume beyond "is this id already a key".

Here: bounded-concurrency asyncio, append-only jsonl with resume by id, retry with
exponential backoff, and the endpoint/key read from the environment.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from PIL import Image


@dataclass
class TeacherRequest:
    uid: str
    text: str
    images: list[str]
    """Ordered image paths; captions[i] labels images[i] in the message."""
    captions: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class TeacherConfig:
    base_url: str = os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:8000/v1")
    """One endpoint, or several comma-separated. Several are round-robined per
    request, which is how a 72B teacher gets served here: tensor-parallel across
    four H200s dies in vllm 0.25.1's all-reduce path with an AWQ checkpoint, and a
    43 GB AWQ model fits on one 141 GB card anyway — so four independent TP=1
    servers replace TP=4. For bulk generation that is also simply faster, since
    per-token all-reduce disappears."""
    api_key: str = os.environ.get("TEACHER_API_KEY", "EMPTY")
    model: str = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    concurrency: int = 8
    max_retries: int = 5
    timeout_s: float = 600.0
    image_long_edge: int = 896
    """Downscale each view's long edge before base64. Six views at full 1600x900
    is ~4 MB of base64 per request and dominates wall-clock."""
    jpeg_quality: int = 85


def encode_image(path: str, long_edge: int, quality: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > long_edge:
        scale = long_edge / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def build_message(req: TeacherRequest, cfg: TeacherConfig) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": req.text}]
    captions = req.captions or [""] * len(req.images)
    for cap, path in zip(captions, req.images):
        if cap:
            content.append({"type": "text", "text": cap})
        b64 = encode_image(path, cfg.image_long_edge, cfg.jpeg_quality)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    return [{"role": "user", "content": content}]


def endpoints(cfg: "TeacherConfig") -> list[str]:
    return [u.strip().rstrip("/") for u in cfg.base_url.split(",") if u.strip()]


def load_done(path: str) -> dict[str, dict]:
    """Existing results keyed by uid, so a run can be resumed or extended."""
    done: dict[str, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("uid"):
                done[rec["uid"]] = rec
    return done


async def _one(client, req: TeacherRequest, cfg: TeacherConfig, sem: asyncio.Semaphore,
               url: str) -> dict:
    payload = {
        "model": cfg.model,
        "messages": build_message(req, cfg),
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    }
    async with sem:
        last_err = None
        for attempt in range(cfg.max_retries):
            try:
                r = await client.post(
                    f"{url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                    timeout=cfg.timeout_s,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                return {
                    "uid": req.uid,
                    "ok": True,
                    "text": data["choices"][0]["message"]["content"],
                    "meta": req.meta,
                }
            except Exception as e:  # network, 5xx, malformed body
                last_err = e
                if attempt == cfg.max_retries - 1:
                    break
                await asyncio.sleep(min(2**attempt, 30) * (1 + 0.3 * random.random()))
        return {"uid": req.uid, "ok": False, "error": repr(last_err), "meta": req.meta}


async def _run(requests: Sequence[TeacherRequest], cfg: TeacherConfig, out_path: str,
               on_result: Callable[[dict], None] | None) -> None:
    import httpx

    urls = endpoints(cfg)
    # concurrency is per endpoint, so N servers get N x the in-flight requests
    sem = asyncio.Semaphore(cfg.concurrency * len(urls))
    t0 = time.time()
    n_done = n_fail = 0
    limits = httpx.Limits(max_connections=cfg.concurrency * len(urls) + 4)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(out_path, "a", encoding="utf-8") as out:
            tasks = [
                asyncio.create_task(_one(client, r, cfg, sem, urls[i % len(urls)]))
                for i, r in enumerate(requests)
            ]
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                n_fail += 0 if rec.get("ok") else 1
                if on_result:
                    on_result(rec)
                if n_done % 25 == 0 or n_done == len(requests):
                    el = time.time() - t0
                    print(
                        f"  {n_done}/{len(requests)}  {el:.0f}s  "
                        f"{n_done / max(el, 1e-9):.2f} req/s  failed={n_fail}",
                        flush=True,
                    )


def generate(
    requests: Iterable[TeacherRequest],
    cfg: TeacherConfig,
    out_path: str,
    resume: bool = True,
    on_result: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Generate for every request, skipping uids already present in `out_path`.

    Returns all results for this output file, previous runs included.
    """
    requests = list(requests)
    done = load_done(out_path) if resume else {}
    pending = [r for r in requests if r.uid not in done or not done[r.uid].get("ok")]
    print(f"teacher: {len(requests)} requests, {len(done)} cached, {len(pending)} to run")

    if pending:
        asyncio.run(_run(pending, cfg, out_path, on_result))
    return load_done(out_path)


def check_endpoint(cfg: TeacherConfig) -> str:
    """Fail fast, and check *every* endpoint — one dead server out of four would
    otherwise only show up as a third of the requests failing much later."""
    import httpx

    lines = []
    for base in endpoints(cfg):
        url = f"{base}/models"
        try:
            r = httpx.get(url, headers={"Authorization": f"Bearer {cfg.api_key}"}, timeout=15)
            r.raise_for_status()
            names = [m["id"] for m in r.json().get("data", [])]
            if cfg.model not in names:
                print(f"  warning: model {cfg.model!r} not served by {base}; it has {names}")
            lines.append(f"  OK  {base} -> {names}")
        except Exception as e:
            raise SystemExit(
                f"teacher endpoint unreachable at {url}: {e}\n"
                "Start one with scripts/serve_teacher.sh, or set TEACHER_BASE_URL to a "
                "comma-separated list of OpenAI-compatible endpoints."
            )
    return "\n".join(lines)
