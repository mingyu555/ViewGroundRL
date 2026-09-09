"""DriveLMM-o1 test 추론. 공식 파이프라인과 같은 입력 구성을 쓴다.

공식 prepare_data_internvl.py 가 "stitched multiview image stored with idx as name"
라고 밝히듯, 6뷰를 하나의 2x3 격자 이미지로 합성해 모델에 넣는다. 정답이
"top middle image", "bottom row, first image (back right)" 처럼 격자 위치를
가리키므로 6장을 따로 주면 정답과 어긋난다.

격자 배치는 원본 image 배열 순서 그대로:
    top    : CAM_FRONT_LEFT   CAM_FRONT   CAM_FRONT_RIGHT
    bottom : CAM_BACK_RIGHT   CAM_BACK    CAM_BACK_LEFT

출력은 공식 evaluation_script.py 가 받는 형식([{idx, question, llm-response}])으로
저장해, 나중에 GPT 채점만 덧붙일 수 있게 한다.
"""
import argparse, json, os, uuid

CELL_W, CELL_H = 800, 450       # nuScenes 1600x900 의 절반 -> 격자 2400x900


def stitch(paths, cache_dir, key):
    from PIL import Image
    out = os.path.join(cache_dir, f"{key}.jpg")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        try:
            Image.open(out).verify()
            return out
        except Exception:
            pass          # 손상된 캐시는 다시 만든다
    grid = Image.new("RGB", (CELL_W * 3, CELL_H * 2))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((CELL_W, CELL_H), Image.BICUBIC)
        grid.paste(im, ((i % 3) * CELL_W, (i // 3) * CELL_H))
    os.makedirs(cache_dir, exist_ok=True)
    # 두 프로세스가 같은 캐시를 공유하면 반쯤 쓰인 파일을 읽어 PIL 이 죽는다.
    # 임시 파일에 쓰고 원자적으로 이름을 바꾼다.
    # os.getpid() 는 컨테이너 안에서 둘 다 1 이라 두 프로세스의 임시 파일명이 겹쳤다.
    tmp = f"{out}.{uuid.uuid4().hex}.part.jpg"
    grid.save(tmp, "JPEG", quality=92)
    os.replace(tmp, out)
    return out


# 프롬프트 세 갈래. 베이스라인 재현이 어긋난 원인을 분해하기 위한 것이다.
#   neutral : 공식 inference.py 와 같은 최소 구성 ('<image>\n' + question). 격자 설명도,
#             형식 지시도 없다. 논문 조건에 가장 가깝다고 보는 조건.
#   grid    : 격자 배치만 알려주고 형식 지시는 없음. 격자 설명의 효과를 분리한다.
#   guided  : 격자 + 형식 강제. 첫 측정에 쓴 조건 (7B 51.07%).
GRID_DESC = (
    "The image is a 2x3 grid of the six surround-view cameras of the ego vehicle: "
    "the top row is front-left, front, front-right; the bottom row is back-right, "
    "back, back-left."
)
FORMAT_DESC = (
    " Reason step by step about the scene, then state your conclusion. Always end "
    "your reply with a line beginning '**Final Answer**:'. For a multiple-choice "
    "question, the final answer must repeat the chosen option exactly, starting with "
    "its letter, e.g. 'B) ...'."
)
SYSTEMS = {
    "neutral": None,
    "grid": "You are an autonomous driving assistant. " + GRID_DESC,
    "guided": "You are an autonomous driving assistant. " + GRID_DESC + FORMAT_DESC,
}


def mask_cell(img, cell):
    """2x3 격자에서 셀 하나를 검게 칠한다.

    6뷰를 한 장으로 합쳐 넣는 구조라, '뷰를 가린다' = '격자 셀을 가린다' 가 된다.
    셀 인덱스는 CAMERAS 순서(FL,F,FR,BR,B,BL)와 같다.
    """
    from PIL import Image, ImageDraw
    out = img.copy()
    d = ImageDraw.Draw(out)
    x0, y0 = (cell % 3) * CELL_W, (cell // 3) * CELL_H
    d.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], fill=(0, 0, 0))
    return out


def pick_control(evidence, seed_key):
    """근거뷰가 아닌 셀 하나를 결정적으로 고른다."""
    import hashlib
    cand = [i for i in range(6) if i not in evidence]
    if not cand:
        return None
    h = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    return cand[h % len(cand)]


def build_prompt(question, style):
    sysmsg = SYSTEMS[style]
    head = f"<|im_start|>system\n{sysmsg}<|im_end|>\n" if sysmsg else ""
    return (head + "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            + question + "<|im_end|>\n<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="/mnt/ssd1/vgrl/data/dlmm_test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="/mnt/ssd2/mingyu/dlmm_stitch")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--max_pixels", type=int, default=2408448)   # 6뷰 따로 줄 때와 동등
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    # 데이터 병렬: GPU 당 프로세스 하나가 전체의 1/num_shards 를 맡는다.
    # 8B 모델은 한 장에 다 올라가므로 텐서 병렬보다 처리량이 훨씬 낫다.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--prompt_style", choices=list(SYSTEMS), default="guided")
    ap.add_argument("--mask", choices=["none", "evidence", "control"], default="none",
                    help="격자 셀 하나를 가린다 (blank-view 진단)")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    rows = json.load(open(args.dataset))
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard :: args.num_shards]
    # 이어받기: 이미 나온 문항은 건너뛴다 (합성 캐시도 남아 있어 재시작이 싸다)
    done = []
    if args.resume and os.path.exists(args.out):
        try:
            done = json.load(open(args.out))
        except Exception:
            done = []
    have = {x["idx"] for x in done}
    rows = [r for r in rows if r["id"].split("::", 1)[1] not in have]
    print(f"문항 {len(rows)} (이미 완료 {len(have)})", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": 1}, max_model_len=8192,
              gpu_memory_utilization=args.gpu_memory_utilization,
              mm_processor_kwargs={"max_pixels": args.max_pixels},
              disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=True, stop_token_ids=[151645, 151643])

    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        inputs = []
        for r in chunk:
            key = r["id"].split("::", 1)[1].rsplit("_", 1)[0]     # 프레임 단위 캐시
            img = Image.open(stitch(r["views"], args.cache_dir, key)).convert("RGB")
            if args.mask != "none":
                ev = r.get("evidence_views") or []
                cell = ev[0] if args.mask == "evidence" else pick_control(ev, r["id"])
                if cell is None:
                    continue
                img = mask_cell(img, cell)
            inputs.append({"prompt": build_prompt(r["prompt_text"], args.prompt_style),
                           "multi_modal_data": {"image": img}})
        if not inputs:
            continue
        for r, o in zip(chunk, llm.generate(inputs, sampling)):
            done.append({"idx": r["id"].split("::", 1)[1],
                         "question": r["prompt_text"],
                         "llm-response": o.outputs[0].text.strip()})
        print(f"  {len(done)}/{len(rows)}", flush=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
    json.dump(done, open(args.out, "w"), ensure_ascii=False)
    print(f"완료 {len(done)} -> {args.out}")


if __name__ == "__main__":
    main()
