"""DriveLM 공식 test server 제출 파일 생성.

입력은 VGGDrive 가 공개한 test 캐시(15,480 문항 / 149 씬 / 799 프레임)를 쓴다.
공식 test GT 는 비공개이므로 질문·이미지 목록만 여기서 얻고 답변은 우리 모델이 만든다.
질문 텍스트의 "These six images are ..." 공통 접두어(15480/15480)는 벗겨낸다 —
우리 모델은 그 정보를 system 메시지로 받고 학습했다.
"""
import argparse, json, os, re, sys

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
PREFIX = ("These six images are the front view, front left view, front right view, "
          "back view, back left view and back right view of the ego vehicle. ")
SYSTEM = (
    "You are the perception and reasoning module of an autonomous vehicle. You are "
    "given the six surround-view camera images of the current frame, in this order: "
    "CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, "
    "CAM_BACK_RIGHT. Objects are referred to as <cID,CAMERA,x,y>, where CAMERA names "
    "the view the object appears in and (x,y) is its pixel location in that view. "
    "Answer using the same reference format. Reason step by step inside "
    "<think>...</think>, then give only the final answer inside <answer>...</answer>."
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def load_test(cache, root):
    rows = []
    for x in json.load(open(cache)):
        imgs, q = [], None
        for c in x["messages"][0]["content"]:
            if c["type"] == "image":
                imgs.append(os.path.join(root, c["image"].replace("file://nuscenes_dataset/", "")))
            elif c["type"] == "text":
                q = c["text"]
        assert len(imgs) == 6, x["id"]
        for cam, p in zip(CAMERAS, imgs):
            assert f"/{cam}/" in p, f"카메라 순서 불일치: {p} vs {cam}"
        rows.append({"id": x["id"], "images": imgs,
                     "question": q[len(PREFIX):] if q.startswith(PREFIX) else q})
    return rows


def build_prompt(question):
    body = "<|vision_start|><|image_pad|><|vision_end|>" * 6 + question
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")


def clean(text):
    """제출은 최종 답변만 담는다 — <think> 를 남기면 공식 지표가 그것까지 채점한다."""
    m = ANSWER_RE.search(text)
    out = m.group(1) if m else THINK_RE.sub("", text)
    for tok in ("<|im_end|>", "<|endoftext|>", "<answer>", "</answer>",
                "<think>", "</think>"):
        out = out.replace(tok, "")
    return " ".join(out.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="/mnt/ssd1/vgrl/baselines/vggdrive_json/"
                                       "nuScenes_cache/Drivelm_Qwen_test_15480.json")
    ap.add_argument("--nuscenes_root", default="/nuscenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--image_resolution", type=int, default=401408)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vllm import LLM, SamplingParams
    from eval_grounding import load_views

    rows = load_test(args.cache, args.nuscenes_root)
    if args.limit:
        rows = rows[: args.limit]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(rows)} 문항", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=1, limit_mm_per_prompt={"image": 6},
              max_model_len=8192, gpu_memory_utilization=args.gpu_memory_utilization,
              disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=False, stop_token_ids=[151645, 151643])

    done, empty = [], 0
    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        inputs = [{"prompt": build_prompt(r["question"]),
                   "multi_modal_data": {"image": load_views(r["images"], args.image_resolution)}}
                  for r in chunk]
        for r, o in zip(chunk, llm.generate(inputs, sampling)):
            a = clean(o.outputs[0].text)
            if not a:
                empty += 1
            done.append({"id": r["id"], "question": r["question"], "answer": a})
        print(f"  {len(done)}/{len(rows)}  빈답변 {empty}", flush=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
    json.dump(done, open(args.out, "w"), ensure_ascii=False)
    print(f"완료 {len(done)} → {args.out} (빈답변 {empty})")


if __name__ == "__main__":
    main()
