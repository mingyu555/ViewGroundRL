"""학습 안 한 Qwen2.5-VL 을 OmniDrive 의 nuScenes open-loop planning 에 붙인다.

OmniDrive 의 예측 파일은 토큰 이름의 JSON 하나이고, eval_planning.py 는 정규식으로
'[PT, (x, y), ...]' 6개를 뽑는다. 즉 mmdet3d 파이프라인을 거칠 필요 없이 이 문자열만
만들면 어떤 모델이든 같은 지표로 잴 수 있다.

프롬프트 두 갈래:
  images  - 6뷰 + OmniDrive 의 원 질문만
  ego     - 위 + ego status 를 텍스트로 (VGGDrive 방식). OmniDrive 는 ego status 를
            학습된 임베딩 토큰으로 넣으므로 학습 안 한 모델로는 그 경로를 못 쓴다.
            텍스트 나열이 가장 가까운 대응이다.
"""
import argparse, json, os, re, sys, math

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
QUESTION = "Please provide the planning trajectory for the ego car without reasons."
CMD = {0: "Turn Right", 1: "Turn Left", 2: "Go Straight"}

SYSTEM = (
    "You are the planning module of an autonomous vehicle. You are given the six "
    "surround-view camera images of the current frame, in this order: "
    + ", ".join(CAMS) + ". The X axis points along your heading and the Y axis to "
    "your left; you are at (0,0); units are metres. Plan the ego vehicle's waypoints "
    "for the next 3 seconds at 0.5 s intervals. Reply with exactly one line in this "
    "format and nothing else:\n"
    "Here is the planning trajectory [PT, (+x1, +y1), (+x2, +y2), (+x3, +y3), "
    "(+x4, +y4), (+x5, +y5), (+x6, +y6)]."
)

# eval_planning.py 의 정규식은 소수점을 요구한다: r'-?\d+\.\d+'
TRAJ_RE = re.compile(r"\[PT,\s*\(\s*[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+\s*\)"
                     r"(?:\s*,\s*\(\s*[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+\s*\))*\]")
PAIR_RE = re.compile(r"\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)")


def ego_block(info):
    """can_bus 에서 ego status 를 뽑아 VGGDrive 스타일로 나열한다.

    이 pkl 의 can_bus 는 13차원이고 열 통계로 확인한 배치는 다음과 같다:
      [0:4] 쿼터니언(w,0,0,z)  [4:7] 가속도(ax,ay,az≈9.77=중력)
      [7:10] 회전율(·,·,yaw)   [10:13] 속도(종방향, 0, 0)
    내비 명령은 'command' 가 아니라 'gt_planning_command' 에 있다 (0/1/2, val 전량 채워짐).
    """
    cb = list(info.get("can_bus") or [])
    cmd = CMD.get(int(info.get("gt_planning_command", 2)), "Go Straight")
    if len(cb) < 13:
        return f"- Navigation Information = {cmd}\n"
    ax, ay = cb[4], cb[5]
    yaw_rate = cb[9]
    speed = cb[10]
    return (f"- Longitudinal velocity: {speed:.2f} m/s\n"
            f"- Longitudinal acceleration: {ax:.2f} m/s^2\n"
            f"- Lateral acceleration: {ay:.2f} m/s^2\n"
            f"- Yaw rate: {yaw_rate:.2f} rad/s\n"
            f"- Vehicle length: 4.08 m\n- Vehicle width: 1.85 m\n"
            f"- Navigation Information = {cmd}\n")


def build_prompt(info, mode):
    body = "<|vision_start|><|image_pad|><|vision_end|>" * len(CAMS)
    if mode == "ego":
        body += "Here is the ego vehicle state:\n" + ego_block(info) + "\n" + QUESTION
    else:
        body += QUESTION
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse(text, lenient=False):
    """strict: eval_planning.py 의 정규식이 그대로 받는 문자열만 통과시킨다.

    lenient: 학습 안 한 모델은 요청한 형식을 거의 못 지킨다. 그때 앞에서부터 숫자
    12개를 (x,y) 6쌍으로 읽어 재구성한다 - '형식을 못 맞춘 것'과 '계획을 못 하는 것'을
    분리해 보기 위한 완화 규약이며, OmniDrive 공식 규약이 아니다."""
    m = TRAJ_RE.search(text)
    if m and len(PAIR_RE.findall(m.group(0))) >= 6:
        return m.group(0)
    if not lenient:
        return None
    nums = NUM_RE.findall(text)
    if len(nums) < 12:
        return None
    pts = [(float(nums[i]), float(nums[i + 1])) for i in range(0, 12, 2)]
    return "[PT, " + ", ".join(f"({x:+.2f}, {y:+.2f})" for x, y in pts) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["images", "ego"], default="ego")
    ap.add_argument("--out_dir", required=True)
    # 원본 pkl 은 map_geoms 에 shapely 객체가 있어 shapely 없는 환경에서 언피클이 안 된다.
    # 추론에 필요한 필드(token / 카메라 파일명 / can_bus / 내비 명령)만 뽑아둔 JSON 을 쓴다.
    ap.add_argument("--info", default="/mnt/ssd1/vgrl/data/omnidrive_val_lite.json")
    ap.add_argument("--nusc_root", default="/data/nuScenes")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--image_resolution", type=int, default=401408)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0, help="실패한 원출력을 N건 찍는다")
    ap.add_argument("--lenient", action="store_true", help="숫자 12개를 6쌍으로 재구성 (비공식)")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    def load_views(paths):
        out = []
        for p in paths:
            im = Image.open(p)
            if im.width * im.height > args.image_resolution:
                f = math.sqrt(args.image_resolution / (im.width * im.height))
                im = im.resize((int(im.width * f), int(im.height * f)), Image.BICUBIC)
            out.append(im.convert("RGB"))
        return out

    infos = json.load(open(args.info))
    if args.limit:
        infos = infos[: args.limit]
    print(f"프레임 {len(infos)}  mode={args.mode}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": len(CAMS)}, max_model_len=8192,
              gpu_memory_utilization=args.gpu_memory_utilization, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=False, stop_token_ids=[151645, 151643])

    ok = bad = 0
    for s in range(0, len(infos), args.batch_size):
        chunk = infos[s : s + args.batch_size]
        inputs, keep = [], []
        for info in chunk:
            try:
                paths = [os.path.join(args.nusc_root, "samples", c, info["cams"][c])
                         for c in CAMS]
                imgs = load_views(paths)
            except Exception:
                bad += 1
                continue
            inputs.append({"prompt": build_prompt(info, args.mode),
                           "multi_modal_data": {"image": imgs}})
            keep.append(info)
        if not inputs:
            continue
        for info, o in zip(keep, llm.generate(inputs, sampling)):
            raw = o.outputs[0].text
            traj = parse(raw, args.lenient)
            if traj is None:
                bad += 1
                if args.debug and bad <= args.debug:
                    print(f"  [실패 {bad}] {raw[:400]!r}", flush=True)
                continue          # 파싱 실패는 파일을 쓰지 않는다 (eval 이 건너뜀)
            ok += 1
            json.dump([{"Q": QUESTION, "A": [f"Here is the planning trajectory {traj}."]}],
                      open(os.path.join(args.out_dir, info["token"]), "w"))
        print(f"  {ok+bad}/{len(infos)}  파싱성공 {ok}  실패 {bad}", flush=True)
    print(f"완료: 파싱성공 {ok}/{len(infos)} = {ok/len(infos):.1%}, 실패 {bad}")


if __name__ == "__main__":
    main()
