"""DriveLMM-o1 을 우리 SFT / RL / 평가 형식으로 변환한다.

DriveLMM-o1 (ayeshaishaq/DriveLMMo1): nuScenes 기반, train 18,507 QA / test 4,634 QA.
train 345 씬 / test 115 씬으로 공식 split 이 이미 씬 단위이고 겹침이 0 이다.

여기서 챙기는 세 가지:
  1) 이미지 순서를 원본 그대로 둔다 - FL, F, FR, BR, B, BL 의 2x3 격자이고
     정답이 "top middle image", "bottom row, first image (back right)" 처럼
     격자 위치를 직접 가리키므로 순서를 바꾸면 정답이 틀려진다.
  2) 정답에 이미 '**Step-by-Step Reasoning**' + '**Final Answer**:' 가 들어 있어
     (99.0%) CoT 증류가 필요 없다. DriveLM 때와 달리 교사 호출이 0 이다.
  3) 근거뷰는 질문의 객체 태그 <oN,CAMERA,x,y> 에서 뽑는다 (train 37.1%).
     태그가 없는 문항은 vg_usable=False 로 두되 SFT 에는 그대로 쓴다.
"""
import argparse, json, os, random, re
from collections import Counter

# 원본 순서. 2x3 격자의 top(FL,F,FR) / bottom(BR,B,BL) 에 대응한다.
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
GRID = ["top left", "top middle", "top right",
        "bottom left (back right)", "bottom middle (back)", "bottom right (back left)"]
TAG = re.compile(r"<o\d+,(CAM_[A-Z_]+),([\d.]+),([\d.]+)>")
FINAL = re.compile(r"\*\*Final Answer\*\*:\s*(.*)", re.S)

# tools/dlmm_infer.py 의 "guided" 와 글자까지 동일해야 한다 - 학습과 평가의
# 프롬프트가 갈리면 형식 준수율이 달라져 MCQ 점수가 흔들린다 (측정으로 확인:
# 형식 지시 유무로 strict MCQ 가 0% <-> 51% 로 움직였다).
SYSTEM = (
    "You are an autonomous driving assistant. The image is a 2x3 grid of the six "
    "surround-view cameras of the ego vehicle: the top row is front-left, front, "
    "front-right; the bottom row is back-right, back, back-left. Reason step by step "
    "about the scene, then state your conclusion. Always end your reply with a line "
    "beginning '**Final Answer**:'. For a multiple-choice question, the final answer "
    "must repeat the chosen option exactly, starting with its letter, e.g. 'B) ...'."
)


def evidence_views(question):
    """질문이 지목한 카메라의 인덱스. 없으면 빈 리스트."""
    cams = {m.group(1) for m in TAG.finditer(question)}
    return sorted(CAMERAS.index(c) for c in cams if c in CAMERAS)


def image_paths(rec, nusc_root):
    """원본은 'samples/CAM_X/....jpg' 상대경로를 순서대로 담고 있다."""
    out = []
    for cam in CAMERAS:
        hit = [p for p in rec["image"] if f"/{cam}/" in p]
        if len(hit) != 1:
            return None
        out.append(os.path.join(nusc_root, hit[0]))
    return out


def stitch_path(frame, stitch_dir):
    """평가와 같은 2x3 격자 합성 이미지 경로. tools/dlmm_infer.py 가 만든 캐시를 공유한다."""
    return os.path.join(stitch_dir, f"{frame}.jpg")


def to_row(rec, nusc_root, split, stitch_dir):
    imgs = image_paths(rec, nusc_root)
    if imgs is None:
        return None
    q = rec["question"].strip()
    if split == "train":
        target = rec["answer"].strip()
        final = (FINAL.search(target).group(1).strip()
                 if FINAL.search(target) else target)
    else:
        steps = (rec.get("steps") or "").strip()
        final = (rec.get("final_answer") or "").strip()
        target = (f"**Step-by-Step Reasoning**:\n{steps}\n\n**Final Answer**: {final}"
                  if steps else final)
    ev = evidence_views(q)
    scene, frame = rec["idx"].split("_")[0], rec["idx"].split("_")[1]
    _ = frame
    return {
        "id": f"dlmm::{rec['idx']}",
        "scene": scene,
        "frame": frame,
        # 학습·평가 모두 2x3 격자 합성 이미지 1장을 쓴다. 정답이 "top middle image"
        # 처럼 격자 위치를 가리키므로 6장을 따로 주면 정답과 어긋난다.
        "images": [stitch_path(frame, stitch_dir)],
        "views": imgs,               # 합성 캐시가 없을 때 만들기 위한 원본 6장
        "system": SYSTEM,
        "prompt_text": q,
        # GRPO(train_rl.py)는 chat 형식 `prompt` 를 요구한다. SFT(train_sft.py)는
        # `conversations` 를 쓰므로 둘 다 넣어 하나의 파일로 양쪽을 돌린다.
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": q}]},
        ],
        "conversations": [{"from": "human", "value": "<image>\n" + q},
                          {"from": "gpt", "value": target}],
        "solution": final,          # 평가·보상은 최종 답변만 본다
        "steps": rec.get("steps", ""),
        "evidence_views": ev,
        "vg_usable": len(ev) > 0 and len(ev) < len(CAMERAS),
        "mcq": bool(re.search(r"Choose from the following answers only", q, re.I)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="/mnt/ssd1/vgrl/raw/drivelmm")
    ap.add_argument("--nusc_root", default="/nuscenes")
    ap.add_argument("--out_dir", default="/mnt/ssd1/vgrl/data")
    ap.add_argument("--stitch_dir", default="/mnt/ssd1/vgrl/tmp/dlmm_stitch")
    ap.add_argument("--val_scenes", type=int, default=40,
                    help="train 씬에서 떼어낼 자체 검증 씬 수 (공식 test 는 따로 둔다)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = json.load(open(f"{args.raw_dir}/DriveLMMo1_TRAIN.json"))
    te = json.load(open(f"{args.raw_dir}/DriveLMMo1_TEST.json"))

    rows_tr = [r for r in (to_row(x, args.nusc_root, "train", args.stitch_dir) for x in tr) if r]
    rows_te = [r for r in (to_row(x, args.nusc_root, "test", args.stitch_dir) for x in te) if r]

    # 자체 검증은 씬 단위로 뗀다 (프레임 누수 방지)
    scenes = sorted({r["scene"] for r in rows_tr})
    random.Random(args.seed).shuffle(scenes)
    val_sc = set(scenes[: args.val_scenes])
    sft_train = [r for r in rows_tr if r["scene"] not in val_sc]
    sft_val = [r for r in rows_tr if r["scene"] in val_sc]
    rl_train = [r for r in sft_train if r["vg_usable"]]

    out = {
        "dlmm_sft_train.json": sft_train,
        "dlmm_sft_val.json": sft_val,
        "dlmm_rl_train.json": rl_train,
        "dlmm_test.json": rows_te,
    }
    for name, rows in out.items():
        p = os.path.join(args.out_dir, name)
        json.dump(rows, open(p, "w"), ensure_ascii=False)
        sc = len({r["scene"] for r in rows}); fr = len({r["frame"] for r in rows})
        vg = sum(r["vg_usable"] for r in rows); mc = sum(r["mcq"] for r in rows)
        print(f"{name:24} {len(rows):6} QA  {sc:4} 씬  {fr:5} 프레임  "
              f"vg_usable {vg/max(1,len(rows)):5.1%}  객관식 {mc/max(1,len(rows)):5.1%}")

    f_tr = {r["frame"] for r in sft_train}
    print(f"\n프레임 겹침  sft_train ∩ sft_val  : {len(f_tr & {r['frame'] for r in sft_val})}")
    print(f"프레임 겹침  sft_train ∩ test     : {len(f_tr & {r['frame'] for r in rows_te})}")
    print(f"근거뷰 분포 (rl_train): "
          f"{Counter(len(r['evidence_views']) for r in rl_train).most_common()}")


if __name__ == "__main__":
    main()
