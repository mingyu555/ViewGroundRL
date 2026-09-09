"""DriveLMM-o1 전 프레임의 2x3 격자 합성 이미지를 미리 만든다.

학습(1,727+235 프레임)과 평가(539 프레임)가 같은 캐시를 공유한다.
tools/dlmm_infer.py 의 stitch() 와 같은 규격(셀 800x450, JPEG q92)이다.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dlmm_infer import stitch, CELL_W, CELL_H


def job(args):
    frame, views, cache = args
    try:
        stitch(views, cache, frame)
        return 1
    except Exception as e:
        return f"{frame}: {e}"


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd1/vgrl/tmp/dlmm_stitch"
    tasks, seen = [], set()
    for f in ("dlmm_sft_train.json", "dlmm_sft_val.json", "dlmm_test.json"):
        for r in json.load(open(f"/mnt/ssd1/vgrl/data/{f}")):
            if r["frame"] in seen:
                continue
            seen.add(r["frame"])
            tasks.append((r["frame"], r["views"], cache))
    print(f"고유 프레임 {len(tasks)}  (셀 {CELL_W}x{CELL_H})", flush=True)
    ok = 0; errs = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i, res in enumerate(ex.map(job, tasks, chunksize=8), 1):
            if res == 1: ok += 1
            else: errs.append(res)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}  성공 {ok}", flush=True)
    print(f"완료 {ok}/{len(tasks)}  실패 {len(errs)}")
    for e in errs[:5]: print("  ", e)


if __name__ == "__main__":
    main()
