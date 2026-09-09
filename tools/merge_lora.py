"""LoRA 어댑터를 베이스에 병합해 vLLM 이 바로 읽을 수 있는 모델로 저장한다."""
import argparse, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None,
                help="생략하면 adapter_config 의 base_model_name_or_path 를 쓴다. "
                     "RL 어댑터는 SFT 병합본 위에서 학습되므로 원본 베이스에 얹으면 "
                     "SFT 학습분이 통째로 사라진다 (실측: MCQ 63.95% -> 51.57%).")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

if a.base is None:
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    a.base = cfg["base_model_name_or_path"]
print("base :", a.base)
print("adapter:", a.adapter)

m = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16,
                                                       device_map="cpu")
m = PeftModel.from_pretrained(m, a.adapter, device_map="cpu")
m = m.merge_and_unload()
os.makedirs(a.out, exist_ok=True)
m.save_pretrained(a.out, safe_serialization=True)
AutoProcessor.from_pretrained(a.base, trust_remote_code=True).save_pretrained(a.out)
# 어댑터 쪽 토크나이저/템플릿을 우선 반영
for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
    p = os.path.join(a.adapter, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(a.out, f))
print("병합 완료 ->", a.out)
