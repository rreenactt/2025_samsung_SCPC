#!/usr/bin/env python
"""
make_submission_slim.py
────────────────────────────────────────────────────────
입력  : test.csv(id,image_path,question,A,B,C,D)
출력  : submission.csv(id,answer)
모델  : blip2_vitl14_opt13b_slim (8-bit)
"""
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear8bitLt, get_some_tensors


# 1) monkey-patch Linear8bitLt._save_to_state_dict
def _patched_save_to_state_dict(self, destination, prefix, keep_vars):
    scb_name = "SCB"
    if not hasattr(self.weight, scb_name):
        setattr(self.weight, scb_name, get_some_tensors(self.weight)[1])  # create dummy SCB
    return super(Linear8bitLt, self)._save_to_state_dict(destination, prefix, keep_vars)

Linear8bitLt._save_to_state_dict = _patched_save_to_state_dict

import csv, torch, os
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
)

# ───────── 경로 설정 ────────────────────────────────────────────────────
MODEL_PATH = "blip2_vitl14_opt13b_slim"
TEST_CSV   = "open/test.csv"
SUBMIT_CSV = "open/submission.csv"

# ───────── 모델 & 프로세서 로드 ────────────────────────────────────────
print("🔄  Loading slim BLIP-2 …")
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_module_state_dict=True
)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_cfg,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=True,
)
processor = Blip2Processor.from_pretrained(MODEL_PATH)
model.eval()

# ───────── 예측 함수 ────────────────────────────────────────────────────
@torch.inference_mode()
def predict_letter(img_path, question, choices):
    prompt = (
        f"Question: {question}\n"
        + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", choices))
        + "\nAnswer:"
    )
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_length=2, do_sample=False)[0]
    out     = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    return next((ch for ch in "ABCD" if ch in out), "A")

# ───────── CSV 루프 ────────────────────────────────────────────────────
print("📝  Predicting test set …")
with open(TEST_CSV, newline="", encoding="utf-8") as fin, \
     open(SUBMIT_CSV, "w", newline="", encoding="utf-8") as fout:

    reader  = csv.DictReader(fin)
    writer  = csv.writer(fout)
    writer.writerow(["id", "answer"])

    for row in tqdm(reader, total=sum(1 for _ in open(TEST_CSV))-1):
        pred = predict_letter(
            row["image_path"],
            row["question"],
            [row["A"], row["B"], row["C"], row["D"]],
        )
        writer.writerow([row["id"], pred])

print(f"✅  Done! → {SUBMIT_CSV}")
