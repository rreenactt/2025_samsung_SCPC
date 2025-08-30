#!/usr/bin/env python
# run_infer.py – Visual7W 4-지선다 제출 CSV 생성

import os, re, torch, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import bnb_patch                           # ① CB/SCB 패치
from load_blip2_lora import load_model     # ② 공용 로더
from transformers import Blip2Processor

CKPT_DIR      = "blip2_qformer_lora_8bit"  # ← 학습 결과 폴더
TEST_CSV_PATH = "open/test.csv"
OUT_CSV_PATH  = "open/submission.csv"
BATCH         = 4

# ── 1) 모델 + 프로세서 ───────────────────────────
model, tok_src = load_model(CKPT_DIR)
processor = Blip2Processor.from_pretrained(tok_src)
tok = processor.tokenizer

# 답안 토큰 ID (앞 공백!)
ans_token_ids = torch.tensor(
    tok.convert_tokens_to_ids([" A", " B", " C", " D"]),
    device=model.device,
)

# ── 2) 데이터 로드 ────────────────────────────────
df = pd.read_csv(TEST_CSV_PATH)  # 열: ID,img_path,Question,A,B,C,D
def build_prompt(row):
    return (
        f"Question: {row['Question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "Answer:"
    )

prompts = [build_prompt(r) for _, r in df.iterrows()]
images  = [Image.open(os.path.join("open", p)).convert("RGB")
           for p in df["img_path"]]

# ── 3) 배치 추론 ──────────────────────────────────
pred_letters = []
for i in tqdm(range(0, len(df), BATCH), desc="infer"):
    enc = processor(
        images = images[i:i+BATCH],
        text   = prompts[i:i+BATCH],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            pixel_values   = enc["pixel_values"],
        ).logits                               # (B,T,V)

    choice_logits = logits[:, -1, ans_token_ids]   # 마지막 토큰 → 4-way
    pred_letters.extend("ABCD"[j] for j in choice_logits.argmax(1).tolist())

# ── 4) CSV 저장 ───────────────────────────────────
pd.DataFrame({"ID": df["ID"], "answer": pred_letters})\
  .to_csv(OUT_CSV_PATH, index=False)
print(f"✅ submission saved → {OUT_CSV_PATH} (rows={len(df)})")
