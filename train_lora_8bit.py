#!/usr/bin/env python
# train_lora_8bit.py  –  BLIP-2 slim + Q-Former LoRA  (Windows / CUDA 11.8)

# ── 필수 설치(이미 완료했다면 생략) ─────────────────────────────
# pip install torch==2.4.0+cu118 torchvision torchaudio \
#       --index-url https://download.pytorch.org/whl/cu118
# pip install transformers==4.41.2 peft==0.11.1 bitsandbytes==0.43.1 \
#       accelerate==0.27.2 datasets pillow tqdm --upgrade
# ────────────────────────────────────────────────────────────────

import os, re, types, warnings, torch
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset

# ═════════════ 0. bitsandbytes 패치 ═════════════
import bitsandbytes as bnb

# ★ FIX 1 : 먼저 원본 forward 저장
_old_linear_fwd = bnb.nn.modules.Linear8bitLt.forward

def _safe_sd(self, dst, prefix, keep_vars):
    """메타-텐서·SCB 저장 오류 방지"""
    w, b = self.weight, self.bias
    dst[prefix + "weight"] = (
        w if keep_vars else torch.empty(0, dtype=w.dtype)
        if w.is_meta else w.detach().cpu()
    )
    if b is not None:
        dst[prefix + "bias"] = (
            b if keep_vars else torch.empty(0, dtype=b.dtype)
            if b.is_meta else b.detach().cpu()
        )

def _ensure_cb(self, x, *a, **kw):
    """CB / SCB 누락 시 안전 생성"""
    if not hasattr(self.weight, "CB"):
        rows = int(self.weight.shape[0])
        n_grp = max(1, (rows + 255) // 256)       # ★ FIX 2 – ceil 256
        z32 = torch.zeros(n_grp, dtype=torch.int32, device=self.weight.device)
        self.weight.CB  = z32.clone()
        self.weight.SCB = z32.clone()
    return _old_linear_fwd(self, x, *a, **kw)

bnb.nn.modules.Linear8bitLt.forward               = _ensure_cb
bnb.nn.modules.Linear8bitLt._save_to_state_dict   = _safe_sd
if hasattr(bnb.nn.modules, "Linear8bitLtTP"):
    bnb.nn.modules.Linear8bitLtTP.forward         = _ensure_cb
    bnb.nn.modules.Linear8bitLtTP._save_to_state_dict = _safe_sd
# ════════════════════════════════════════════════

from transformers import (
    Blip2ForConditionalGeneration, BitsAndBytesConfig,
    AutoTokenizer, BlipImageProcessor, Blip2Processor,
)

# ── inputs_embeds 충돌 제거 ─────────────────────
_orig_blip_fwd = Blip2ForConditionalGeneration.forward
def _blip_fwd_ignore_embeds(self, *a, **kw):
    kw.pop("inputs_embeds", None)
    return _orig_blip_fwd(self, *a, **kw)
Blip2ForConditionalGeneration.forward = _blip_fwd_ignore_embeds
# ────────────────────────────────────────────────

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 기본 설정
BASE = "blip2_vitl14_opt13b_slim"
OUT  = "blip2_qformer_lora_8bit"; os.makedirs(OUT, exist_ok=True)
BATCH, LR, MAX_ROWS = 2, 5e-5, 5_000          # 전체 학습은 MAX_ROWS=None

# 2. 8-bit 모델 로드
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["vision_model"],
)
model = Blip2ForConditionalGeneration.from_pretrained(
    BASE, device_map="auto", torch_dtype=torch.float16,
    quantization_config=bnb_cfg, low_cpu_mem_usage=True,
)

# 필요하다면 체크포인팅 해제해 속도 ↑ / VRAM ↓
model.gradient_checkpointing_disable()

model = prepare_model_for_kbit_training(model)

# ★ FIX 3 : lm_head 8-bit → FP16 고정(학습 제외)
with torch.no_grad():
    old = model.language_model.lm_head          # 8-bit Linear
    fp32_head = torch.nn.Linear(                # ← fp16 대신 fp32
        old.in_features, old.out_features,
        bias=(old.bias is not None)
    ).to(torch.float32).to(old.weight.device)   # ★ dtype=float32
    fp32_head.weight.copy_(old.weight.float())  # float() → fp32
    if old.bias is not None:
        fp32_head.bias.copy_(old.bias.float())
    fp32_head.weight.requires_grad = False
    if fp32_head.bias is not None:
        fp32_head.bias.requires_grad = False
    model.language_model.lm_head = fp32_head

# 3. LoRA (Q-Former 층)
l_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "query_key_value"],
    bias="none", task_type="SEQ_CLS",
)
model = get_peft_model(model, l_cfg)
model.print_trainable_parameters()

# 4. Visual7W 스트리밍 파서
stream = (
    load_dataset("HuggingFaceM4/the_cauldron", "visual7w",
                 split="train", streaming=True)
    .shuffle(seed=42, buffer_size=10_000)
)

blank = Image.new("RGB", (224, 224), "white")
choice_re = re.compile(r"^[ABCD]\.\s*(.+)", re.M)
ans_re    = re.compile(r"Answer:\s*([ABCD])", re.I)

def iter_rows():
    seen = 0
    for row in tqdm(stream, desc="flattening visual7w"):
        img0 = row.get("images", [blank])[0] or blank
        # ★ FIX 4 : RGB 변환 한 번만
        pil_base = img0 if img0.mode == "RGB" else img0.convert("RGB")

        for u, a in zip(row["texts"][0::2], row["texts"][1::2]):
            qblk, ablk = u["user"], a["assistant"]

            qm = re.search(r"Question:\s*(.+?)\nChoices:", qblk, re.S)
            if not qm:
                continue
            q = qm.group(1).strip()

            choices = choice_re.findall(qblk.split("Choices:")[-1])
            if len(choices) != 4:
                continue

            m = ans_re.search(ablk)
            if not m:
                continue
            label = "ABCD".index(m.group(1).upper())

            prompt = (
                f"Question: {q}\n" +
                "\n".join(f"{c}. {t}" for c, t in zip("ABCD", choices)) +
                "\nAnswer:"
            )
            yield pil_base, prompt, label
            seen += 1
            if MAX_ROWS and seen >= MAX_ROWS:
                return

class QADataset(torch.utils.data.IterableDataset):
    def __iter__(self): return iter_rows()

print(f"\n✓ streaming dataset ready — target = {MAX_ROWS if MAX_ROWS else 'ALL'}")
dataset = QADataset()

# 5. 프로세서
tok  = AutoTokenizer.from_pretrained(BASE, use_fast=False)
imgp = BlipImageProcessor.from_pretrained(BASE)
processor = Blip2Processor(tokenizer=tok, image_processor=imgp)

def collate(batch):
    imgs, prompts, labels = zip(*batch)
    enc = processor(images=list(imgs), text=list(prompts),
                    return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # letter 토큰 id (‘ĠA’, ‘ĠB’ …) 4개 미리 계산
    if not hasattr(collate, "ans_ids"):
        letters = [" A", " B", " C", " D"]           # 앞에 공백!
        collate.ans_ids = torch.tensor(
            [tok.encode(l)[-1] for l in letters], device=model.device
        )

    enc["labels"] = torch.tensor(labels, device=model.device)     # (B,)
    return enc

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH, collate_fn=collate
)
# 6. 학습 ─────────────────────────────────────────────
ce  = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

for step, batch in enumerate(loader):
    # ⬇️ pixel_values 추가!
    logits = model(
        pixel_values   = batch["pixel_values"],     # ← 이미지
        input_ids      = batch["input_ids"],
        attention_mask = batch["attention_mask"],
    ).logits                                   # (B, S, V)

    last_logits   = logits[:, -1, :]                 # (B, V)
    choice_logits = last_logits[:, collate.ans_ids]  # (B, 4)

    loss = ce(choice_logits, batch["labels"])
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step and step % 200 == 0:
        print(f"[step {step:>5}]  loss = {loss.item():.4f}")

# 7. 저장
model.save_pretrained(OUT)
processor.save_pretrained(OUT)
print("\n✅ LoRA adapter saved ➜", OUT)
