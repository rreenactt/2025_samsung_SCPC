#!/usr/bin/env python
"""
build_slim_blip2_manual.py
────────────────────────────────────────────────────────
● 목적
  BLIP-2 OPT-2.7B 체크포인트에서 Q-Former(+조금의 ViT)만 가져와,
  • Vision : OpenCLIP ViT-L/14 (hidden 1024)
  • LLM    : OPT-1.3B
  로 새로 “슬림(≈1.73 B)” 모델을 조립해 저장한다.

● 전제
  blip2_qformer_only/
      ├─ model-00001-of-00002.safetensors   (Q-Former + ViT 일부)
      ├─ model.safetensors.index.json
      ├─ config.json, generation_config.json
      ├─ preprocessor_config.json
      └─ tokenizer.json …

● 설치
  pip install torch safetensors transformers open-clip-torch==2.24.0 timm

● 결과
  blip2_vitl14_opt13b_slim/
      ├─ config.json, generation_config.json, preprocessor_config.json …
      ├─ model-00001-of-00002.safetensors   (8-bit 양자화)
      └─ tokenizer.*  → 1.73 B 파라미터
"""

# ───────────────────────────────────── imports
import os, torch, safetensors.torch as sf
from transformers import (
    Blip2Config, Blip2ForConditionalGeneration, Blip2Processor,
    OPTForCausalLM, CLIPVisionConfig, BitsAndBytesConfig
)
from open_clip import create_model_and_transforms

# ───────────────────────────────────── paths
SRC_DIR  = "blip2_qformer_only"
SHARD    = os.path.join(SRC_DIR, "model-00001-of-00002.safetensors")
DEST_DIR = "blip2_vitl14_opt13b_slim"
os.makedirs(DEST_DIR, exist_ok=True)

# ─────────────────────────── 1. Load small LLM (OPT-1.3B)
print("🔄  Loading OPT-1.3B …")
lm_small = OPTForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    torch_dtype=torch.float16,
    use_safetensors=True,       # .safetensors만
    low_cpu_mem_usage=True
)

# ─────────────────────────── 2. Load ViT-L/14 backbone
print("🔄  Loading OpenCLIP ViT-L/14 …")
vit_l14, _, _ = create_model_and_transforms("ViT-L-14", "openai")
vision_model  = vit_l14.visual
vision_cfg    = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")

# ─────────────────────────── 3. Build empty BLIP-2 skeleton
print("🛠️   Building BLIP-2 skeleton …")
cfg = Blip2Config(
    vision_config  = vision_cfg.to_dict(),          # dict 형태 필요
    qformer_config = {"hidden_size": 768, "num_hidden_layers": 12},
    text_config    = lm_small.config.to_dict(),
)
model = Blip2ForConditionalGeneration(cfg)
model.vision_model   = vision_model
model.language_model = lm_small
model.lm_proj        = torch.nn.Linear(768, lm_small.config.hidden_size, bias=True)

# ─────────────────────────── 4. Inject Q-Former & ViT weights
print("🔄  Injecting pretrained Q-Former & ViT weights …")
state = sf.load_file(SHARD, device="cpu")

def copy_submodule(prefix: str, target):
    """
    prefix: 'qformer.' or 'vision_model.'
    weight shape가 일치하는 항목만 복사, 나머지는 초기화 유지
    """
    src = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    tgt_sd = target.state_dict()
    filtered = {k: v for k, v in src.items() if k in tgt_sd and v.shape == tgt_sd[k].shape}
    skipped  = [k for k in src if k not in filtered]
    tgt_sd.update(filtered)
    target.load_state_dict(tgt_sd, strict=False)
    print(f"    • {prefix[:-1]:<12}  copied {len(filtered):>4}  |  skipped {len(skipped):>2} (shape mismatch)")

copy_submodule("qformer.",      model.qformer)      # cross-attn key/value(1408→1024)는 skip
copy_submodule("vision_model.", model.vision_model)
print("✅  Weight injection done.")

# ─────────────────────────── 5. Freeze everything except Q-Former & lm_proj
for n, p in model.named_parameters():
    if not n.startswith(("qformer", "lm_proj")):
        p.requires_grad_(False)

# ─────────────────────────── 6. Save processor + 8-bit model
print("💾  Saving processor & 8-bit model …")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor.save_pretrained(DEST_DIR)

bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
model.save_pretrained(DEST_DIR, safe_serialization=True, quantization_config=bnb_cfg)

param_B = sum(p.numel() for p in model.parameters()) / 1e9
print(f"🎉  Done!  Slim checkpoint → {DEST_DIR}/   (params ≈ {param_B:.2f} B)")
