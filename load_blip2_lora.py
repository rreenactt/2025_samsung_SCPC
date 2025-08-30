# load_blip2_lora.py
from pathlib import Path
import torch, importlib
from transformers import (
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

def load_model(ckpt_dir: str, dtype=torch.float16):
    """
    ckpt_dir이 LoRA 어댑터 폴더인지, 합쳐진 전체 모델인지 자동 판별 후 로드.
    bitsandbytes CB/SCB 패치를 사용하려면 `import bnb_patch` 먼저 하세요.
    """
    ckpt = Path(ckpt_dir)
    has_full_bin = any(f.suffix in {".bin", ".safetensors"} and f.stat().st_size > 9e9
                       for f in ckpt.glob("*"))
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    if has_full_bin:  # 전체 모델
        model = Blip2ForConditionalGeneration.from_pretrained(
            ckpt_dir,
            device_map="auto",
            quantization_config=bnb_cfg,
            torch_dtype=dtype,
        )
        tokenizer_dir = ckpt_dir
    else:             # 어댑터만
        peft_cfg = PeftConfig.from_pretrained(ckpt_dir)
        base = Blip2ForConditionalGeneration.from_pretrained(
            peft_cfg.base_model_name_or_path,
            device_map="auto",
            quantization_config=bnb_cfg,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, ckpt_dir)
        tokenizer_dir = peft_cfg.base_model_name_or_path

    model.eval()
    return model, tokenizer_dir
