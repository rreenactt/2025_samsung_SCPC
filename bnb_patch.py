# bnb_patch.py  (refresh)
"""
Windows bitsandbytes 8-bit CB/SCB 패치
  • _save_to_state_dict : 키워드·위치 인수 모두 지원
  • forward            : CB/SCB 없으면 더미 생성
"""
import bitsandbytes as bnb
import torch

# ────────────────────────────────────────────────
# 1) _save_to_state_dict 패치
# ────────────────────────────────────────────────
orig_save = bnb.nn.Linear8bitLt._save_to_state_dict

def _patched_save(self, *args, **kwargs):
    # ↳ (destination, prefix, keep_vars)  ← 위치 인수로 올 때가 많음
    if args:
        destination = args[0]
        prefix      = args[1] if len(args) > 1 else ""
    else:                       # 키워드 인수 fallback
        destination = kwargs["destination"]
        prefix      = kwargs.get("prefix", "")

    # SCB, CB 키가 없으면 빈 tensor 주입
    for k in ("SCB", "CB"):
        key = f"{prefix}weight.{k}"
        if key not in destination:
            destination[key] = torch.empty(0, dtype=torch.int8)

    return orig_save(self, *args, **kwargs)

bnb.nn.Linear8bitLt._save_to_state_dict = _patched_save

# ────────────────────────────────────────────────
# 2) forward 패치 (그대로)
# ────────────────────────────────────────────────
orig_fwd = bnb.nn.Linear8bitLt.forward

def _patched_fwd(self, x):
    if not hasattr(self.weight, "CB"):
        dev = self.weight.device
        self.weight.CB  = torch.zeros(1, dtype=torch.int32, device=dev)
        self.weight.SCB = torch.zeros(1, dtype=torch.int32, device=dev)
    return orig_fwd(self, x)

bnb.nn.Linear8bitLt.forward = _patched_fwd
