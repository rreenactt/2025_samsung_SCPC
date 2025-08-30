# build_slim_blip2.py
import torch, os
from transformers import Blip2ForConditionalGeneration, Blip2Processor, OPTForCausalLM
from open_clip import create_model_and_transforms

BASE_ID   = "blip2_qformer_only"           # 방금 다운로드한 폴더
OPT_ID    = "facebook/opt-1.3b"            # 작은 LLM
SAVE_PATH = "blip2_vitl14_opt13b_slim"     # 출력 폴더

# 1. 전처리기
processor = Blip2Processor.from_pretrained(BASE_ID)

# 2. 원본 모델(Q-Former+ViT 일부) 로드
orig = Blip2ForConditionalGeneration.from_pretrained(
    BASE_ID, device_map="cpu", load_in_8bit=True,
    # OPT 샤드가 없으므로 shape 안 맞아도 무시
    ignore_mismatched_sizes=True,
)

# 3. ViT-L/14 백본으로 교체
vit_l14, _, _ = create_model_and_transforms("ViT-L-14", "openai")
orig.vision_model = vit_l14.visual

# 4. 작은 OPT-1.3B 로 교체
opt13 = OPTForCausalLM.from_pretrained(OPT_ID, torch_dtype=torch.float16)
orig.language_model   = opt13
orig.config.text_config = opt13.config

# 5. Q-Former → LLM 프로젝션 재생성
in_dim  = orig.qformer.config.hidden_size      # 768
out_dim = opt13.config.hidden_size             # 2048
orig.lm_proj = torch.nn.Linear(in_dim, out_dim, bias=True)

# 6. 파라미터 수 확인
total_B = sum(p.numel() for p in orig.parameters()) / 1e9
print(f"Total parameters ≈ {total_B:.2f} B (<3 B)")

# 7. 저장
os.makedirs(SAVE_PATH, exist_ok=True)
processor.save_pretrained(SAVE_PATH)
orig.save_pretrained(SAVE_PATH)
print(f"✅ Slim checkpoint saved to  {SAVE_PATH}/")
