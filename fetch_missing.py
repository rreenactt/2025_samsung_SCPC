from huggingface_hub import hf_hub_download

repo  = "Salesforce/blip2-opt-2.7b"
local = "blip2_qformer_only"          # 폴더 그대로

# 바뀐 파일명 3개
for fname in [
    "model-00001-of-00002.safetensors",   # Q-Former + ViT 일부
    "model-00002-of-00002.safetensors",   # OPT-2.7B (우린 안 쓸 거지만 index가 요구)
    "model.safetensors.index.json",       # 2-샤드 인덱스
]:
    hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=local,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

print("✅ 새 샤드 2개 + index.json 다운로드 완료")
