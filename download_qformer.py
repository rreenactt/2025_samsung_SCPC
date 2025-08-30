from huggingface_hub import hf_hub_download

repo = "Salesforce/blip2-opt-2.7b"
local = "blip2_qformer_only"          # 현재 폴더 그대로

for name in [
    "pytorch_model-00001-of-00007.safetensors",
    "pytorch_model-00002-of-00007.safetensors",
    "pytorch_model.safetensors.index.json",
]:
    hf_hub_download(repo_id=repo, filename=name,
                    local_dir=local, local_dir_use_symlinks=False,
                    resume_download=True)      # 끊겨도 이어받기

print("✅ 3개 파일 모두 다운로드 완료")