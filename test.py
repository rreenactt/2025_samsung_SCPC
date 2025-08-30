from transformers import Blip2ForConditionalGeneration, BitsAndBytesConfig
print("➡️  Salesforce/blip2-opt-13b 다운로드 시도…")
Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-13b",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"": "cpu"},      # GPU 메모리 안 쓰고 CPU에만 적재
)
print("✅ 다운로드 완료 – 캐시에 저장되었습니다")