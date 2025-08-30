'''
inference_final_corrected.py

훈련된 모델 클래스(BlipForQuestionAnswering)와 일치하는 방식으로 추론을 수행하고,
생성된 답변을 선택지와 비교하여 최종 답을 결정하는 최종 스크립트.
'''
import torch
from transformers import (
    BlipForQuestionAnswering, # 📌 1. 훈련 시 사용했던 올바른 클래스로 변경
    BlipProcessor
)
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import sys

# --- 1. 설정 (Configuration) ---
class CONFIG:
    # 📌 2. 원본 모델 ID와 체크포인트 경로를 명확히 분리
    BASE_MODEL_ID = "Salesforce/blip-vqa-base"
    FINETUNED_MODEL_PATH = "./stage1_checkpoints/checkpoint-500" 
    
    TEST_CSV_PATH = "open/test.csv"
    SUBMISSION_CSV_PATH = "submission.csv"
    IMAGE_BASE_PATH = "open"

def load_model_and_processor(model_path, base_model_id):
    """
    프로세서는 원본 ID에서, 모델은 체크포인트 경로에서 올바른 클래스로 로드합니다.
    """
    print(f"프로세서 로딩: {base_model_id}")
    processor = BlipProcessor.from_pretrained(base_model_id)

    print(f"학습된 모델 로딩: {model_path}")
    model = BlipForQuestionAnswering.from_pretrained(model_path, device_map="auto")
    model.eval()
    return model, processor

# 📌 3. 예측 함수 로직을 '답변 생성 후 비교' 방식으로 변경
def predict_vqa_and_choose_option(
    image_path, question, options,
    model, processor
):
    """
    모델이 자유 답변을 생성하게 한 뒤, 선택지와 비교하여 가장 유사한 옵션을 선택합니다.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return "A" # 기본값

    device = model.device
    
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, num_beams=3)
    
    generated_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 생성된 답변과 가장 유사한 선택지 찾기 (단어 겹침 기준)
    best_option = "A" 
    max_overlap = -1

    for key, value in options.items():
        if generated_answer.lower() == str(value).lower():
            return key

    for key, value in options.items():
        option_words = set(str(value).lower().split())
        answer_words = set(generated_answer.lower().split())
        overlap = len(option_words.intersection(answer_words))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_option = key
            
    return best_option

def main():
    model, processor = load_model_and_processor(
        model_path=CONFIG.FINETUNED_MODEL_PATH,
        base_model_id=CONFIG.BASE_MODEL_ID
    )
    
    df_test = pd.read_csv(CONFIG.TEST_CSV_PATH)
    results = []

    print("추론을 시작합니다...")
    for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing"):
        img_id = row['ID']
        img_path = os.path.join(CONFIG.IMAGE_BASE_PATH, row['img_path'].lstrip('./'))
        question = row['Question']
        options = {
            'A': row['A'], 'B': row['B'],
            'C': row['C'], 'D': row['D']
        }

        answer = predict_vqa_and_choose_option(
            image_path=img_path,
            question=question,
            options=options,
            model=model,
            processor=processor
        )

        results.append({'ID': img_id, 'answer': answer})

    print("결과를 CSV 파일로 저장 중...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(CONFIG.SUBMISSION_CSV_PATH, index=False)
    print(f"성공적으로 제출 파일 생성: {CONFIG.SUBMISSION_CSV_PATH}")

if __name__ == "__main__":
    main()