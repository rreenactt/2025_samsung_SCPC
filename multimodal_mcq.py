'''
multimodal_mcq.py

Reads a test.csv file, predicts the answer for each row using the
BLIP ITM scoring method, and saves the results to submission.csv.
'''
import torch
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    BitsAndBytesConfig
)
from PIL import Image
import argparse
import sys
import pandas as pd
import os
from tqdm import tqdm

def load_model(use_8bit=True): # 8비트 사용을 기본값으로 설정
    """
    Loads the BLIP model fine-tuned for Image-Text Matching (ITM).
    """
    model_id = "Salesforce/blip-itm-base-coco"
    
    processor = BlipProcessor.from_pretrained(model_id)
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None
    
    model = BlipForImageTextRetrieval.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        use_safetensors=True
    )
    return model, processor

def predict_mcq(
    image_path, question, options,
    model, processor
):
    """
    Performs multiple-choice VQA by scoring each option's relevance to the image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        # 이미지 경로가 없을 경우를 대비해 None을 반환
        print(f"Warning: Image not found at {image_path}", file=sys.stderr)
        return None
    
    device = model.device
    best_option = ''
    highest_score = -float('inf')

    # Each option is iterated over to calculate its score
    for key, value in options.items():
        text_prompt = f"{question} {value}"
        
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        score = outputs.itm_score[0][1].item()

        if score > highest_score:
            highest_score = score
            best_option = key
            
    return best_option

def main():
    # 파일 경로 설정
    input_csv_path = 'open/test.csv'
    output_csv_path = 'submission.csv'
    image_base_path = 'open'

    print("Loading BLIP ITM model...")
    model, processor = load_model(use_8bit=True)

    print(f"Reading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    results = []

    print("Predicting answers for each row...")
    # tqdm을 사용하여 진행 상황을 시각적으로 표시
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        img_id = row['ID']
        # CSV의 img_path와 기본 경로를 조합하여 전체 이미지 경로 생성
        # 예: 'open' + './test_input_images/TEST_000.jpg' -> 'open/test_input_images/TEST_000.jpg'
        img_path = os.path.join(image_base_path, row['img_path'].lstrip('./'))
        question = row['Question']
        options = {
            'A': row['A'],
            'B': row['B'],
            'C': row['C'],
            'D': row['D']
        }

        answer = predict_mcq(
            image_path=img_path,
            question=question,
            options=options,
            model=model,
            processor=processor
        )

        results.append({
            'ID': img_id,
            'answer': answer
        })

    print("Saving results to CSV...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Successfully created submission file at: {output_csv_path}")


if __name__ == "__main__":
    main()