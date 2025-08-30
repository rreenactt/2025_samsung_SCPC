'''
train_stage2_only.py

1단계 학습이 완료된 모델을 불러와 2단계 최종 미세 조정만 수행하는 코드.
'''
import torch
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    TrainingArguments,
    Trainer
)
from PIL import Image
import pandas as pd
import os

# --- 1. 설정 (Configuration) ---
class CONFIG:
    SEED = 42
    
    # 1단계 학습 완료 모델 경로
    STAGE1_MODEL_PATH = "./model_stage1_vitual7w"
    
    # 최종 모델을 저장할 경로
    FINAL_MODEL_SAVE_PATH = "./final_vqa_model"

    # 2단계 학습 데이터 경로
    STAGE2_DATA_PATH = "open/train.csv"
    IMAGE_BASE_PATH = "open"
    
    # 2단계 학습 파라미터
    STAGE2_TRAINING_ARGS = {
        "output_dir": "./stage2_checkpoints",
        "seed": SEED,
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "logging_steps": 20,
        "remove_unused_columns": False,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "dataloader_num_workers": os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
    }

# --- 2. 데이터셋 클래스 ---
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, processor, image_base_path):
        self.df = dataframe
        self.processor = processor
        self.image_base_path = image_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_base_path, str(row['img_path']).lstrip('./'))
            image = Image.open(img_path).convert("RGB")
            question = str(row['Question'])
            answer_text = str(row['answer_text'])

            encoding = self.processor(images=image, text=question, padding="max_length", truncation=True, return_tensors="pt")
            labels = self.processor.tokenizer(answer_text, padding="max_length", truncation=True, return_tensors="pt").input_ids
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": encoding.input_ids.squeeze(),
                "attention_mask": encoding.attention_mask.squeeze(),
                "pixel_values": encoding.pixel_values.squeeze(),
                "labels": labels.squeeze(),
            }
        except Exception:
            return None

# --- 3. 헬퍼 함수 ---
def get_answer_text(row):
    if 'A' in row and 'answer' in row and str(row['answer']).strip().upper() in ['A', 'B', 'C', 'D']:
        return str(row[str(row['answer']).strip().upper()])
    elif 'answer' in row:
        return str(row['answer'])
    return None

def filter_dataframe(df):
    original_len = len(df)
    df['answer_text'] = df.apply(get_answer_text, axis=1)
    df.dropna(subset=['answer_text', 'img_path', 'Question'], inplace=True)
    df = df[df['answer_text'].str.strip().astype(bool)].copy()
    print(f"결측값/빈 답변 제거 후: {original_len} -> {len(df)} 개")
    return df.reset_index(drop=True)

# --- 4. 메인 실행 함수 ---
def main():
    print("--- 2단계 시작: 최종 미세 조정 ---")
    
    processor = BlipProcessor.from_pretrained(CONFIG.STAGE1_MODEL_PATH)
    model = BlipForQuestionAnswering.from_pretrained(CONFIG.STAGE1_MODEL_PATH)

    df = pd.read_csv(CONFIG.STAGE2_DATA_PATH)
    clean_df = filter_dataframe(df)
    
    train_df = clean_df.sample(frac=0.9, random_state=CONFIG.SEED)
    eval_df = clean_df.drop(train_df.index)

    train_dataset = VQADataset(train_df, processor, CONFIG.IMAGE_BASE_PATH)
    eval_dataset = VQADataset(eval_df, processor, CONFIG.IMAGE_BASE_PATH)
    
    training_args = TrainingArguments(**CONFIG.STAGE2_TRAINING_ARGS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    
    print(f"최종 모델 저장 중: {CONFIG.FINAL_MODEL_SAVE_PATH}")
    trainer.save_model(CONFIG.FINAL_MODEL_SAVE_PATH)
    processor.save_pretrained(CONFIG.FINAL_MODEL_SAVE_PATH)
    
    print("\n--- 2단계 미세 조정 완료! ---")
    print(f"최종 모델 저장 경로: {CONFIG.FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()