'''
train_model_final.py

train.csv의 객관식 형식과 vitual7w.csv의 서술형 정답 형식을
모두 자동으로 처리할 수 있도록 VQADataset 클래스를 수정한 최종 버전.
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
    BASE_MODEL_ID = "Salesforce/blip-vqa-base"
    
    STAGE1_DATA_PATH = "open/vitual7w.csv"
    STAGE1_MODEL_SAVE_PATH = "./model_stage1_vitual7w"

    STAGE2_DATA_PATH = "open/train.csv"
    FINAL_MODEL_SAVE_PATH = "./final_vqa_model"

    IMAGE_BASE_PATH = "open"
    
    STAGE1_TRAINING_ARGS = {
        "output_dir": "./stage1_checkpoints",
        "seed": SEED,
        "num_train_epochs": 1,
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "save_strategy": "steps",
        "save_steps": 500,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "logging_steps": 50,
        "remove_unused_columns": False,
        "fp16": True,
    }
    STAGE2_TRAINING_ARGS = {
        "output_dir": "./stage2_checkpoints",
        "seed": SEED,
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "logging_steps": 20,
        "remove_unused_columns": False,
        "fp16": True,
    }

# --- 2. 데이터셋 클래스 정의 ---
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, processor, image_base_path):
        self.df = dataframe
        self.processor = processor
        self.image_base_path = image_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_base_path, str(row['img_path']).lstrip('./'))
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new('RGB', (self.processor.image_processor.size['height'], self.processor.image_processor.size['width']))
            
        question = str(row['Question'])
        
        # 📌 train.csv 형식(A,B,C,D 컬럼 존재)과 vitual7w 형식을 모두 처리하는 로직
        if 'A' in row and 'answer' in row and row['answer'] in ['A', 'B', 'C', 'D']:
            # train.csv 형식: answer 컬럼의 알파벳으로 정답 텍스트를 찾아옴
            correct_option_letter = str(row['answer']).strip().upper()
            answer_text = str(row[correct_option_letter])
        else:
            # vitual7w.csv 형식: answer 컬럼에 이미 정답 텍스트가 있음
            answer_text = str(row['answer'])
        
        encoding = self.processor(images=image, text=question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer(answer_text, padding="max_length", truncation=True, return_tensors="pt").input_ids

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = labels.squeeze()
        return encoding

# --- 3. 파라미터 출력 함수 ---
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(
        f"총 파라미터: {all_param/1e6:.2f}M | "
        f"훈련 가능 파라미터: {trainable_params/1e6:.2f}M | "
        f"훈련 가능 비율: {100 * trainable_params / all_param:.2f}%"
    )

# --- 4. 메인 실행 함수 ---
def main():
    processor = BlipProcessor.from_pretrained(CONFIG.BASE_MODEL_ID)
    
    # --- 1단계: 중간 학습 ---
    print("--- 1단계 시작: 중간 학습 ---")
    model_s1 = BlipForQuestionAnswering.from_pretrained(CONFIG.BASE_MODEL_ID)
    print_trainable_parameters(model_s1)

    df_s1 = pd.read_csv(CONFIG.STAGE1_DATA_PATH)
    train_df_s1 = df_s1.sample(frac=0.9, random_state=CONFIG.SEED)
    eval_df_s1 = df_s1.drop(train_df_s1.index)
    
    train_dataset_s1 = VQADataset(train_df_s1, processor, CONFIG.IMAGE_BASE_PATH)
    eval_dataset_s1 = VQADataset(eval_df_s1, processor, CONFIG.IMAGE_BASE_PATH)

    training_args_s1 = TrainingArguments(**CONFIG.STAGE1_TRAINING_ARGS)
    trainer_s1 = Trainer(
        model=model_s1,
        args=training_args_s1,
        train_dataset=train_dataset_s1,
        eval_dataset=eval_dataset_s1,
    )
    trainer_s1.train()
    trainer_s1.save_model(CONFIG.STAGE1_MODEL_SAVE_PATH)
    processor.save_pretrained(CONFIG.STAGE1_MODEL_SAVE_PATH)
    del model_s1, trainer_s1, df_s1, train_df_s1, eval_df_s1
    torch.cuda.empty_cache()

    # --- 2단계: 최종 미세 조정 ---
    print("\n--- 2단계 시작: 최종 미세 조정 ---")
    model_s2 = BlipForQuestionAnswering.from_pretrained(CONFIG.STAGE1_MODEL_SAVE_PATH)
    print_trainable_parameters(model_s2)

    df_s2 = pd.read_csv(CONFIG.STAGE2_DATA_PATH)
    train_df_s2 = df_s2.sample(frac=0.9, random_state=CONFIG.SEED)
    eval_df_s2 = df_s2.drop(train_df_s2.index)

    train_dataset_s2 = VQADataset(train_df_s2, processor, CONFIG.IMAGE_BASE_PATH)
    eval_dataset_s2 = VQADataset(eval_df_s2, processor, CONFIG.IMAGE_BASE_PATH)
    
    training_args_s2 = TrainingArguments(**CONFIG.STAGE2_TRAINING_ARGS)
    trainer_s2 = Trainer(
        model=model_s2,
        args=training_args_s2,
        train_dataset=train_dataset_s2,
        eval_dataset=eval_dataset_s2,
    )
    trainer_s2.train()
    trainer_s2.save_model(CONFIG.FINAL_MODEL_SAVE_PATH)
    processor.save_pretrained(CONFIG.FINAL_MODEL_SAVE_PATH)
    
    print("\n--- 2단계 미세 조정 완료! ---")
    print(f"최종 모델 저장 경로: {CONFIG.FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()