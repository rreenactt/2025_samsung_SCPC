'''
train_itm_model_final.py

ITMCustomTrainer의 compute_loss 함수 시그니처 오류를 수정한 최종 버전.
'''
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    TrainingArguments,
    Trainer
)
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# --- 1. 설정 (Configuration) ---
class CONFIG:
    SEED = 42
    BASE_MODEL_ID = "Salesforce/blip-itm-base-coco"
    
    STAGE2_DATA_PATH = "open/train.csv"
    FINAL_MODEL_SAVE_PATH = "./final_itm_model"

    IMAGE_BASE_PATH = "open"
    
    STAGE2_TRAINING_ARGS = {
        "output_dir": "./stage2_checkpoints_itm", 
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
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }

# 📌 1. compute_loss 함수의 인수에 **kwargs를 추가합니다.
class ITMCustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        itm_logits = outputs.itm_score
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(itm_logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            itm_logits = outputs.itm_score
            loss = CrossEntropyLoss()(itm_logits, labels)
        
        return (loss, itm_logits, labels)

# --- 2. ITM 학습용 데이터셋 클래스 ---
class ITMDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, processor, image_base_path):
        self.processor = processor
        self.image_base_path = image_base_path
        self.samples = []

        print("긍정/부정 학습 샘플 생성 중...")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            question = str(row['Question'])
            
            correct_letter = str(row.get('answer')).strip().upper()
            if correct_letter not in ['A', 'B', 'C', 'D']: continue
            correct_text = str(row.get(correct_letter))
            
            self.samples.append({"img_path": str(row['img_path']), "text": f"{question} {correct_text}", "label": 1})
            
            for letter in ['A', 'B', 'C', 'D']:
                if letter != correct_letter:
                    wrong_text = str(row.get(letter))
                    self.samples.append({"img_path": str(row['img_path']), "text": f"{question} {wrong_text}", "label": 0})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_base_path, sample["img_path"].lstrip('./'))
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            return None
        
        encoding = self.processor(
            images=image, 
            text=sample["text"], 
            padding="max_length",
            truncation=True, 
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(sample["label"])
        return encoding

# --- 3. 헬퍼 함수 ---
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {all_param/1e6:.2f}M | 훈련 가능 파라미터: {trainable_params/1e6:.2f}M | 훈련 가능 비율: {100 * trainable_params / all_param:.2f}%")

# --- 4. 메인 실행 함수 ---
def main():
    processor = BlipProcessor.from_pretrained(CONFIG.BASE_MODEL_ID)
    
    print("\n--- ITM 모델 미세 조정 시작 ---")
    
    model = BlipForImageTextRetrieval.from_pretrained(
        CONFIG.BASE_MODEL_ID,
        use_safetensors=True,
    )
    print_trainable_parameters(model)

    df_s2 = pd.read_csv(CONFIG.STAGE2_DATA_PATH)
    train_df_s2 = df_s2.sample(frac=0.9, random_state=CONFIG.SEED)
    eval_df_s2 = df_s2.drop(train_df_s2.index)

    train_dataset_s2 = ITMDataset(train_df_s2, processor, CONFIG.IMAGE_BASE_PATH)
    eval_dataset_s2 = ITMDataset(eval_df_s2, processor, CONFIG.IMAGE_BASE_PATH)
    
    training_args_s2 = TrainingArguments(**CONFIG.STAGE2_TRAINING_ARGS)
    
    trainer_s2 = ITMCustomTrainer(
        model=model,
        args=training_args_s2,
        train_dataset=train_dataset_s2,
        eval_dataset=eval_dataset_s2,
    )
    trainer_s2.train()
    
    print(f"최종 모델 저장 중: {CONFIG.FINAL_MODEL_SAVE_PATH}")
    trainer_s2.save_model(CONFIG.FINAL_MODEL_SAVE_PATH)
    processor.save_pretrained(CONFIG.FINAL_MODEL_SAVE_PATH)
    
    print("\n--- ITM 모델 미세 조정 완료! ---")
    print(f"최종 모델 저장 경로: {CONFIG.FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()