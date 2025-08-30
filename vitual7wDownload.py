# download_dataset.py
from datasets import load_dataset
import os
import pandas as pd
import re
from tqdm import tqdm

def parse_entry(entry):
    """
    데이터셋의 한 행(entry)을 받아 질문, 선택지, 정답을 파싱하는 함수
    """
    try:
        text_data_list = entry.get('texts')
        if not text_data_list:
            return None

        first_turn = text_data_list[0]
        user_text = first_turn.get('user')
        assistant_text = first_turn.get('assistant')

        if not user_text or not assistant_text:
            return None

        # 정규식을 사용하여 각 부분을 추출
        question_match = re.search(r"Question:(.*?)\nChoices:", user_text, re.DOTALL)
        option_a_match = re.search(r"\nA\.\s(.*?)\nB\.", user_text, re.DOTALL)
        option_b_match = re.search(r"\nB\.\s(.*?)\nC\.", user_text, re.DOTALL)
        option_c_match = re.search(r"\nC\.\s(.*?)\nD\.", user_text, re.DOTALL)
        option_d_match = re.search(r"\nD\.\s(.*?)\nAnswer with the letter\.", user_text, re.DOTALL)
        answer_letter_match = re.search(r"Answer:\s*([A-D])", assistant_text)

        if not all([question_match, option_a_match, option_b_match, option_c_match, option_d_match, answer_letter_match]):
            return None

        question = question_match.group(1).strip()
        options = {
            'A': option_a_match.group(1).strip(),
            'B': option_b_match.group(1).strip(),
            'C': option_c_match.group(1).strip(),
            'D': option_d_match.group(1).strip(),
        }
        answer_letter = answer_letter_match.group(1).strip().upper()
        
        answer_text = options.get(answer_letter)

        if not answer_text:
            return None

        return {
            "question": question,
            "answer": answer_text,
            "image": entry['images'][0],
            "id": entry.get('image_id')
        }
    except (KeyError, IndexError, AttributeError):
        return None


def download_and_prepare_dataset():
    """
    Hugging Face Hub의 the_cauldron 데이터셋에서 visual7w 서브셋 전체를 다운로드하고,
    학습에 필요한 형식의 CSV 파일로 변환하여 저장합니다.
    """
    output_folder = "open"
    os.makedirs(output_folder, exist_ok=True)
    # 📌 테스트용 파일 이름(_test)을 원래대로 변경
    output_path = os.path.join(output_folder, "vitual7w.csv")

    print("HuggingFaceM4/the_cauldron에서 'visual7w' 서브셋 전체를 다운로드합니다...")
    
    try:
        dataset = load_dataset("HuggingFaceM4/the_cauldron", name="visual7w", split="train")
    except Exception as e:
        print(f"데이터셋 다운로드 중 오류가 발생했습니다: {e}")
        return

    processed_data = []
    print("데이터를 학습 형식에 맞게 파싱하고 변환 중입니다 (전체 데이터)...")
    
    # 📌 5개 제한 로직을 제거하여 전체 데이터셋을 순회하도록 함
    for index, entry in enumerate(tqdm(dataset, desc="Processing")):
        parsed = parse_entry(entry)
        
        if parsed:
            image_id = parsed.get('id') or f'v7w_{index}'
            image_folder = os.path.join(output_folder, 'v7w_images')
            os.makedirs(image_folder, exist_ok=True)
            
            image_path = os.path.join(image_folder, f"{image_id}.png")
            parsed['image'].save(image_path)
            
            relative_img_path = os.path.join('v7w_images', f"{image_id}.png").replace('\\', '/')

            processed_data.append({
                'ID': image_id,
                'img_path': relative_img_path,
                'Question': parsed['question'],
                'answer': parsed['answer']
            })

    final_df = pd.DataFrame(processed_data)
    
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n다운로드 및 변환 완료! 전체 파일이 '{output_path}'에 저장되었습니다.")
    print("이미지들은 'open/v7w_images' 폴더에 저장되었습니다.")
    print("\nCSV 샘플:")
    print(final_df.head())

if __name__ == "__main__":
    download_and_prepare_dataset()