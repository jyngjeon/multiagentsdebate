import json
import os
import argparse

def clean_json_file(input_file_path: str, output_file_path: str, add_question_id: bool = False): # <-- add_question_id 인자 추가
    """
    CIAR.json 파일에서 'explanation', 'incorrect answer', 'incorrect explanation' 키를 제거하고,
    선택적으로 문제 번호(question_id)를 추가합니다.

    Args:
        input_file_path (str): 원본 JSON 파일의 경로 (예: CIAR.json).
        output_file_path (str): 정리된 내용을 저장할 새 JSON 파일의 경로.
        add_question_id (bool): True이면 각 문제에 1부터 시작하는 question_id를 추가합니다.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_file_path}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_file_path}': {e}")
        return

    cleaned_data = []
    keys_to_remove = ["explanation", "incorrect answer", "incorrect explanation"]

    for i, item in enumerate(data): # <-- enumerate를 사용하여 인덱스 i를 가져옵니다.
        cleaned_item = item.copy() # 원본 아이템 복사
        
        for key in keys_to_remove:
            if key in cleaned_item:
                del cleaned_item[key] # 해당 키 제거
        
        if add_question_id: # <-- add_question_id가 True일 경우에만 실행
            # question_id는 1부터 시작하도록 i + 1을 사용합니다.
            cleaned_item['question_id'] = i + 1 
            
        cleaned_data.append(cleaned_item)

    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully cleaned data and saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving cleaned data to '{output_file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CIAR.json by removing explanation and incorrect answer/explanation keys.")
    parser.add_argument("-i", "--input-file", type=str, required=True,
                        help="Path to the original CIAR JSON file.")
    parser.add_argument("-o", "--output-file", type=str, required=True,
                        help="Path to save the cleaned JSON file.")
    parser.add_argument("--add-question-id", action="store_true", # <-- 새로운 인자 추가
                        help="Add a 'question_id' field to each question, starting from 1.")
    
    args = parser.parse_args()
    
    clean_json_file(args.input_file, args.output_file, args.add_question_id) # <-- 함수 호출 시 인자 전달