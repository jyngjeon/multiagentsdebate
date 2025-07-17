import json
import re
import argparse
import os
import random
from datasets import load_dataset

def extract_boxed_answer(solution_text: str) -> str:
    """
    주어진 풀이 텍스트에서 \\boxed{...} 안의 원본 내용을 그대로 추출합니다.
    중첩된 괄호를 올바르게 처리하여 내용이 잘리는 문제를 해결합니다.
    박스 안의 답을 찾지 못하면 None을 반환합니다.
    """
    # \boxed{ 시작 태그를 찾습니다.
    start_tag = r'\boxed{'
    start_index = solution_text.find(start_tag)
    if start_index == -1:
        return None

    # 실제 내용이 시작되는 인덱스
    content_start_index = start_index + len(start_tag)
    
    # 중첩된 괄호의 깊이를 추적하기 위한 카운터
    brace_level = 1
    
    # 내용 시작 지점부터 문자열을 스캔합니다.
    for i in range(content_start_index, len(solution_text)):
        char = solution_text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
        
        # 카운터가 0이 되면, 가장 바깥쪽 \boxed{}의 짝이 맞는 닫는 괄호를 찾은 것입니다.
        if brace_level == 0:
            content_end_index = i
            # 시작 태그 바로 다음부터, 짝이 맞는 닫는 괄호까지의 내용을 추출합니다.
            return solution_text[content_start_index:content_end_index].strip()
            
    # 문자열 끝까지 스캔했지만 짝이 맞는 닫는 괄호를 찾지 못한 경우 (형식 오류)
    return None

def prepare_level5_math_questions(output_file_path: str, total_questions_to_sample: int = 100):
    """
    hendrycks_math 데이터셋의 'test' 스플릿에서 "Level 5" 문제만 로드합니다.
    다양한 카테고리에서 문제를 추출하고, \\boxed{} 안의 답을 파싱합니다.
    요청된 수만큼의 문제를 샘플링하여 JSON 파일로 저장합니다.

    Args:
        output_file_path (str): 결과 .json 파일을 저장할 경로.
        total_questions_to_sample (int): 샘플링할 총 문제 수.
    """
    processed_questions = []
    
    # Hendrycks MATH 데이터셋의 사용 가능한 카테고리(configs)
    available_configs = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    all_raw_loaded_items = []
    
    print("--- Loading 'hendrycks_math' dataset categories (TEST SPLIT) ---")
    for config_name in available_configs:
        try:
            print(f"Loading config: '{config_name}' (test split)")
            dataset_split_list = load_dataset("EleutherAI/hendrycks_math", config_name, split="test").to_list()
            
            for item in dataset_split_list:
                item['category'] = config_name 
                all_raw_loaded_items.append(item)

            print(f"Loaded {len(dataset_split_list)} examples from '{config_name}'.")
        except Exception as e:
            print(f"Error loading config '{config_name}': {e}. Skipping this config.")
            
    if not all_raw_loaded_items:
        print("No dataset items loaded. Please check dataset ID, internet connection, or available configs. Exiting.")
        return

    print(f"\nTotal examples loaded from all categories: {len(all_raw_loaded_items)}")

    # --- 필터링 로직: "Level 5" 문제 중 유효한 답이 있는 것만 선택 ---
    print("--- Filtering for 'Level 5' questions with valid boxed answers ---")
    valid_level5_questions = []
    
    for item in all_raw_loaded_items:
        # "Level 5"가 아닌 문제는 건너뜀
        if item.get("level") != "Level 5":
            continue 

        extracted_answer = extract_boxed_answer(item.get("solution"))
        
        if extracted_answer:
            valid_level5_questions.append({
                "raw_item": item,
                "extracted_answer": extracted_answer
            })
        else:
            # 경고 메시지: 답을 추출할 수 없는 경우
            problem_id_display = item.get('id', item.get('problem', '')[:40].replace('\n', ' '))
            print(f"Warning: Could not extract \\boxed{{}} answer for problem '{problem_id_display}' (Cat: {item.get('category', 'N/A')}). Skipping.")

    if not valid_level5_questions:
        print("No valid 'Level 5' questions with extractable answers found. Exiting.")
        return

    print(f"\nFound {len(valid_level5_questions)} valid 'Level 5' questions across all categories.")
    print(f"--- Sampling a total of {total_questions_to_sample} questions ---")

    # 전체 유효한 Level 5 질문 중에서 무작위로 샘플링
    if len(valid_level5_questions) > total_questions_to_sample:
        final_sampled_items = random.sample(valid_level5_questions, total_questions_to_sample)
        print(f"Successfully sampled {total_questions_to_sample} questions.")
    else:
        # 샘플링할 문제 수가 전체 유효 문제 수보다 적거나 같으면 모두 사용
        final_sampled_items = valid_level5_questions
        print(f"Warning: Using all {len(final_sampled_items)} available questions (less than or equal to {total_questions_to_sample} requested).")
    
    # 최종 결과 리스트 구성
    question_counter = 0 
    for entry in final_sampled_items:
        question_counter += 1
        original_item = entry["raw_item"]
        extracted_answer = entry["extracted_answer"]

        processed_questions.append({
            "question_id": question_counter,
            "question": original_item.get("problem"),
            "level": original_item.get("level"),
            "category": original_item.get("category"),
            "answer": [extracted_answer]
        })

    # 처리된 질문들을 JSON 파일로 저장
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_questions, outfile, ensure_ascii=False, indent=4)
    print(f"\nSuccessfully created '{output_file_path}' with {len(processed_questions)} 'Level 5' questions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sampled 'Level 5' questions from hendrycks_math dataset.")
    parser.add_argument("-o", "--output-json", type=str, required=True,
                        help="Path to save the new .json file (e.g., data/math/math_level5_sampled_100.json).")
    parser.add_argument("-n", "--total-samples", type=int, default=100,
                        help="Total number of 'Level 5' questions to sample. Default 100.")
    
    args = parser.parse_args()
    
    prepare_level5_math_questions(args.output_json, args.total_samples)
