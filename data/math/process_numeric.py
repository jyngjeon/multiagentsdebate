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
    start_tag = r'\boxed{'
    start_index = solution_text.find(start_tag)
    if start_index == -1:
        return None

    content_start_index = start_index + len(start_tag)
    brace_level = 1
    
    for i in range(content_start_index, len(solution_text)):
        char = solution_text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
        
        if brace_level == 0:
            content_end_index = i
            return solution_text[content_start_index:content_end_index].strip()
            
    return None

def is_pure_number(s: str) -> bool:
    """
    주어진 문자열이 쉼표가 없는 순수 숫자인지 (정수 또는 실수) 확인합니다.
    """
    if not isinstance(s, str):
        return False
    
    s = s.strip()
    if not s:
        return False
        
    # 쉼표가 포함된 경우는 다중 답변이나 서식 있는 숫자로 간주하여 제외
    if ',' in s:
        return False
        
    try:
        float(s)
        return True
    except ValueError:
        return False

def prepare_numeric_math_questions(output_file_path: str, total_questions_to_sample: int = 100):
    """
    hendrycks_math 데이터셋에서 답변이 단일 순수 숫자인 문제만 필터링합니다.
    다양한 카테고리에서 문제를 추출하고, 요청된 수만큼 샘플링하여 JSON 파일로 저장합니다.

    Args:
        output_file_path (str): 결과 .json 파일을 저장할 경로.
        total_questions_to_sample (int): 샘플링할 총 문제 수.
    """
    processed_questions = []
    
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
        print("No dataset items loaded. Exiting.")
        return

    print(f"\nTotal examples loaded from all categories: {len(all_raw_loaded_items)}")

    # --- 필터링 로직: 단일 순수 숫자 답변을 가진 문제만 선택 ---
    print("--- Filtering for questions with single, pure numeric answers ---")
    valid_numeric_questions = []
    
    for item in all_raw_loaded_items:
        extracted_answer = extract_boxed_answer(item.get("solution"))
        
        if extracted_answer and is_pure_number(extracted_answer):
            valid_numeric_questions.append({
                "raw_item": item,
                "extracted_answer": extracted_answer
            })
        
    if not valid_numeric_questions:
        print("No valid questions with single, pure numeric answers found. Exiting.")
        return

    print(f"\nFound {len(valid_numeric_questions)} valid questions with numeric answers across all categories.")
    print(f"--- Sampling a total of {total_questions_to_sample} questions ---")

    if len(valid_numeric_questions) > total_questions_to_sample:
        final_sampled_items = random.sample(valid_numeric_questions, total_questions_to_sample)
        print(f"Successfully sampled {total_questions_to_sample} questions.")
    else:
        final_sampled_items = valid_numeric_questions
        print(f"Warning: Using all {len(final_sampled_items)} available questions (less than or equal to {total_questions_to_sample} requested).")
    
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

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_questions, outfile, ensure_ascii=False, indent=4)
    print(f"\nSuccessfully created '{output_file_path}' with {len(processed_questions)} questions with numeric answers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sampled questions with single, pure numeric answers from the hendrycks_math dataset.")
    parser.add_argument("-o", "--output-json", type=str, required=True,
                        help="Path to save the new .json file (e.g., data/math/math_numeric_sampled_100.json).")
    parser.add_argument("-n", "--total-samples", type=int, default=100,
                        help="Total number of questions with numeric answers to sample. Default 100.")
    
    args = parser.parse_args()
    
    prepare_numeric_math_questions(args.output_json, args.total_samples)