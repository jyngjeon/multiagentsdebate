import json
import re
import argparse
import os
import random
from datasets import load_dataset

def extract_boxed_answer(solution_text: str) -> str:
    """
    Extracts the content within \\boxed{} from the solution text.
    Handles basic LaTeX math within \\boxed{} and attempts to sanitize it
    to a simple number or fraction string.
    Returns the extracted and sanitized string, or None if no boxed answer is found
    or if the content cannot be reasonably simplified to a number/fraction.
    """
    # Regex to find \boxed{...} and capture its content
    match = re.search(r'\\boxed{(.*?)}', solution_text, re.DOTALL)
    if match:
        extracted_content = match.group(1).strip()
        
        # Basic LaTeX math cleanup and numerical simplification
        cleaned_content = re.sub(r'\\text{.*?}|\\[a-zA-Z]+\b', '', extracted_content)
        cleaned_content = cleaned_content.replace(' ', '') 

        # Handle fractions: \frac{a}{b} -> (a)/(b)
        cleaned_content = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1)/(\2)', cleaned_content)
        
        # Handle multiplication: \times, \cdot -> *
        cleaned_content = cleaned_content.replace('\\times', '*').replace('\\cdot', '*')
        
        # Basic cleanup for common symbols
        cleaned_content = cleaned_content.replace('{', '').replace('}', '').strip()

        # Final attempt to extract a simple number or fraction from the cleaned string
        num_frac_match = re.search(r'([-+]?\d+\.?\d*(?:/\d+\.?\d*)?)', cleaned_content)
        if num_frac_match:
            return num_frac_match.group(0).strip()
        
        return None 
    return None

def prepare_math_questions(output_file_path: str, total_questions_to_sample: int = 100):
    """
    Loads the 'test' split of hendrycks_math dataset, extracts questions,
    answers from \\boxed{}, adds 'question_id', 'level', 'category', and 'answer' keys,
    and samples a total number of questions uniformly across all categories and levels.

    Args:
        output_file_path (str): Path to save the new .json file.
        total_questions_to_sample (int): The total number of questions to sample across all levels and categories.
    """
    processed_questions = [] # Final list of processed questions
    
    # Hendrycks MATH dataset available categories (configs)
    available_configs = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    # Levels to consider for sampling, matching the format "Level X" from the dataset.
    levels_to_consider = [f"Level {i}" for i in range(0, 6)] # Includes Level 0 to Level 5

    all_raw_loaded_items = [] # Temporary list to store all items loaded from all configs and splits
    
    print(f"--- Loading 'hendrycks_math' dataset categories and levels (TEST SPLIT) ---")
    for config_name in available_configs: # config_name is the category (e.g., 'algebra')
        try:
            print(f"Loading config: '{config_name}' (test split)")
            # --- 수정: 데이터셋 ID 변경 및 'test' split 로드 ---
            dataset_split_list = load_dataset("EleutherAI/hendrycks_math", config_name, split="test").to_list()
            
            for item in dataset_split_list:
                # --- 수정: 'category' 필드 추가 ---
                # 'item' 딕셔너리에 'category' 필드를 명시적으로 추가합니다.
                item['category'] = config_name 
                all_raw_loaded_items.append(item)

            print(f"Loaded {len(dataset_split_list)} examples from '{config_name}'.")
        except Exception as e:
            print(f"Error loading config '{config_name}': {e}. Skipping this config.")
            
    if not all_raw_loaded_items:
        print("No dataset items loaded from any category. Please check dataset ID, internet connection, or available configs. Exiting.")
        return

    print(f"Total examples loaded from all categories: {len(all_raw_loaded_items)}")

    # --- 샘플링 로직 재조정: 전체에서 무작위 샘플링 ---
    # 먼저, \boxed{} 답을 추출할 수 있는 유효한 질문만 필터링합니다.
    # 그리고, 원하는 난이도 레벨만 포함합니다.
    valid_questions_with_answers = []
    
    for item in all_raw_loaded_items:
        # 원하는 레벨만 포함 (여기서는 Level 1-5를 기본으로, 필요시 Level 0 포함 가능)
        if item.get("level") not in levels_to_consider: # 'Level 0' ~ 'Level 5'
            continue 

        extracted_answer = extract_boxed_answer(item.get("solution"))
        
        if extracted_answer:
            valid_questions_with_answers.append({
                "raw_item": item, # 원본 item을 저장하여 정보 손실 방지
                "extracted_answer": extracted_answer
            })
        else:
            problem_id_display = item.get('id', item.get('problem', '')[:40].replace('\n', ' '))
            print(f"Warning: Could not extract \\boxed{{}} answer for problem '{problem_id_display}' (Cat: {item.get('category', 'N/A')}, Level: {item.get('level', 'N/A')}). Skipping.")

    if not valid_questions_with_answers:
        print("No valid questions with extractable answers found after filtering. Exiting.")
        return

    print(f"\n--- Sampling a total of {total_questions_to_sample} questions from {len(valid_questions_with_answers)} valid questions ---")

    # 전체 유효 질문 중에서 무작위로 total_questions_to_sample 개수를 샘플링
    if len(valid_questions_with_answers) > total_questions_to_sample:
        final_sampled_items = random.sample(valid_questions_with_answers, total_questions_to_sample)
        print(f"Successfully sampled {total_questions_to_sample} questions.")
    else:
        final_sampled_items = valid_questions_with_answers
        print(f"Using all {len(final_sampled_items)} available questions (less than {total_questions_to_sample} requested).")
    
    # 최종 결과 리스트 구성
    question_counter = 0 
    for entry in final_sampled_items:
        question_counter += 1
        original_item = entry["raw_item"]
        extracted_answer = entry["extracted_answer"]

        processed_questions.append({
            "question_id": question_counter, # Unique sequential ID
            "question": original_item.get("problem"),
            "level": original_item.get("level"),         # E.g., "Level 1"
            "category": original_item.get("category"),   # E.g., "algebra" (now guaranteed to exist)
            "answer": [extracted_answer] # Store extracted answer in a list
        })

    # Save the processed questions to the output JSON file
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_questions, outfile, ensure_ascii=False, indent=4)
    print(f"\nSuccessfully created '{output_file_path}' with {len(processed_questions)} questions (total sampled).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare uniformly sampled questions from hendrycks_math dataset TEST SPLIT with \\boxed{} answers.")
    parser.add_argument("-o", "--output-json", type=str, required=True,
                        help="Path to save the new .json file (e.g., math_sampled_uniform_100.json).")
    parser.add_argument("-n", "--total-samples", type=int, default=100, # <-- 인자명 변경
                        help="Total number of questions to sample across all levels and categories. Default 100.")
    
    args = parser.parse_args()
    
    prepare_math_questions(args.output_json, args.total_samples)