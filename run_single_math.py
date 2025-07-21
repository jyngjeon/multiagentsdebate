# run_single.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re

# Agent 클래스는 code/utils/agent.py에 있습니다.
from code.utils.agent import Agent

def extract_json_from_response(raw_text: str) -> str | None:
    """
    LLM의 응답 텍스트에서 JSON 객체 문자열을 추출합니다.
    1. 마크다운 코드 블록(```json ... ```)을 우선적으로 찾습니다.
    2. 마크다운이 없으면, 첫 '{'부터 마지막 '}'까지의 내용을 찾습니다.
    """
    if not isinstance(raw_text, str):
        return None

    # 1. 마크다운 ```json ... ``` 블록이 있는지 확인
    # re.DOTALL은 줄바꿈 문자(\n)도 . 에 포함시켜 검색하게 함
    match = re.search(r'```json\s*(\{.*\})\s*```', raw_text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. 마크다운 블록이 없다면, 가장 바깥쪽의 JSON 객체를 찾기
    # '{'로 시작하고 '}'로 끝나는 가장 큰 덩어리를 찾음
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        return match.group(0)
        
    return None

def extract_boxed_answer(solution_text: str) -> str | None:
    """
    주어진 텍스트에서 \\boxed{...} 안의 원본 내용을 그대로 추출합니다.
    (이 함수는 더 이상 직접 사용되지 않지만, 호환성을 위해 유지됩니다.)
    """
    if not isinstance(solution_text, str):
        return None
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

def normalize_latex_answer(answer_str: str) -> str:
    """
    LaTeX 답변 문자열을 일관된 비교를 위해 정규화합니다.
    (이 함수는 더 이상 직접 사용되지 않지만, 호환성을 위해 유지됩니다.)
    """
    if not isinstance(answer_str, str):
        return ""
    normalized_str = answer_str.replace(r'\dfrac', r'\frac')
    normalized_str = re.sub(r'\s+', '', normalized_str)
    return normalized_str

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single LLM agent's accuracy on math questions.")
    # ... (기존 argparse 설정)
    parser.add_argument("-i", "--input-json-file", type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("-lu", "--local-llm-url", type=str, default="http://0.0.0.0:8000", help="URL of the local LLM server.")
    parser.add_argument("-m", "--model-name", type=str, default="Qwen/Qwen3-14B", help="Model name to use for the agent.")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str, default=None, help="OpenAI API key (only if not using local LLM).")
    # 수정: 시스템 프롬프트를 JSON 숫자 답변을 요구하도록 변경
    parser.add_argument("-s", "--system-prompt", type=str, 
                        default="You are a brilliant mathematician. Please solve the following math problem.\n\n**IMPORTANT:** Your output MUST be a single JSON object. The value for the 'answer' key must be a single number. For example: {\"reasoning\": \"Step-by-step thinking...\", \"answer\": 42}", 
                        help="The system prompt given to the single agent.")
    parser.add_argument("-n", "--noise-text", type=str, default="", help="Optional: Text to append as noise to the end of each question.")
    parser.add_argument("--exp-name", type=str, default="default_experiment", help="A name for the experiment to be included in the summary file name.")
    args = parser.parse_args()

    # --- 디렉토리 및 파일 설정 ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # --- 수정: 결과 요약 파일 경로를 미리 정의 ---
    summary_file_name = f"single_results_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)

    total_questions = len(questions_data)
    correct_answers_count = 0
    # all_results = [] # 수정: 리스트에 모든 결과를 모으지 않음

    single_agent = Agent(model_name=args.model_name, name="SingleAgent", temperature=args.temperature, local_llm_url=args.local_llm_url, openai_api_key=args.api_key)
    
    print(f"\nStarting single agent evaluation for {total_questions} questions...")
    print(f"Results will be saved incrementally to: {summary_file_path}\n")

    for i, question_entry in enumerate(tqdm(questions_data, desc="Processing Questions")):
        original_question_text = question_entry["question"]
        question_with_noise = f"{original_question_text} {args.noise_text}" if args.noise_text else original_question_text
        
        single_agent.memory_lst = []
        single_agent.set_meta_prompt(args.system_prompt)
        single_agent.add_event(question_with_noise)
        raw_model_response = single_agent.ask()

        # --- 결과 추출 및 정확도 계산 (수정된 로직) ---
        model_answer_val = None
        model_reasoning = ""
        json_str_from_response = extract_json_from_response(raw_model_response)

        if json_str_from_response:
            try:
                # 추출된 JSON 문자열을 파싱
                # --- 추가된 라인: 백슬래시를 이중 백슬래시로 변경하여 이스케이프 처리 ---
                corrected_json_str = json_str_from_response.replace('\\', '\\\\')
        
                # 수정된 문자열로 파싱 시도
                response_json = json.loads(corrected_json_str)
                model_answer_val = response_json.get("answer")
                model_reasoning = response_json.get("reasoning", "")
            except json.JSONDecodeError:
                print(f"\nWarning: Failed to decode the extracted JSON for question {i+1}. Extracted: {json_str_from_response}")
        else:
            print(f"\nWarning: Could not find any JSON in the response for question {i+1}. Response: {raw_model_response}")
            
        # # --- 기존 로직 주석 처리 ---
        # model_extracted_answer = extract_boxed_answer(raw_model_response)
        # ground_truth_answers = [str(ans).strip() for ans in question_entry.get("answer", [])]
        # normalized_model_answer = normalize_latex_answer(model_extracted_answer)
        # normalized_ground_truth = [normalize_latex_answer(ans) for ans in ground_truth_answers]
        # is_correct = False
        # if model_extracted_answer is not None:
        #     if normalized_model_answer in normalized_ground_truth:
        #         is_correct = True
        
        # --- 새로운 정확도 계산 로직 ---
        ground_truth_str_list = [str(ans).strip() for ans in question_entry.get("answer", [])]
        is_correct = False
        if isinstance(model_answer_val, (int, float)):
            try:
                ground_truth_float_list = [float(gt) for gt in ground_truth_str_list]
                if float(model_answer_val) in ground_truth_float_list:
                    is_correct = True
            except (ValueError, TypeError):
                is_correct = False
        
        if is_correct:
            correct_answers_count += 1
        
        # --- 최종 결과 JSON에 저장할 정보 구성 ---
        result_entry = {
            "question_id_in_file": question_entry.get("question_id"),
            "question_id_processed": i + 1,
            "question": original_question_text,
            "question_with_noise": question_with_noise,
            "noise_added": bool(args.noise_text),
            "noise_text_used": args.noise_text,
            "ground_truth_answers": ground_truth_str_list,
            "model_full_output": raw_model_response,
            "model_final_answer": model_answer_val,
            "model_reasoning": model_reasoning,
            "is_correct": is_correct,
            "agent_memory": single_agent.memory_lst.copy() # 프롬프트와 응답 기록
        }

        # all_results.append(result_entry) # 수정: 리스트에 추가하는 대신 파일에 직접 기록
        
        # --- 수정: 각 질문 처리 후 결과를 파일에 바로 추가 ---
        with open(summary_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            print(f"Result for question {i+1} saved to {summary_file_path}")

        print(f"\n--- Question {i+1}/{total_questions} ---")
        if is_correct:
            print(f"Result: CORRECT (Answered: '{model_answer_val}' | Expected: {ground_truth_str_list})")
        else:
            print(f"Result: INCORRECT (Answered: '{model_answer_val}' | Expected: {ground_truth_str_list})")
        
        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")

    # --- 전체 정확도 계산 및 요약 출력 ---
    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Single Agent Evaluation Summary =====")
    print(f"Total Questions Processed: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    
    # --- 수정: 모든 결과를 하나의 요약 파일로 저장하는 로직 제거 ---
    # summary_file_name = f"single_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # summary_file_path = os.path.join(args.output_dir, summary_file_name)
    # with open(summary_file_path, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All single agent results have been incrementally saved to: {summary_file_path}")