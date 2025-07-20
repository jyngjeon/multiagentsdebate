# run_debate.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re

# code/utils/debate.py에서 필요한 클래스를 임포트합니다.
from code.utils.debate import Debate, DebatePlayer, NAME_LIST

def extract_boxed_answer(solution_text: str) -> str | None:
    """
    주어진 텍스트에서 \\boxed{...} 안의 원본 내용을 그대로 추출합니다.
    중첩된 괄호를 올바르게 처리합니다.
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
    """
    if not isinstance(answer_str, str):
        return ""
    normalized_str = answer_str.replace(r'\dfrac', r'\frac')
    normalized_str = re.sub(r'\s+', '', normalized_str)
    return normalized_str

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch multi-agent debate and calculate accuracy.")
    # ... (인자 파서는 동일)
    parser.add_argument("-i", "--input-json-file", type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save individual debate results.")
    parser.add_argument("-lu", "--local-llm-url", type=str, default="http://0.0.0.0:8000", help="URL of the local LLM server (e.g., http://0.0.0.0:8000).")
    parser.add_argument("-m", "--model-name", type=str, default="Qwen/Qwen3-14B", help="Model name to use for the debate (e.g., Qwen/Qwen3-14B).")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str, default=None, help="OpenAI API key (only if not using local LLM).")
    parser.add_argument("-c", "--config-prompt-path", type=str, default="code/utils/config4all.json", help="Path to the JSON file containing prompt templates (e.g., config4all.json).")
    parser.add_argument("-n", "--noise-text", type=str, default="", help="Optional: Text to append as noise to the end of each question. If empty, no noise is added.")
    parser.add_argument("--exp-name", type=str, default="default_experiment", help="A name for the experiment to be included in the summary file name.")
    args = parser.parse_args()

    # --- 디렉토리 및 파일 설정 ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    with open(args.config_prompt_path, "r", encoding='utf-8') as f:
        prompt_templates = json.load(f)

    # --- 수정: 결과 요약 파일 경로를 미리 정의 ---
    summary_file_name = f"debate_results_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)

    total_questions = len(questions_data)
    correct_answers_count = 0
    # all_results = [] # 수정: 리스트에 모든 결과를 모으지 않음

    print(f"\nStarting batch debate for {total_questions} questions...")
    print(f"Results will be saved incrementally to: {summary_file_path}\n")

    for i, question_entry in enumerate(tqdm(questions_data, desc="Processing Questions")):
        # ... (Debate 설정 및 실행 부분은 동일)
        original_question_text = question_entry["question"]
        question_with_noise = f"{original_question_text}{args.noise_text}" if args.noise_text else original_question_text
        current_config_for_debate = prompt_templates.copy()
        current_config_for_debate.update(question_entry)
        current_config_for_debate['question'] = question_with_noise
        current_config_for_debate['debate_topic'] = question_with_noise
        print(f"\n--- Question {i+1}/{total_questions}: {original_question_text[:80]}... ---")
        debate = Debate(num_players=3, openai_api_key=args.api_key, local_llm_url=args.local_llm_url, model_name=args.model_name, config=current_config_for_debate, temperature=args.temperature, sleep_time=0, question_id=i + 1)
        
        # final_debate_config는 토론의 모든 정보를 담고 있음 (LLM 답변 포함)
        final_debate_config = debate.run() 

        # --- 결과 추출 및 정확도 계산 (숫자 답변용 로직) ---
        model_answer_val = final_debate_config.get("debate_answer")
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
        # result_details는 토론의 메타데이터와 LLM 생성 내용을 모두 포함
        result_details = final_debate_config.copy()
        result_details['question_id_in_file'] = question_entry.get("question_id")
        result_details['noise_added'] = bool(args.noise_text)
        result_details['noise_text_used'] = args.noise_text
        result_details['question_original_text'] = original_question_text
        result_details['model_final_answer'] = model_answer_val
        result_details['ground_truth_answers'] = ground_truth_str_list
        result_details['is_correct'] = is_correct

        # all_results.append(result_details) # 수정: 리스트에 추가하는 대신 파일에 직접 기록

        # --- 수정: 각 질문 처리 후 결과를 파일에 바로 추가 ---
        with open(summary_file_path, 'a', encoding='utf-8') as f:
            # result_details를 JSON 문자열로 변환하여 한 줄로 기록
            f.write(json.dumps(result_details, ensure_ascii=False) + '\n')
            print(f"Result for question {i+1} saved to {summary_file_path}")

        if is_correct:
            print(f"Question {i+1}: CORRECT (Answered: '{model_answer_val}' | Expected: {ground_truth_str_list})")
        else:
            print(f"Question {i+1}: INCORRECT (Answered: '{model_answer_val}' | Expected: {ground_truth_str_list})")
        
        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")
    
    # --- 전체 정확도 계산 및 요약 출력 ---
    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Batch Debate Summary =====")
    print(f"Total Questions Processed: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    # --- 수정: 모든 결과를 하나의 요약 파일로 저장하는 로직 제거 ---
    # summary_file_name = f"debate_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # summary_file_path = os.path.join(args.output_dir, summary_file_name)
    # with open(summary_file_path, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All debate results have been incrementally saved to: {summary_file_path}")