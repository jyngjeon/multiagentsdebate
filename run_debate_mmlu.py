# run_debate.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re

# code/utils/debate.py에서 필요한 클래스를 임포트합니다.
from code.utils.debate import Debate

def extract_json_from_response(raw_text: str) -> str | None:
    """
    LLM의 응답 텍스트에서 JSON 객체 문자열을 추출합니다.
    1. 마크다운 코드 블록(```json ... ```)을 우선적으로 찾습니다.
    2. 마크다운이 없으면, 첫 '{'부터 마지막 '}'까지의 내용을 찾습니다.
    """
    if not isinstance(raw_text, str):
        return None
    # 1. 마크다운 ```json ... ``` 블록이 있는지 확인
    match = re.search(r'```json\s*(\{.*\})\s*```', raw_text, re.DOTALL)
    if match:
        return match.group(1)
    # 2. 마크다운 블록이 없다면, 가장 바깥쪽의 JSON 객체를 찾기
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch multi-agent debate for multiple-choice questions.")
    parser.add_argument("-i", "--input-json-file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save debate results.")
    parser.add_argument("-lu", "--local-llm-url", type=str, default="http://0.0.0.0:8000", help="URL of the local LLM server.")
    parser.add_argument("-m", "--model-name", type=str, default="Qwen/Qwen3-14B", help="Model name to use for the debate.")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str, default=None, help="OpenAI API key.")
    parser.add_argument("-c", "--config-prompt-path", type=str, default="code/utils/config4all.json", help="Path to the prompt templates JSON file.")
    parser.add_argument("-n", "--noise-text", type=str, default="", help="Optional: Text to append as noise to each question.")
    parser.add_argument("--exp-name", type=str, default="default_experiment", help="A name for the experiment.")
    args = parser.parse_args()

    # --- 디렉토리 및 파일 설정 ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    with open(args.config_prompt_path, "r", encoding='utf-8') as f:
        prompt_templates = json.load(f)

    summary_file_name = f"debate_results_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)

    total_questions = len(questions_data)
    correct_answers_count = 0

    print(f"\nStarting batch debate for {total_questions} questions...")
    print(f"Results will be saved incrementally to: {summary_file_path}\n")

    for i, question_entry in enumerate(tqdm(questions_data, desc="Processing Questions")):
        original_question_text = question_entry["question"]
        question_with_noise = f"{original_question_text}{args.noise_text}" if args.noise_text else original_question_text
        current_config_for_debate = prompt_templates.copy()
        current_config_for_debate.update(question_entry)
        current_config_for_debate['question'] = question_with_noise
        current_config_for_debate['debate_topic'] = question_with_noise
        print(f"\n--- Question {i+1}/{total_questions}: {original_question_text[:80]}... ---")

        debate = Debate(num_players=3, openai_api_key=args.api_key, local_llm_url=args.local_llm_url, model_name=args.model_name, config=current_config_for_debate, temperature=args.temperature, sleep_time=0, question_id=i + 1)
        final_debate_config = debate.run()

        # --- 결과 추출 로직 ---
        raw_moderator_output = final_debate_config.get("debate_answer")
        model_answer_val = None
        json_str_from_response = extract_json_from_response(raw_moderator_output)

        if json_str_from_response:
            try:
                # 백슬래시 문제를 회피하기 위한 전처리
                corrected_json_str = json_str_from_response.replace('\\', '\\\\')
                response_json = json.loads(corrected_json_str)
                # 사회자의 JSON에서 'debate_answer' 키의 값을 가져옴
                model_answer_val = response_json.get("debate_answer")
            except json.JSONDecodeError:
                print(f"\nWarning: Failed to decode the moderator's JSON for question {i+1}. Extracted: {json_str_from_response}")
        elif isinstance(raw_moderator_output, str):
             # debate.run() 내부에서 이미 파싱이 성공하여 문자열 값으로 들어온 경우
            model_answer_val = raw_moderator_output
        else:
            print(f"\nWarning: Could not find JSON in moderator's output for question {i+1}. Output: {raw_moderator_output}")

        # --- MMLU 채점을 위한 정확도 계산 로직 ---
        ground_truth_answer = str(question_entry.get("answer", "")).strip()

        is_correct = False
        # 모델 답변이 문자열인지 확인
        if isinstance(model_answer_val, str):
            # 추가: "."가 포함된 답변 형식("C. Blue")을 처리
            cleaned_answer = model_answer_val.split('.')[0]

            # 정제된 답변으로 비교
            normalized_model_answer = cleaned_answer.strip().upper()
            normalized_ground_truth = ground_truth_answer.upper()
            
            if normalized_model_answer == normalized_ground_truth:
                is_correct = True

        if is_correct:
            correct_answers_count += 1

        # --- 결과 저장 ---
        result_details = final_debate_config.copy()
        result_details['question_id_in_file'] = question_entry.get("task_id", question_entry.get("question_id"))
        result_details['question_id_processed'] = i + 1
        result_details['question_original_text'] = original_question_text
        result_details['ground_truth_answer'] = ground_truth_answer
        result_details['model_final_answer'] = model_answer_val
        result_details['is_correct'] = is_correct

        with open(summary_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_details, ensure_ascii=False, indent=4))

        print(f"\n--- Question {i+1}/{total_questions} ---")
        if is_correct:
            print(f"Result: CORRECT (Answered: '{model_answer_val}' | Expected: '{ground_truth_answer}')")
        else:
            print(f"Result: INCORRECT (Answered: '{model_answer_val}' | Expected: '{ground_truth_answer}')")

        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")

    # --- 최종 요약 출력 ---
    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Batch Debate Summary =====")
    print(f"Total Questions Processed: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"All debate results have been incrementally saved to: {summary_file_path}")