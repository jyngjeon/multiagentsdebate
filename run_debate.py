# run_debate.py

import os
import json
import random
import argparse
from tqdm import tqdm
from datetime import datetime
import re # normalize_answer 함수에서 사용 (최상단에 위치)

# DebatePlayer와 Debate 클래스를 code/utils/debate.py에서 임포트합니다.
from code.utils.debate import DebatePlayer, Debate, NAME_LIST

# --- 답변 정규화 함수 (run_single.py와 동일한 최신 버전) ---
# 이 함수는 JSON에서 추출된 'debate_answer' 값(문자열)을 받아 순수 숫자로 정규화합니다.
def normalize_answer(answer_str: str, round_decimals: int = 2) -> str:
    """
    모델 답변 또는 정답 리스트의 각 항목을 비교를 위해 정규화합니다.
    오직 숫자 (정수, 소수, 분수) 형태를 추출하고 표준화하며,
    불필요한 텍스트, 단위, 설명을 제거하고, 숫자는 지정된 소수점 자리까지 반올림합니다.
    """
    if not isinstance(answer_str, str):
        answer_str = str(answer_str)

    normalized = answer_str.lower().strip()

    # <think> 태그 제거
    normalized = re.sub(r'<think>.*?</think>', '', normalized, flags=re.DOTALL).strip()
    
    # 괄호, 따옴표, 콜론, 세미콜론, 쉼표, 달러 기호 등 일반적인 구두점 및 기호 제거
    normalized = re.sub(r'[(),"\':;%,$]', '', normalized).replace(',', '') 
    
    # "the answer is" 같은 일반적인 서론 제거
    phrases_to_remove_regex = [
        r'\bthe answer is\b', r'\bfinal answer is\b', r'\bthe final answer is\b',
        r'\bthe result is\b', r'\bit is\b', r'\b\s+is\s+\b',
        r'\bconclusion\b', r'\bhere\'s the breakdown\b',
        r'\bkey observations\b', r'\bsolving the recurrence\b',
        r'\bverification via alternative approaches\b', r'\bdirect substitution\b',
        r'\bgeometric distribution\b',
        r'\bapproximately\b'
    ]
    for phrase_regex in phrases_to_remove_regex:
        normalized = re.sub(phrase_regex, '', normalized).strip()

    # 단위 문자열 제거
    units_to_remove = [
        r'\bm/s\b', r'\bmeters/second\b', r'\bper second\b',
        r'\bkg\b', r'\bkilogram\b', r'\bton(?:ne)?s?\b', # 톤도 숫자로만 변환되도록
        r'\bminute(?:s)?\b', r'\bhour(?:s)?\b', r'\bsecond(?:s)?\b',
        r'\bdegrees?\s*celsiu(?:s)?\b', r'\bdegree\b',
        r'\bhz\b', r'\bohm\b', r'\bvolt\b', r'\bg\b', r'\bml\b', r'\bl\b',
    ]
    for unit_regex in units_to_remove:
        normalized = re.sub(unit_regex, '', normalized).strip()
    
    # 문자열 끝에서 숫자/분수 형태 추출 시도
    final_number_match = re.search(
        r'([-+]?\d+\.?\d*(?:/\d+\.?\d*)?)\s*$', 
        normalized)
    
    if final_number_match:
        extracted_num_str = final_number_match.group(1).strip()
        normalized = extracted_num_str
    else:
        # 끝에서 숫자 패턴을 찾지 못하면, 전체 문자열에서 숫자 관련 문자만 남깁니다. (비상용)
        normalized = re.sub(r'[^0-9./-]', '', normalized).strip()

    if normalized.endswith('.'):
        normalized = normalized[:-1]

    # 최종적으로 숫자로 변환 가능한 형태로 표준화 및 반올림
    try:
        if '/' in normalized:
            parts = normalized.split('/')
            if len(parts) == 2 and parts[0].replace('.', '', 1).replace('-', '', 1).isdigit() and parts[1].replace('.', '', 1).replace('-', '', 1).isdigit():
                if float(parts[1]) != 0:
                    val = float(parts[0]) / float(parts[1])
                    return str(round(val, round_decimals))
        
        val = float(normalized)
        return str(round(val, round_decimals))

    except ValueError:
        return normalized

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch multi-agent debate and calculate accuracy.")
    parser.add_argument("-i", "--input-json-file", type=str, required=True,
                        help="Path to the input JSON file containing questions.")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Directory to save individual debate results.")
    parser.add_argument("-lu", "--local-llm-url", type=str,
                        default="http://0.0.0.0:8000",
                        help="URL of the local LLM server (e.g., http://0.0.0.0:8000).")
    parser.add_argument("-m", "--model-name", type=str,
                        default="Qwen/Qwen3-14B",
                        help="Model name to use for the debate (e.g., Qwen/Qwen3-14B).")
    parser.add_argument("-t", "--temperature", type=float,
                        default=0,
                        help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str,
                        default=None,
                        help="OpenAI API key (only if not using local LLM).")
    parser.add_argument("-c", "--config-prompt-path", type=str,
                        default="code/utils/config4all.json",
                        help="Path to the JSON file containing prompt templates (e.g., config4all.json).")
    
    # --- 노이즈 관련 인자 ---
    parser.add_argument("-n", "--noise-text", type=str, default="",
                        help="Optional: Text to append as noise to the end of each question. If empty, no noise is added.")
    # --- 실험 이름 인자 ---
    parser.add_argument("--exp-name", type=str, default="default_experiment",
                        help="A name for the experiment to be included in the summary file name.")

    args = parser.parse_args()

    # 결과 저장 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 질문 JSON 파일 로드
    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # 프롬프트 템플릿 로드
    current_script_path = os.path.abspath(__file__)
    MAD_path_for_config = os.path.dirname(os.path.abspath(__file__)) 
    config_full_path = os.path.join(MAD_path_for_config, args.config_prompt_path) 
    
    with open(config_full_path, "r", encoding='utf-8') as f:
        prompt_templates = json.load(f)

    total_questions = len(questions_data)
    correct_answers_count = 0
    all_results = []

    print(f"\nStarting batch debate for {total_questions} questions...\n")

    for i, question_entry in enumerate(tqdm(questions_data, desc="Processing Questions")):
        original_question_text = question_entry["question"]
        
        # --- 노이즈 추가 로직 ---
        question_with_noise = original_question_text
        if args.noise_text: # --noise-text가 비어있지 않다면
            # 지정된 노이즈 텍스트를 그대로 뒤에 붙입니다.
            question_with_noise = f"{original_question_text}{args.noise_text}"
            print(f"Added custom noise to question {i+1}.") # 노이즈 추가 확인용 출력
        # --- 노이즈 추가 로직 끝 ---

        current_config_for_debate = prompt_templates.copy() # 프롬프트 템플릿 복사
        current_config_for_debate.update(question_entry)    # 질문 데이터를 병합 (question, answer 등)
        
        # Debate 클래스에 전달될 질문은 노이즈가 추가된 질문입니다.
        current_config_for_debate['question'] = question_with_noise
        current_config_for_debate['debate_topic'] = question_with_noise # debate_topic도 노이즈 포함

        print(f"\n--- Question {i+1}/{total_questions}: {original_question_text[:80]}... ---") # 출력은 원본 질문 사용

        # Debate 인스턴스 생성 및 실행
        debate = Debate(num_players=3,
                        openai_api_key=args.api_key,
                        local_llm_url=args.local_llm_url,
                        model_name=args.model_name,
                        config=current_config_for_debate, 
                        temperature=args.temperature,
                        sleep_time=0,
                        question_id=i + 1 # 현재 문제의 ID 전달
                        )
        
        # Debate.run()은 이제 최종 config를 반환합니다.
        final_debate_config = debate.run() 

        # --- 결과 추출 및 정확도 계산 ---
        # model_raw_answer는 debate.run()이 반환한 config에서 가져옵니다.
        model_raw_answer = str(final_debate_config.get("debate_answer", "")).strip()
        # normalize_answer 함수를 사용하여 정규화합니다.
        model_normalized_answer = normalize_answer(model_raw_answer, round_decimals=2)

        # 원본 정답 리스트를 가져와 정규화합니다.
        original_correct_answers = [str(ans).strip() for ans in question_entry.get("answer", [])]
        normalized_correct_answers = [normalize_answer(ans, round_decimals=2) for ans in original_correct_answers]

        is_correct = False
        # 1. 정규화된 문자열 간의 직접 비교
        if model_normalized_answer in normalized_correct_answers:
            is_correct = True
        else:
            # 2. 부동소수점 비교 (정규화된 답변이 숫자로 변환 가능한 경우)
            try:
                model_float = float(model_normalized_answer)
                for correct_ans_normalized in normalized_correct_answers:
                    try:
                        correct_float = float(correct_ans_normalized)
                        # 작은 오차 허용 (부동소수점 비교). 오차 범위 1e-6은 충분히 작습니다.
                        if abs(model_float - correct_float) < 1e-6: 
                            is_correct = True
                            break # 정답 찾았으면 루프 종료
                    except ValueError:
                        continue # 정답 항목이 숫자가 아니면 float 비교 건너뜀
            except ValueError:
                pass # 모델 답변이 숫자가 아니면 float 비교 생략

        if is_correct:
            correct_answers_count += 1
        
        # --- 최종 결과 JSON에 추가 정보 저장 ---
        if args.noise_text: # 노이즈 텍스트가 있다면 관련 정보 저장
            final_debate_config['noise_added'] = True
            final_debate_config['noise_text_used'] = args.noise_text 
            final_debate_config['question_original_text'] = original_question_text 
            final_debate_config['question_with_noise'] = question_with_noise
        else: # 노이즈가 없다면 관련 정보 없음으로 설정
            final_debate_config['noise_added'] = False
            final_debate_config['noise_text_used'] = ""
            final_debate_config['question_original_text'] = original_question_text 
            final_debate_config['question_with_noise'] = original_question_text # 노이즈 없으면 원본과 동일

        # is_correct, model_raw_answer 등은 final_debate_config에 없으므로 직접 추가합니다.
        # (debate.run()이 반환하는 config는 Moderator/Judge의 최종 JSON 내용만 업데이트했으므로)
        final_debate_config['model_raw_answer'] = model_raw_answer
        final_debate_config['model_normalized_answer'] = model_normalized_answer
        final_debate_config['normalized_correct_answers'] = normalized_correct_answers
        final_debate_config['is_correct'] = is_correct
        final_debate_config['original_correct_answers'] = original_correct_answers # 원본 정답도 다시 저장

        # 모든 질문의 상세 결과를 수집하는 all_results 리스트에 추가
        all_results.append(final_debate_config)

        # --- 개별 결과 파일 저장 ---
        # output_file_name = f"debate_result_{i+1}_{'correct' if is_correct else 'incorrect'}.json"
        # output_file_path = os.path.join(args.output_dir, output_file_name)
        # with open(output_file_path, 'w', encoding='utf-8') as outfile:
        #     json.dump(final_debate_config, outfile, ensure_ascii=False, indent=4)
        # print(f"Result for Question {i+1} saved to {output_file_path}")

        if is_correct:
            print(f"Question {i+1}: CORRECT (Answered: '{model_raw_answer}' | Expected: {original_correct_answers})")
        else:
            print(f"Question {i+1}: INCORRECT (Answered: '{model_raw_answer}' | Expected: {original_correct_answers})")
        
        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")
    
    # --- 전체 정확도 계산 및 요약 출력 ---
    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Batch Debate Summary =====")
    print(f"Total Questions: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    # --- 모든 결과를 하나의 요약 파일로 저장 (파일 이름에 experiment-name 포함) ---
    summary_file_name = f"debate_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All debate results summarized in: {summary_file_path}")