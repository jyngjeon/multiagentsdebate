# run_single.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re # normalize_answer 함수에서 사용 (최상단에 위치)

# Agent 클래스는 code/utils/agent.py에 있습니다.
from code.utils.agent import Agent

# --- 답변 정규화 함수 (run_debate.py와 동일한 최신 버전) ---
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

    # --- 1단계: <think> 태그 제거 ---
    # 이 normalize_answer 함수는 _parse_moderator_response를 거쳐 JSON에서 추출된
    # 'debate_answer' 필드 값을 받는 것이 일반적입니다.
    # 해당 필드 안에는 <think> 태그가 없을 것이므로, 사실 이 로직은 불필요할 수 있지만
    # 혹시 모를 경우를 대비하여 유지합니다.
    normalized = re.sub(r'<think>.*?</think>', '', normalized, flags=re.DOTALL).strip()
    
    # --- 2단계: 구두점 및 특정 기호 제거 ---
    # 괄호, 따옴표, 콜론, 세미콜론, 쉼표, 달러 기호 등 일반적인 구두점 및 기호 제거
    # % 기호도 여기서 제거하여 숫자 추출에 방해되지 않도록 합니다.
    normalized = re.sub(r'[(),"\':;%,$]', '', normalized).replace(',', '') 
    
    # --- 3단계: 단위 문자열 제거 ---
    # 이 단계에서 단위를 제거하여 순수 숫자만 남깁니다.
    units_to_remove = [
        r'\bm/s\b', r'\bmeters/second\b', r'\bper second\b',
        r'\bkg\b', r'\bkilogram\b', r'\bton(?:ne)?s?\b', # 톤도 숫자로만 변환되도록
        r'\bminute(?:s)?\b', r'\bhour(?:s)?\b', r'\bsecond(?:s)?\b',
        r'\bdegrees?\s*celsiu(?:s)?\b', r'\bdegree\b',
        r'\bhz\b', r'\bohm\b', r'\bvolt\b', r'\bg\b', r'\bml\b', r'\bl\b',
    ]
    # 단어 경계 \b를 사용하여 단위가 단어의 일부로 인식되지 않도록 합니다.
    for unit_regex in units_to_remove:
        normalized = re.sub(unit_regex, '', normalized).strip()
    
    # --- 4단계: 문자열 끝에서 숫자/분수 형태를 포함하는 패턴 추출 ---
    # 모델이 답변을 텍스트 마지막에 두는 경향을 활용
    # 이 정규식은 숫자, 소수, 분수, 그리고 선택적인 음수 부호만 포함합니다.
    # 이제 단위는 앞 단계에서 제거되었으므로, 여기서 단위를 포함하는 패턴은 없습니다.
    final_number_match = re.search(
        r'([-+]?\d+\.?\d*(?:/\d+\.?\d*)?)\s*$', 
        normalized)
    
    if final_number_match:
        extracted_num_str = final_number_match.group(1).strip()
        normalized = extracted_num_str
    else:
        # 끝에서 숫자 패턴을 찾지 못하면, 전체 문자열에서 숫자 관련 문자만 남깁니다. (비상용)
        normalized = re.sub(r'[^0-9./-]', '', normalized).strip()

    # 소수점 뒤에 불필요한 점이 남는 경우 제거 (예: "9.09." -> "9.09")
    if normalized.endswith('.'):
        normalized = normalized[:-1]

    # --- 5단계: 최종적으로 숫자로 변환 가능한 형태로 표준화 및 반올림 ---
    try:
        # 1. 분수 처리 (예: "3/2" -> "1.5")
        if '/' in normalized:
            parts = normalized.split('/')
            # 분모가 0이 아니고, 양쪽 부분이 숫자로 구성되어 있는지 확인
            if len(parts) == 2 and parts[0].replace('.', '', 1).replace('-', '', 1).isdigit() and parts[1].replace('.', '', 1).replace('-', '', 1).isdigit():
                if float(parts[1]) != 0:
                    val = float(parts[0]) / float(parts[1])
                    return str(round(val, round_decimals)) # <-- 반올림 적용
        
        # 2. 일반적인 숫자 (정수, 소수)
        val = float(normalized)
        return str(round(val, round_decimals)) # <-- 반올림 적용

    except ValueError:
        # 숫자로 변환할 수 없는 경우 (수식, 복잡한 텍스트 등) 원본 문자열 반환
        return normalized

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single LLM agent's accuracy on CIAR questions.")
    parser.add_argument("-i", "--input-json-file", type=str, required=True,
                        help="Path to the input JSON file containing CIAR questions.")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Directory to save evaluation results.")
    parser.add_argument("-lu", "--local-llm-url", type=str,
                        default="http://0.0.0.0:8000",
                        help="URL of the local LLM server (e.g., http://0.0.0.0:8000).")
    parser.add_argument("-m", "--model-name", type=str,
                        default="Qwen/Qwen3-14B",
                        help="Model name to use for the agent (e.g., Qwen/Qwen3-14B).")
    parser.add_argument("-t", "--temperature", type=float,
                        default=0,
                        help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str,
                        default=None,
                        help="OpenAI API key (only if not using local LLM).")
    parser.add_argument("-s", "--system-prompt", type=str,
                        default="You are a helpful and accurate assistant. Provide a direct answer to the question. The final answer MUST be a single numerical value (integer, decimal, or fraction like 'X/Y') WITHOUT any units, text, explanations, or parentheses. For percentages, output as a decimal (e.g., 9.09% should be 0.0909). For quantities with units (e.g., 500 kg, 5 minutes), convert to a pure numerical value in the most standard base unit for direct comparison (e.g., for 500 kg, output 500; for 5 minutes, output 5). If the question implies a specific unit, ensure the final answer is a pure number for that implied unit. Only output the final answer.",
                        help="The system prompt given to the single agent.")
    
    # --- 변경된 노이즈 인자 이름 ---
    parser.add_argument("-n", "--noise-text", type=str, default="",
                        help="Optional: Text to append as noise to the end of each question. If empty, no noise is added.")
    # --- 변경된 실험 이름 인자 ---
    parser.add_argument("--exp-name", type=str, default="default_experiment",
                        help="A name for the experiment to be included in the summary file name.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    total_questions = len(questions_data)
    correct_answers_count = 0
    all_results = []

    single_agent = Agent(
        model_name=args.model_name,
        name="SingleAgent",
        temperature=args.temperature,
        local_llm_url=args.local_llm_url,
        openai_api_key=args.api_key 
    )
    
    single_agent.set_meta_prompt(args.system_prompt)

    print(f"\nStarting single agent evaluation for {total_questions} questions...\n")

    for i, question_entry in enumerate(tqdm(questions_data, desc="Processing Questions")):
        original_question_text = question_entry["question"]

        question_with_noise = original_question_text
        if args.noise_text:
            question_with_noise = f"{original_question_text}\n\n[NOISE_START] {args.noise_text} [NOISE_END]"
            print(f"Added custom noise to question {i+1}.")
        
        if single_agent.memory_lst and single_agent.memory_lst[0]['role'] == 'system':
            system_message = single_agent.memory_lst[0]
            single_agent.memory_lst = [system_message]
        else:
            single_agent.memory_lst = []
            single_agent.set_meta_prompt(args.system_prompt)

        single_agent.add_event(question_with_noise)
        raw_model_response = single_agent.ask() # 모델의 원본 응답 받기 (JSON 형식 예상)

        # --- JSON 파싱 로직 (run_debate.py의 _parse_moderator_response와 유사) ---
        parsed_response_dict = {}
        try:
            # 1. <think> 태그와 그 내용을 제거합니다.
            cleaned_response = re.sub(r'<think>.*?</think>', '', raw_model_response, flags=re.DOTALL).strip()
            
            # 2. JSON 객체의 시작 { 을 기준으로 그 이전 텍스트를 제거합니다.
            json_start_char_index = cleaned_response.find('{')
            if json_start_char_index != -1:
                cleaned_response = cleaned_response[json_start_char_index:]
            else:
                print(f"Warning: Single agent response did not contain a valid JSON object. Raw response (cleaned): '{cleaned_response}'")
                parsed_response_dict = {"debate_answer": ""}  

            # 3. 추출된 JSON 문자열을 파이썬 딕셔너리로 로드합니다.
            parsed_response_dict = json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from single agent: {e}\nRaw response (cleaned): '{cleaned_response}'")
            parsed_response_dict = {"debate_answer": ""}
        except Exception as e:
            print(f"Unexpected error processing single agent response: {e}\nRaw response (cleaned): '{cleaned_response}'")
            parsed_response_dict = {"debate_answer": ""}
        # --- JSON 파싱 로직 끝 ---

        # `model_raw_answer`는 이제 JSON에서 파싱된 `debate_answer` 값
        model_raw_answer = parsed_response_dict.get("debate_answer", "") 
        
        # 결과 정규화 및 정확도 판단
        model_normalized_answer = normalize_answer(model_raw_answer, round_decimals=2)
        original_correct_answers = [str(ans).strip() for ans in question_entry.get("answer", [])]
        normalized_correct_answers = [normalize_answer(ans, round_decimals=2) for ans in original_correct_answers]

        is_correct = False
        if model_normalized_answer in normalized_correct_answers:
            is_correct = True
        else:
            try:
                model_float = float(model_normalized_answer)
                for correct_ans_normalized in normalized_correct_answers:
                    try:
                        correct_float = float(correct_ans_normalized)
                        if abs(model_float - correct_float) < 1e-6:
                            is_correct = True
                            break
                    except ValueError:
                        continue
            except ValueError:
                pass

        if is_correct:
            correct_answers_count += 1
        
        # 결과 저장 (result_entry)
        result_entry = {
            "question_id": i + 1,
            "question": original_question_text,
            "question_with_noise": question_with_noise,
            "original_correct_answers": original_correct_answers,
            "model_raw_answer": model_raw_answer, # 파싱된 JSON 내부의 "debate_answer" 값
            "model_normalized_answer": model_normalized_answer,
            "normalized_correct_answers": normalized_correct_answers,
            "is_correct": is_correct,
            # Single Agent는 "Reason", "Supported Side"가 없으므로 JSON에서 가져온 "Reason" 또는 기본값
            "Reason": parsed_response_dict.get("Reason", "Single agent direct response."),
            "players": {"SingleAgent": single_agent.memory_lst.copy()},
            "raw_llm_output": raw_model_response # 모델의 원본 (파싱 전) 출력을 저장
        }
        if args.noise_text:
            result_entry['noise_added'] = True
            result_entry['noise_text_used'] = args.noise_text
        else:
            result_entry['noise_added'] = False
            result_entry['noise_text_used'] = ""

        all_results.append(result_entry)

        print(f"\n--- Question {i+1}/{total_questions} ---")
        print(f"Question: {original_question_text}")
        if args.noise_text:
            print(f"  (Noise added: '{args.noise_text[:50]}...')")
        print(f"Model Raw Answer: {model_raw_answer}") # 파싱된 답변 출력
        print(f"Original Correct Answers: {original_correct_answers}")
        print(f"Is Correct: {is_correct}")
        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")

    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Single Agent Evaluation Summary =====")
    print(f"Total Questions: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    # --- 파일 이름에 exp-name 포함 ---
    summary_file_name = f"single_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All single agent results summarized in: {summary_file_path}")