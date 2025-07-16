# run_single.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re 

# Agent 클래스는 code/utils/agent.py에 있습니다.
from code.utils.agent import Agent

# --- 답변 정규화 함수 (run_debate.py와 동일한 최신 버전) ---
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
    # --- system-prompt 기본값 수정 ---
    parser.add_argument("-s", "--system-prompt", type=str,
                        default="You are a helpful and accurate assistant. Provide a direct answer to the question. The final answer MUST be a single numerical value (integer, decimal, or fraction like 'X/Y') WITHOUT any units, text, explanations, or parentheses. For percentages, output as a decimal (e.g., 9.09% should be 0.0909). For fractions like '12/29', output as the fraction itself (e.g., '12/29') or its decimal equivalent (e.g., '0.41379'). Quantities with units (e.g., 500 kg, 5 minutes) must be converted to a pure numerical value (e.g., for 500 kg, output 500; for 5 minutes, output 5). If the question implies a specific unit, ensure the final answer is a pure number for that implied unit. Now output your answer in JSON format, with the format as follows: {\"debate_answer\": \"\"}. Please strictly output ONLY the JSON, do not output irrelevant content or any additional text outside the JSON object.",
                        help="The system prompt given to the single agent.")
    
    # --- 노이즈 관련 인자 (이름 통일) ---
    parser.add_argument("-n", "--noise-text", type=str, default="",
                        help="Optional: Text to append as noise to the end of each question. If empty, no noise is added.")
    # --- 실험 이름 인자 (이름 통일) ---
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
            question_with_noise = f"{original_question_text} {args.noise_text}" 
            print(f"Added custom noise to question {i+1}.")
        
        if single_agent.memory_lst and single_agent.memory_lst[0]['role'] == 'system':
            system_message = single_agent.memory_lst[0]
            single_agent.memory_lst = [system_message]
        else:
            single_agent.memory_lst = []
            single_agent.set_meta_prompt(args.system_prompt)

        single_agent.add_event(question_with_noise)
        raw_model_response = single_agent.ask() # 모델의 원본 응답 받기 (JSON 형식 예상)

        # --- JSON 파싱 로직 (run_debate.py의 _parse_moderator_response와 동일하게) ---
        parsed_response_dict = {"debate_answer": ""} # 항상 딕셔너리로 초기화
        try:
            # 1. <think> 태그와 그 내용을 제거합니다.
            cleaned_response = re.sub(r'<think>.*?</think>', '', raw_model_response, flags=re.DOTALL).strip()
            
            # 2. JSON 객체의 시작 { 을 기준으로 그 이전 텍스트를 제거합니다.
            json_start_char_index = cleaned_response.find('{')
            if json_start_char_index != -1:
                cleaned_response = cleaned_response[json_start_char_index:]
            else:
                # JSON 시작 괄호가 없으면, 순수한 숫자/텍스트 답변일 가능성이 높음
                print(f"Warning: Single agent response did not contain an opening '{{' for JSON. Raw response (cleaned): '{cleaned_response}'")
                # 이 경우 parsed_response_dict는 초기값 {"debate_answer": ""}를 유지.
                # 모델이 순수 숫자만 반환했을 때의 처리를 위해, cleaned_response를 debate_answer로 할당.
                parsed_response_dict = {"debate_answer": cleaned_response} # <-- JSON 파싱 실패 시 fallback
            
            # 3. 추출된 JSON 문자열을 파이썬 딕셔너리로 로드합니다.
            # 이 부분은 json_start_char_index != -1 일때만 실행되어야 합니다.
            # 하지만 cleaned_response가 조정되었고, json.loads가 오류를 낼 경우를 대비하는 것이므로,
            # 앞선 if/else 구문과 별개로 try-except로 감싸는 것이 더 안전합니다.
            if json_start_char_index != -1: # JSON 시작 괄호를 찾은 경우에만 json.loads 시도
                parsed_response_dict = json.loads(cleaned_response)
                # JSON은 로드했지만, debate_answer 필드가 없거나 비어있을 경우에 대한 방어
                if not parsed_response_dict.get("debate_answer", ""):
                    print(f"Warning: JSON loaded but 'debate_answer' field is empty/missing. Raw response (cleaned): '{cleaned_response}'")
                    parsed_response_dict["debate_answer"] = cleaned_response.strip() # JSON은 맞지만 debate_answer 필드가 없으면 raw를 넣기
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from single agent: {e}\nRaw response (cleaned): '{cleaned_response}'")
            parsed_response_dict = {"debate_answer": raw_model_response.strip()} # JSON 파싱 실패 시, 원본 응답을 debate_answer로 fallback
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
            "Reason": parsed_response_dict.get("Reason", "Single agent direct response."), # JSON에서 Reason 필드도 가져옴
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
        print(f"Model Raw Answer: {model_raw_answer}")
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