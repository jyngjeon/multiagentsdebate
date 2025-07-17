# run_single.py

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import re

# Agent 클래스는 code/utils/agent.py에 있습니다.
from code.utils.agent import Agent

def extract_boxed_answer(solution_text: str) -> str | None:
    """
    주어진 텍스트에서 \\boxed{...} 안의 원본 내용을 그대로 추출합니다.
    중첩된 괄호를 올바르게 처리하여 내용이 잘리는 문제를 해결합니다.
    박스 안의 답을 찾지 못하면 None을 반환합니다.
    """
    if not isinstance(solution_text, str):
        return None

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

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single LLM agent's accuracy on math questions.")
    parser.add_argument("-i", "--input-json-file", type=str, required=True,
                        help="Path to the input JSON file containing questions.")
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
    
    # --- 시스템 프롬프트 기본값 수정 ---
    # LLM이 단계별 설명을 하고 \boxed{}로 답을 내놓도록 지시하는 프롬프트로 변경합니다.
    parser.add_argument("-s", "--system-prompt", type=str,
                        default="You are a brilliant mathematician. Please solve the following math problem.\n\n**Instructions:**\n1. First, think step-by-step and write down your detailed reasoning and calculations.\n2. After you have finished your reasoning, provide the final answer.\n3. The final answer **MUST** be enclosed in a `\\boxed{}` environment. For example: `\\boxed{42}` or `\\boxed{\\frac{1}{2}}`.",
                        help="The system prompt given to the single agent.")
    
    # --- 노이즈 및 실험 이름 인자 ---
    parser.add_argument("-n", "--noise-text", type=str, default="",
                        help="Optional: Text to append as noise to the end of each question. If empty, no noise is added.")
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
        
        # 매 질문마다 에이전트의 메모리를 초기화하고 시스템 프롬프트를 다시 설정합니다.
        single_agent.memory_lst = []
        single_agent.set_meta_prompt(args.system_prompt)

        # 질문을 프롬프트에 추가하고 답변을 요청합니다.
        single_agent.add_event(question_with_noise)
        raw_model_response = single_agent.ask()

        # --- 결과 추출 및 정확도 계산 (수정된 로직) ---
        model_extracted_answer = extract_boxed_answer(raw_model_response)

        # JSON 파일의 정답은 이미 추출된 순수 문자열 리스트입니다.
        ground_truth_answers = [str(ans).strip() for ans in question_entry.get("answer", [])]

        is_correct = False
        if model_extracted_answer is not None:
            # 추출된 답이 정답 리스트에 있는지 직접 비교
            if model_extracted_answer in ground_truth_answers:
                is_correct = True
        
        if is_correct:
            correct_answers_count += 1
        
        # --- 최종 결과 JSON에 저장할 정보 구성 ---
        result_entry = {
            "question_id": i + 1,
            "question": original_question_text,
            "question_with_noise": question_with_noise,
            "noise_added": bool(args.noise_text),
            "noise_text_used": args.noise_text,
            "ground_truth_answers": ground_truth_answers,
            "model_full_output": raw_model_response,
            "model_extracted_answer": model_extracted_answer,
            "is_correct": is_correct,
            "agent_memory": single_agent.memory_lst.copy()
        }

        all_results.append(result_entry)

        # --- 진행 상황 출력 ---
        print(f"\n--- Question {i+1}/{total_questions} ---")
        if is_correct:
            print(f"Result: CORRECT (Answered: '{model_extracted_answer}' | Expected: {ground_truth_answers})")
        else:
            print(f"Result: INCORRECT (Answered: '{model_extracted_answer}' | Expected: {ground_truth_answers})")
        
        print(f"Accuracy so far: {correct_answers_count}/{i+1} ({correct_answers_count / (i+1):.2%})\n")

    # --- 전체 정확도 계산 및 요약 출력 ---
    overall_accuracy = correct_answers_count / total_questions if total_questions > 0 else 0
    print(f"\n===== Single Agent Evaluation Summary =====")
    print(f"Total Questions: {total_questions}")
    print(f"Correctly Answered: {correct_answers_count}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    # --- 모든 결과를 하나의 요약 파일로 저장 ---
    summary_file_name = f"single_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file_path = os.path.join(args.output_dir, summary_file_name)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All single agent results summarized in: {summary_file_path}")
