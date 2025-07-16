# analyze_results.py

import json
import os
import argparse
from datetime import datetime

def analyze_debate_results(json_file_path: str):
    """
    디베이트/싱글 에이전트 결과 JSON 파일을 분석하여 전체 정답률을 계산하고
    오답인 문제들의 상세 정보를 출력합니다.

    Args:
        json_file_path (str): 모든 디베이트/싱글 에이전트 결과가 담긴 JSON 파일의 경로.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: Input file not found at '{json_file_path}'")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_file_path}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{json_file_path}': {e}")
        return

    total_questions = len(data)
    correct_count = 0
    incorrect_questions = []

    print(f"\n--- Analyzing Results from: '{os.path.basename(json_file_path)}' ---")
    print(f"Total Questions Processed: {total_questions}")
    print("-" * 50)

    for i, entry in enumerate(data):
        question_id = entry.get("question_id", i + 1) # question_id가 없으면 순번 사용
        is_correct = entry.get("is_correct", False)
        model_raw_answer = entry.get("model_raw_answer", "N/A")
        original_correct_answers = entry.get("original_correct_answers", [])
        question_text = entry.get("question_original_text", entry.get("question", "N/A")) # 원본 질문 텍스트

        if is_correct:
            correct_count += 1
        else:
            incorrect_questions.append({
                "question_id": question_id,
                "question": question_text,
                "model_raw_answer": model_raw_answer,
                "original_correct_answers": original_correct_answers,
                "model_normalized_answer": entry.get("model_normalized_answer", "N/A"),
                "normalized_correct_answers": entry.get("normalized_correct_answers", [])
            })
        
        # 각 문제별 결과 간략하게 출력
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"Question ID: {question_id} | Status: {status}")
        print(f"  Model Answer (Raw): '{model_raw_answer}'")
        print(f"  Expected Answer: {original_correct_answers}")
        if not is_correct: # 오답일 경우 상세 파싱 정보 출력
            print(f"  Model Answer (Normalized): '{entry.get('model_normalized_answer', 'N/A')}'")
            print(f"  Expected Answer (Normalized): {entry.get('normalized_correct_answers', [])}")
        print("-" * 20)


    print(f"\n===== Analysis Summary =====")
    print(f"Total Questions: {total_questions}")
    print(f"Correctly Answered: {correct_count}")
    overall_accuracy = correct_count / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print("-" * 50)

    if incorrect_questions:
        print(f"\n===== Incorrectly Answered Questions ({len(incorrect_questions)}/{total_questions}) =====")
        for i, item in enumerate(incorrect_questions):
            print(f"--- Incorrect Question {i+1} (ID: {item['question_id']}) ---")
            print(f"Question: {item['question'][:100]}...") # 질문이 길면 잘라서 출력
            print(f"  Model Raw Answer: '{item['model_raw_answer']}'")
            print(f"  Model Normalized Answer: '{item['model_normalized_answer']}'")
            print(f"  Original Correct Answers: {item['original_correct_answers']}")
            print(f"  Normalized Correct Answers: {item['normalized_correct_answers']}")
            print("-" * 50)
    else:
        print("\nAll questions were answered correctly! 🎉")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze debate/single agent results from a JSON summary file.")
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="Path to the overall results summary JSON file (e.g., overall_debate_results_summary_*.json or single_agent_results_summary_*.json).")
    
    args = parser.parse_args()
    
    analyze_debate_results(args.file)