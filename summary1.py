import json
import os
import argparse # Import argparse

def print_raw_answers_summary(json_file_path: str):
    """
    Parses a debate results JSON file and prints the question ID,
    original correct answers, and the model's raw answer for each entry.

    Args:
        json_file_path (str): The path to the JSON file containing all debate results.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_file_path}: {e}")
        return

    print("\n--- Debate Results Summary ---")
    for entry in data:
        question_id = entry.get("question_id", "N/A")
        model_raw_answer = entry.get("model_raw_answer", "N/A")
        original_correct_answers = entry.get("original_correct_answers", [])
        is_correct = entry.get("is_correct", False) # Also print if it was judged correct

        print(f"Question ID: {question_id}")
        print(f"  Original Correct Answers: {original_correct_answers}")
        print(f"  Model Raw Answer: {model_raw_answer}")
        print(f"  Judged Correct: {is_correct}") # Added
        print("-" * 40) # Separator
    
    print("\n--- End of Summary ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print summary of debate results from a JSON file.")
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="Path to the overall debate results summary JSON file (e.g., debate_results/overall_debate_results_summary_YYYYMMDD_HHMMSS.json).")
    
    args = parser.parse_args()
    
    print_raw_answers_summary(args.file)