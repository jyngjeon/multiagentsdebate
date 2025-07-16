import json
import re
import argparse
import os

def process_jsonl_file(input_file_path: str, output_file_path: str, num_samples: int = 100):
    """
    Processes a JSONL file, extracts 'question' and numerical 'answer',
    adds a 'question_id', and saves a sampled subset to a new JSON file.

    Args:
        input_file_path (str): Path to the input .jsonl file.
        output_file_path (str): Path to save the new .json file.
        num_samples (int): Number of questions to sample from the beginning of the file.
    """
    processed_questions = []
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if i >= num_samples:
                    break # Stop after processing the desired number of samples
                
                try:
                    data = json.loads(line)
                    
                    question_text = data.get("question")
                    raw_answer = data.get("answer")
                    
                    if question_text and raw_answer:
                        # Extract the number after "#### "
                        answer_match = re.search(r'####\s*([-+]?\d+\.?\d*(?:/\d+\.?\d*)?)', raw_answer)
                        
                        extracted_answer = None
                        if answer_match:
                            extracted_answer = answer_match.group(1).strip()
                        
                        if extracted_answer:
                            processed_questions.append({
                                "question_id": i + 1, # Add 1-based question_id
                                "question": question_text,
                                "answer": [extracted_answer] # Store as a list
                            })
                        else:
                            print(f"Warning: Could not extract answer for question {i+1} (ID: {data.get('id', 'N/A')}). Skipping.")
                    else:
                        print(f"Warning: Missing 'question' or 'answer' for line {i+1}. Skipping.")
                        
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i+1}: {e}. Skipping line.")
                
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Save the processed questions to the output JSON file
    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(processed_questions, outfile, ensure_ascii=False, indent=4)
        print(f"Successfully created '{output_file_path}' with {len(processed_questions)} questions.")
    except Exception as e:
        print(f"Error saving output file '{output_file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample questions from a JSONL file and create a new JSON file.")
    parser.add_argument("-i", "--input-jsonl", type=str, required=True,
                        help="Path to the input .jsonl file (e.g., test.jsonl).")
    parser.add_argument("-o", "--output-json", type=str, required=True,
                        help="Path to save the new .json file (e.g., sampled_test_questions.json).")
    parser.add_argument("-n", "--num-samples", type=int, default=100,
                        help="Number of questions to sample from the beginning of the file.")
    
    args = parser.parse_args()
    
    process_jsonl_file(args.input_jsonl, args.output_json, args.num_samples)