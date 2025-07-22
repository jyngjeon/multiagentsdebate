# summarize_results.py

import os
import pandas as pd
import json
import argparse
import re # 정규식 라이브러리 임포트
from collections import defaultdict

def find_jsonl_files(root_dir: str):
    """지정된 디렉토리와 그 하위에서 모든 .jsonl 파일을 찾습니다."""
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def parse_run_number(file_path: str) -> str:
    """파일 경로에서 'runX' 형태의 실행 번호를 추출합니다."""
    parts = file_path.split(os.sep)
    for part in reversed(parts):
        if part.startswith("run"):
            return part
    return "unknown_run"

def summarize_results_to_excel(root_dir: str, output_excel_path: str):
    """
    실험 결과(.jsonl)들을 취합하여 하나의 데이터프레임으로 만들고 엑셀로 저장합니다.
    """
    jsonl_files = find_jsonl_files(root_dir)
    if not jsonl_files:
        print(f"Error: No .jsonl files found in '{root_dir}'. Please check the path.")
        return

    print(f"Found {len(jsonl_files)} result files. Processing...")

    results_dict = defaultdict(dict)

    for file_path in jsonl_files:
        run_name = parse_run_number(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            # <<< --- 수정된 부분 시작 --- >>>
            # 파일 전체 내용을 한 번에 읽습니다.
            content = f.read()
            # 정규식을 사용하여 '{...}' 형태의 모든 JSON 블록을 찾습니다.
            # re.DOTALL 옵션은 줄바꿈 문자를 포함하여 모든 문자를 매칭시킵니다.
            json_objects_str = re.findall(r'\{.*?\}', content, re.DOTALL)

            for json_str in json_objects_str:
                # <<< --- 수정된 부분 끝 --- >>>
                try:
                    data = json.loads(json_str)
                    question_id = data.get("question_id_in_file", "N/A")
                    model_answer = data.get("model_final_answer")
                    ground_truth = data.get("ground_truth_answer")

                    results_dict[question_id][run_name] = model_answer
                    if "정답" not in results_dict[question_id]:
                        results_dict[question_id]["정답"] = ground_truth
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a block in {file_path} due to JSON decoding error. Block: {json_str[:100]}...")

    # --- (이하 코드는 동일) ---
    df = pd.DataFrame.from_dict(results_dict, orient='index')

    if "정답" in df.columns:
        truth_column = df.pop("정답")
        df["정답"] = truth_column

    run_columns = sorted([col for col in df.columns if col.startswith("run")],
                         key=lambda x: int(x.replace("run", "")))
    other_columns = [col for col in df.columns if not col.startswith("run")]
    df = df[run_columns + other_columns]
    
    def count_correct(row):
        correct_answer = row['정답']
        if not isinstance(correct_answer, str):
            return 0
        count = 0
        for col in run_columns:
            model_answer = str(row.get(col, '')).strip().upper()
            if model_answer == correct_answer.strip().upper():
                count += 1
        return count

    df['정답수'] = df.apply(count_correct, axis=1)

    accuracy_row = {}
    for col in run_columns:
        correct_predictions = (df[col].astype(str).str.strip().str.upper() == df['정답'].str.strip().str.upper()).sum()
        total_predictions = df[col].notna().sum()
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        accuracy_row[col] = f"{accuracy:.2f}%"
    
    accuracy_df = pd.DataFrame([accuracy_row], index=['전체 정확도 (%)'])
    df = pd.concat([df, accuracy_df])

    try:
        df.to_excel(output_excel_path)
        print(f"\n✅ Successfully created summary file: {output_excel_path}")
    except Exception as e:
        print(f"\n❌ Error saving to Excel: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results from .jsonl files into an Excel sheet.")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="The base directory containing all experiment result folders.")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="The path for the output Excel file.")
    args = parser.parse_args()
    
    summarize_results_to_excel(args.input_dir, args.output_file)