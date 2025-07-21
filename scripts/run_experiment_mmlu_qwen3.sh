#!/bin/bash

# 실행: nohup bash run_experiment_math_lv5_2.sh > experiment_math_lv5_2.log 2>&1 &

# --- 0. 환경 설정 변수 (이 부분들을 당신의 환경에 맞게 수정하세요) ---
# VLLM 서버의 URL (예: http://0.0.0.0:8000 또는 http://localhost:8000)
VLLM_SERVER_URL="http://localhost:8000" 

# VLLM 서버에 로드된 모델 이름
MODEL_NAME="Qwen/Qwen3-14B" 

# 데이터셋 질문 파일 경로
QUESTION_FILE="data/mmlu/college_mathematics.json"

# 모든 실험 결과가 저장될 최상위 디렉토리
BASE_OUTPUT_DIR="results/"

# Multi-Agent용 프롬프트 템플릿 파일 경로
CONFIG_PROMPT_PATH="code/utils/config4all_mmlu.json" 

# 노이즈 텍스트 정의 (모든 노이즈 실험에서 동일하게 사용)
COMMON_NOISE_TEXT=""

# --- 1. BASE_OUTPUT_DIR 생성 (이미 존재하면 건너뜀) ---
mkdir -p "$BASE_OUTPUT_DIR"
echo "Experiment results will be saved in: $BASE_OUTPUT_DIR"

# 총 실행된 실험 개수를 추적하는 변수
EXPERIMENT_COUNT=0

# --- 실험 함수 정의 (코드 중복 방지를 위해) ---

# Multi-Agent Debate 실험 함수 (백그라운드 &로 실행)
run_multi_agent_experiment() {
    local base_output_dir_arg=$1
    local temp_setting=$2
    local noise_text=$3
    local category_base_name=$4
    local run_num=$5

    ((EXPERIMENT_COUNT++))

    local exp_name="multi_${category_base_name}_run${run_num}"
    # 'current_output_dir' (소문자)로 변수명 정의
    local current_output_dir="$base_output_dir_arg/multi/${category_base_name}/run${run_num}" 
    # 'current_output_dir' (소문자)를 사용하여 로그 파일 경로 지정 (수정됨)
    local current_log_file="$current_output_dir/debate.log"

    echo "DEBUG: Creating directory: '$current_output_dir'"
    # 'current_output_dir' (소문자)를 사용하여 디렉토리 생성 (수정됨)
    mkdir -p "$current_output_dir" || { echo "ERROR: Failed to create directory '$current_output_dir'. Exiting."; exit 1; }
    
    echo -e "\n--- Starting Experiment $EXPERIMENT_COUNT (Multi-Agent Parallel): Temp $temp_setting, Noise: ${noise_text:+Yes} (Run ${run_num}) ---"
    
    nohup python -u run_debate_math.py \
        -i "$QUESTION_FILE" \
        -o "$current_output_dir" \
        -lu "$VLLM_SERVER_URL" \
        -m "$MODEL_NAME" \
        -t "$temp_setting" \
        -c "$CONFIG_PROMPT_PATH" \
        --exp-name "$exp_name" \
        -n "$noise_text" \
        > "$current_log_file" 2>&1 &
        
    echo "Experiment $EXPERIMENT_COUNT launched in background. Log: $current_log_file"
}

# Single Agent 실험 함수 (백그라운드 &로 실행)
run_single_agent_experiment() {
    local base_output_dir_arg=$1
    local temp_setting=$2
    local noise_text=$3
    local category_base_name=$4
    local run_num=$5

    ((EXPERIMENT_COUNT++))

    local exp_name="single_${category_base_name}_run${run_num}"
    # 'current_output_dir' (소문자)로 변수명 정의
    local current_output_dir="$base_output_dir_arg/single/${category_base_name}/run${run_num}"
    # 'current_output_dir' (소문자)를 사용하여 로그 파일 경로 지정 (수정됨)
    local current_log_file="$current_output_dir/single_agent.log"

    echo "DEBUG: Creating directory: '$current_output_dir'"
    # 'current_output_dir' (소문자)를 사용하여 디렉토리 생성 (수정됨)
    mkdir -p "$current_output_dir" || { echo "ERROR: Failed to create directory '$current_output_dir'. Exiting."; exit 1; }

    echo -e "\n--- Starting Experiment $EXPERIMENT_COUNT (Single Agent Parallel): Temp $temp_setting, Noise: ${noise_text:+Yes} (Run ${run_num}) ---"
    
    nohup python -u run_single_math.py \
        -i "$QUESTION_FILE" \
        -o "$current_output_dir" \
        -lu "$VLLM_SERVER_URL" \
        -m "$MODEL_NAME" \
        -t "$temp_setting" \
        --exp-name "$exp_name" \
        -n "$noise_text" \
        > "$current_log_file" 2>&1 &

    echo "Experiment $EXPERIMENT_COUNT launched in background. Log: $current_log_file"
}


# --- 모든 실험 실행 ---

# --------------------------------------------------------------------
# 특정 실험만 실행하고 싶다면, 원하지 않는 블록을 주석 처리하세요.
# 예: 아래 Multi-Agent 실험 전체를 주석 처리하려면 블록을 선택하고 Ctrl+/ 를 누르세요.
# --------------------------------------------------------------------

# --- Multi-Agent Debate 실험 ---

# 1. Multi-Agent, Temp 0, 노이즈 있음 (1회)
CATEGORY_BASE_NAME="multi_temp0_mislead"
run_multi_agent_experiment "$BASE_OUTPUT_DIR" 0 "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" "1"

# 2. Multi-Agent, Temp 0.5, 노이즈 있음 (5회 반복)
TEMPERATURE_SETTING=0.5
CATEGORY_BASE_NAME="multi_temp0.5_mislead" 
echo -e "\n--- Preparing Multi-Agent, Temp $TEMPERATURE_SETTING, with mislead (5 runs) ---"
for i in {1..5}; do
    run_multi_agent_experiment "$BASE_OUTPUT_DIR" "$TEMPERATURE_SETTING" "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" "$i" 
done

# 3. Multi-Agent, Temp 1.0, 노이즈 있음 (5회 반복)
TEMPERATURE_SETTING=1.0
CATEGORY_BASE_NAME="multi_temp1.0_mislead" 
echo -e "\n--- Preparing Multi-Agent, Temp $TEMPERATURE_SETTING, with mislead (5 runs) ---"
for i in {1..5}; do
    run_multi_agent_experiment "$BASE_OUTPUT_DIR" "$TEMPERATURE_SETTING" "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" "$i" 
done


# --- Single Agent 실험 ---

# 1. Single Agent, Temp 0, 노이즈 있음 (1회)
CATEGORY_BASE_NAME="single_temp0_mislead"
run_single_agent_experiment "$BASE_OUTPUT_DIR" 0 "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" 1

# 2. Single Agent, Temp 0.5, 노이즈 있음 (5회 반복)
TEMPERATURE_SETTING=0.5
CATEGORY_BASE_NAME="single_temp0.5_mislead" 
echo -e "\n--- Preparing Single Agent, Temp $TEMPERATURE_SETTING, with mislead (5 runs) ---"
for i in {1..5}; do
    run_single_agent_experiment "$BASE_OUTPUT_DIR" "$TEMPERATURE_SETTING" "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" "$i" 
done

# 5. Single Agent, Temp 1.0, 노이즈 있음 (5회 반복)
TEMPERATURE_SETTING=1.0
CATEGORY_BASE_NAME="single_temp1.0_mislead" 
echo -e "\n--- Preparing Single Agent, Temp $TEMPERATURE_SETTING, with mislead (5 runs) ---"
for i in {1..5}; do
    run_single_agent_experiment "$BASE_OUTPUT_DIR" "$TEMPERATURE_SETTING" "$COMMON_NOISE_TEXT" "$CATEGORY_BASE_NAME" "$i" 
done

echo -e "\n--- All parallel experiments launched. Total experiments: $EXPERIMENT_COUNT ---"
echo "You can check their individual logs for progress."
echo "Use 'jobs' or 'ps aux | grep python' to monitor them."
echo "Use 'wait' command in your terminal to wait for all background jobs to complete, or 'kill <PID>' to stop them."
