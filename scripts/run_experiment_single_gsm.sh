#!/bin/bash

# --- 0. 환경 설정 변수 (이 부분들을 당신의 환경에 맞게 수정하세요) ---
# VLLM 서버의 URL (예: http://0.0.0.0:8000 또는 http://localhost:8000)
VLLM_SERVER_URL="http://localhost:8000" 

# VLLM 서버에 로드된 모델 이름
MODEL_NAME="Qwen/Qwen3-14B" 

# 질문 JSON 파일 경로 (예: ciar_cleaned.json 또는 GSM8K_sampled100.json)
QUESTION_FILE="data/GSM8K/GSM8K_sampled100.json" 

# 모든 실험 결과가 저장될 최상위 디렉토리
BASE_OUTPUT_DIR="experiment_gsm_results"
DUMMY_NOISE_TEXT=" Interesting fact: cats sleep for most of their lives."


# --- 1. BASE_OUTPUT_DIR 생성 (이미 존재하면 건너뜀) ---
mkdir -p "$BASE_OUTPUT_DIR"
echo "Experiment results will be saved in: $BASE_OUTPUT_DIR"

# --- 실험 1: Single Agent, temperature 0, 더미 텍스트 없음 ---
echo -e "\n--- Running Experiment 1: Single Agent, Temp 0, No Noise ---"
EXP_NAME="single_temp0" # 실험 이름 (결과 파일명에 포함)
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp0" # 결과 저장될 고유 폴더
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log" # 이 실험의 로그 파일

nohup python -u run_single.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    --exp-name "$EXP_NAME" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 1: Single Agent, Temp 0, No Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait 
echo "Experiment 1: Single Agent, Temp 0, No Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 2: temperature 0, 더미 텍스트 있음 ---
echo -e "\n--- Running Experiment 2: Single Agent, Temp 0, With Noise ---"
EXP_NAME="single_temp0_cat"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp0_cat"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log"

nohup python -u run_single.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    --exp-name "$EXP_NAME" \
    -n "$DUMMY_NOISE_TEXT" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 2: Single Agent, Temp 0, With Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 2: Single Agent, Temp 0, With Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- Experiment 3-7: Single Agent, temperature 0.5, 더미 텍스트 없음 (5회 반복) ---
# TEMPERATURE_SETTING=0.5

# echo -e "\n--- Running Experiments 3-7: Single Agent, Temp $TEMPERATURE_SETTING, No Noise (5 runs) ---"
# for i in {1..5}; do
#     EXP_NAME="single_temp0.5_$i"
#     CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp0.5_$i"
#     mkdir -p "$CURRENT_OUTPUT_DIR"
#     CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log"

#     nohup python -u run_single.py \
#         -i "$QUESTION_FILE" \
#         -o "$CURRENT_OUTPUT_DIR" \
#         -lu "$VLLM_SERVER_URL" \
#         -m "$MODEL_NAME" \
#         -t "$TEMPERATURE_SETTING" \
#         --exp-name "$EXP_NAME" \
#         > "$CURRENT_LOG_FILE" 2>&1

#     echo "Experiment $((2 + i)): Single Agent, Temp $TEMPERATURE_SETTING, No Noise (Run ${i}) launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
#     wait
#     echo "Experiment $((2 + i)): Single Agent, Temp $TEMPERATURE_SETTING, No Noise (Run ${i}) completed. Results in $CURRENT_OUTPUT_DIR"
# done

# # --- Experiment 8-12: Single Agent, temperature 0.5, 더미 텍스트 있음 (5회 반복) ---
# TEMPERATURE_SETTING=0.5

# echo -e "\n--- Running Experiments 8-12: Single Agent, Temp $TEMPERATURE_SETTING, With Noise (5 runs) ---"
# for i in {1..5}; do
#     EXP_NAME="single_temp0.5_cat$i"
#     CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp0.5_cat$i"
#     mkdir -p "$CURRENT_OUTPUT_DIR"
#     CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log"

#     nohup python -u run_single.py \
#         -i "$QUESTION_FILE" \
#         -o "$CURRENT_OUTPUT_DIR" \
#         -lu "$VLLM_SERVER_URL" \
#         -m "$MODEL_NAME" \
#         -t "$TEMPERATURE_SETTING" \
#         --exp-name "$EXP_NAME" \
#         -n "$DUMMY_NOISE_TEXT" \
#         > "$CURRENT_LOG_FILE" 2>&1

#     echo "Experiment $((7 + i)): Single Agent, Temp $TEMPERATURE_SETTING, With Noise (Run ${i}) launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
#     wait
#     echo "Experiment $((7 + i)): Single Agent, Temp $TEMPERATURE_SETTING, With Noise (Run ${i}) completed. Results in $CURRENT_OUTPUT_DIR"
# done

# # --- Experiment 13-17: Single Agent, temperature 1, 더미 텍스트 없음 (5회 반복) ---
# TEMPERATURE_SETTING=1

# echo -e "\n--- Running Experiments 13-17: Single Agent, Temp $TEMPERATURE_SETTING, No Noise (5 runs) ---"
# for i in {1..5}; do
#     EXP_NAME="single_temp1_$i"
#     CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp1_$i"
#     mkdir -p "$CURRENT_OUTPUT_DIR"
#     CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log"

#     nohup python -u run_single.py \
#         -i "$QUESTION_FILE" \
#         -o "$CURRENT_OUTPUT_DIR" \
#         -lu "$VLLM_SERVER_URL" \
#         -m "$MODEL_NAME" \
#         -t "$TEMPERATURE_SETTING" \
#         --exp-name "$EXP_NAME" \
#         > "$CURRENT_LOG_FILE" 2>&1

#     echo "Experiment $((12 + i)): Single Agent, Temp $TEMPERATURE_SETTING, No Noise (Run ${i}) launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
#     wait
#     echo "Experiment $((12 + i)): Single Agent, Temp $TEMPERATURE_SETTING, No Noise (Run ${i}) completed. Results in $CURRENT_OUTPUT_DIR"
# done

# # --- Experiment 18-22: Single Agent, temperature 1, 더미 텍스트 있음 (5회 반복) ---
# TEMPERATURE_SETTING=1

# echo -e "\n--- Running Experiments 18-22: Single Agent, Temp $TEMPERATURE_SETTING, With Noise (5 runs) ---"
# for i in {1..5}; do
#     EXP_NAME="single_temp1_cat$i"
#     CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single/temp1_cat$i"
#     mkdir -p "$CURRENT_OUTPUT_DIR"
#     CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/single.log"

#     nohup python -u run_single.py \
#         -i "$QUESTION_FILE" \
#         -o "$CURRENT_OUTPUT_DIR" \
#         -lu "$VLLM_SERVER_URL" \
#         -m "$MODEL_NAME" \
#         -t "$TEMPERATURE_SETTING" \
#         --exp-name "$EXP_NAME" \
#         -n "$DUMMY_NOISE_TEXT" \
#         > "$CURRENT_LOG_FILE" 2>&1

#     echo "Experiment $((18 + i)): Single Agent, Temp $TEMPERATURE_SETTING, With Noise (Run ${i}) launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
#     wait
#     echo "Experiment $((18 + i)): Single Agent, Temp $TEMPERATURE_SETTING, With Noise (Run ${i}) completed. Results in $CURRENT_OUTPUT_DIR"
# done

echo -e "\n--- All Single Agent experiments completed. ---"
echo "Check logs and results in their respective subdirectories within $BASE_OUTPUT_DIR."