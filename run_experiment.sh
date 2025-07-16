#!/bin/bash

VLLM_SERVER_URL="http://localhost:8000" 

MODEL_NAME="Qwen/Qwen3-14B" 

QUESTION_FILE="data/GSM8K/GSM8K_sampled100.json" 

# 모든 실험 결과가 저장될 최상위 디렉토리
BASE_OUTPUT_DIR="experiment_gsm_results"

# 기본 시스템 프롬프트 (Single Agent용 - Multi-Agent는 config4all.json 사용)
SYSTEM_PROMPT="You are a helpful and accurate assistant. Provide a direct answer to the question. The final answer MUST be a single numerical value (integer, decimal, or fraction like 'X/Y') WITHOUT any units, text, explanations, or parentheses. For percentages, output as a decimal (e.g., 9.09% should be 0.0909). For quantities with units (e.g., 500 kg, 5 minutes), convert to a pure numerical value in the most standard base unit for direct comparison (e.g., for 500 kg, output 500; for 5 minutes, output 5). If the question implies a specific unit, ensure the final answer is a pure number for that implied unit. Only output the final answer."

# Multi-Agent용 프롬프트 템플릿 파일 경로
CONFIG_PROMPT_PATH="code/utils/config4all_new.json"

# --- 1. BASE_OUTPUT_DIR 생성 (이미 존재하면 건너뜀) ---
mkdir -p "$BASE_OUTPUT_DIR"
echo "Experiment results will be saved in: $BASE_OUTPUT_DIR"

# --- 실험 1: temperature 0, 더미 텍스트 없음 (baseline) ---
echo -e "\n--- Running Experiment 1: GSM8K, Multi-Agent, Temp 0, No Noise ---"
EXP_NAME="multi_temp0.5_4"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_4"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 1: GSM8K, Multi-Agent, Temp 0, No Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 1: GSM8K, Multi-Agent, Temp 0, No Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 2: temperature 0, 더미 텍스트 있음 ---
echo -e "\n--- Running Experiment 2: GSM8K, Multi-Agent, Temp 0, With Noise ---"
EXP_NAME="multi_temp0"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"
DUMMY_NOISE_TEXT="Interesting fact: cats sleep for most of their lives."

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    -n "$DUMMY_NOISE_TEXT" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 2: GSM8K, Multi-Agent, Temp 0, With Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 2: GSM8K, Multi-Agent, Temp 0, With Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 3: temperature 0.5, 더미 텍스트 없음 ---
echo -e "\n--- Running Experiment 3: GSM8K, Multi-Agent, Temp 0.5, No Noise ---"
EXP_NAME="multi_temp0.5_1"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_1"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Running Experiment 3: GSM8K, Multi-Agent, Temp 0.5, No Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Running Experiment 3: GSM8K, Multi-Agent, Temp 0.5, No Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 4: temperature 0.5, 더미 텍스트 없음 ---
echo -e "\n--- Running Experiment 4: GSM8K, Multi-Agent, Temp 0.5, No Noise ---"
EXP_NAME="multi_temp0.5_2"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_2"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Running Experiment 4: GSM8K, Multi-Agent, Temp 0.5, No Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Running Experiment 4: GSM8K, Multi-Agent, Temp 0.5, No Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 5: temperature 0.5, 더미 텍스트 없음 ---
echo -e "\n--- Running Experiment 5: GSM8K, Multi-Agent, Temp 0.5, No Noise ---"
EXP_NAME="multi_temp0.5_3"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_3"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Running Experiment 5: GSM8K, Multi-Agent, Temp 0.5, No Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Running Experiment 5: GSM8K, Multi-Agent, Temp 0.5, No Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 6: temperature 0.5, 더미 텍스트 있음 ---
echo -e "\n--- Running Experiment 6: GSM8K, Multi-Agent, Temp 0, With Noise ---"
EXP_NAME="multi_temp0.5_cat1"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_cat1"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"
DUMMY_NOISE_TEXT="Interesting fact: cats sleep for most of their lives."

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    -n "$DUMMY_NOISE_TEXT" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 6: GSM8K, Multi-Agent, Temp 0.5, With Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 6: GSM8K, Multi-Agent, Temp 0.5, With Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 6: temperature 0.5, 더미 텍스트 있음 ---
echo -e "\n--- Running Experiment 6: GSM8K, Multi-Agent, Temp 0, With Noise ---"
EXP_NAME="multi_temp0.5_cat2"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_cat2"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"
DUMMY_NOISE_TEXT="Interesting fact: cats sleep for most of their lives."

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    -n "$DUMMY_NOISE_TEXT" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 7: GSM8K, Multi-Agent, Temp 0.5, With Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 7: GSM8K, Multi-Agent, Temp 0.5, With Noise completed. Results in $CURRENT_OUTPUT_DIR"

# --- 실험 6: temperature 0.5, 더미 텍스트 있음 ---
echo -e "\n--- Running Experiment 6: GSM8K, Multi-Agent, Temp 0, With Noise ---"
EXP_NAME="multi_temp0.5_cat3"
CURRENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi/temp0.5_cat3"
mkdir -p "$CURRENT_OUTPUT_DIR"
CURRENT_LOG_FILE="$CURRENT_OUTPUT_DIR/debate.log"
DUMMY_NOISE_TEXT="Interesting fact: cats sleep for most of their lives."

nohup python -u run_debate.py \
    -i "$QUESTION_FILE" \
    -o "$CURRENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0.5 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "$EXP_NAME" \
    -n "$DUMMY_NOISE_TEXT" \
    > "$CURRENT_LOG_FILE" 2>&1

echo "Experiment 8: GSM8K, Multi-Agent, Temp 0.5, With Noise launched. Waiting for completion... (Check $CURRENT_LOG_FILE)"
wait
echo "Experiment 8: GSM8K, Multi-Agent, Temp 0.5, With Noise completed. Results in $CURRENT_OUTPUT_DIR"

# 모든 실험이 완료된 후 메시지 출력
echo -e "\n--- All experiments completed. ---"
echo "Check logs and results in their respective subdirectories within $BASE_OUTPUT_DIR."