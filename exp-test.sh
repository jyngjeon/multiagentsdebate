#!/bin/bash

# --- 0. 환경 설정 변수 (이 부분들을 당신의 환경에 맞게 수정하세요) ---
# VLLM 서버의 URL (http://0.0.0.0:8000 또는 http://localhost:8000)
VLLM_SERVER_URL="http://localhost:8000" 

# VLLM 서버에 로드된 모델 이름
MODEL_NAME="Qwen/Qwen3-14B" 

# 테스트 결과 저장할 기본 디렉토리
BASE_OUTPUT_DIR="tempresult"

# 임시 질문 파일명
TEST_QUESTION_FILE="data/gsm-test.json"

# 기본 시스템 프롬프트 (Single Agent용)
# judge_prompt_last2와 moderator_prompt에 있는 것과 유사하게 엄격하게 유지
SYSTEM_PROMPT="You are a helpful and accurate assistant. Provide a direct answer to the question. The final answer MUST be a single numerical value (integer, decimal, or fraction like 'X/Y') WITHOUT any units, text, explanations, or parentheses. For percentages, output as a decimal (e.g., 9.09% should be 0.0909). For quantities with units (e.g., 500 kg, 5 minutes), convert to a pure numerical value in the most standard base unit for direct comparison (e.g., for 500 kg, output 500; for 5 minutes, output 5). If the question implies a specific unit, ensure the final answer is a pure number for that implied unit. Only output the final answer."

# 기본 프롬프트 템플릿 파일 (Multi-Agent용)
CONFIG_PROMPT_PATH="code/utils/config4all.json"

# --- 2. Single Agent 테스트 실행 ---
echo -e "\n--- Running Single Agent Test ---"
SINGLE_AGENT_OUTPUT_DIR="$BASE_OUTPUT_DIR/single_agent_test"
mkdir -p "$SINGLE_AGENT_OUTPUT_DIR" # 결과 디렉토리 생성
SINGLE_AGENT_LOG="single_test.log"

nohup python -u run_single.py \
    -i "$TEST_QUESTION_FILE" \
    -o "$SINGLE_AGENT_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    -s "$SYSTEM_PROMPT" \
    --exp-name "single_agent_basic_test" \
    > "$SINGLE_AGENT_LOG" 2>&1 &

echo "Single Agent test launched. Check $SINGLE_AGENT_LOG for progress: tail -f $SINGLE_AGENT_LOG"

# --- 3. Multi-Agent Debate 테스트 실행 ---
echo -e "\n--- Running Multi-Agent Debate Test ---"
DEBATE_OUTPUT_DIR="$BASE_OUTPUT_DIR/multi_agent_test"
mkdir -p "$DEBATE_OUTPUT_DIR" # 결과 디렉토리 생성
DEBATE_LOG="debate_test.log"

nohup python -u run_debate.py \
    -i "$TEST_QUESTION_FILE" \
    -o "$DEBATE_OUTPUT_DIR" \
    -lu "$VLLM_SERVER_URL" \
    -m "$MODEL_NAME" \
    -t 0 \
    -c "$CONFIG_PROMPT_PATH" \
    --exp-name "multi_agent_basic_test" \
    > "$DEBATE_LOG" 2>&1 &

echo "Multi-Agent Debate test launched. Check $DEBATE_LOG for progress: tail -f $DEBATE_LOG"

echo -e "\n--- Tests launched. Please wait for completion. ---"
echo "Results will be in $BASE_OUTPUT_DIR"
echo "Check individual logs for details: $SINGLE_AGENT_LOG and $DEBATE_LOG"