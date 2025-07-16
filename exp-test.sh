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
    --exp-name "single_agent_test" \
    > "$SINGLE_AGENT_LOG" 2>&1 &

echo "Single Agent test launched. Check $SINGLE_AGENT_LOG for progress: tail -f $SINGLE_AGENT_LOG"