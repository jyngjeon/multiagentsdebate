#!/bin/bash

INPUT_JSON_FILE="data/CounterintuitiveQA/CIAR_new.json" 

OUTPUT_DIR="debate_results"
LOCAL_LLM_URL="http://0.0.0.0:8000" 
MODEL_NAME="Qwen/Qwen3-14B" 

TEMPERATURE=0

# OpenAI API 키 (로컬 LLM을 사용하지 않거나, fallback 용도일 경우)
# OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# 프롬프트 템플릿 파일 경로 (예: code/utils/config4all.json)
# run_debate.py 스크립트의 위치를 기준으로 한 상대 경로입니다.
CONFIG_PROMPT_PATH="code/utils/config4all_new.json"

# 로그 파일 경로
LOG_FILE="logs/debate_result.log"

# --- 스크립트 실행 ---

echo "Starting debate script in background..."
echo "Logs will be saved to: $LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR"
echo "--------------------------------------------------"

# nohup을 사용하여 백그라운드에서 Python 스크립트 실행
# 출력(stdout)과 에러(stderr)를 모두 $LOG_FILE로 리다이렉션합니다.
# 마지막의 '&'는 스크립트를 백그라운드 프로세스로 실행함을 의미합니다.

# OpenAI API 키가 설정된 경우와 아닌 경우를 분기
if [ -n "$OPENAI_API_KEY" ]; then
    nohup python -u run_debate.py \
        --input-json-file "$INPUT_JSON_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --local-llm-url "$LOCAL_LLM_URL" \
        --model-name "$MODEL_NAME" \
        --temperature "$TEMPERATURE" \
        --api-key "$OPENAI_API_KEY" \
        --config-prompt-path "$CONFIG_PROMPT_PATH" \
        > "$LOG_FILE" 2>&1 &
else
    nohup python -u run_debate.py \
        --input-json-file "$INPUT_JSON_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --local-llm-url "$LOCAL_LLM_URL" \
        --model-name "$MODEL_NAME" \
        --temperature "$TEMPERATURE" \
        --config-prompt-path "$CONFIG_PROMPT_PATH" \
        > "$LOG_FILE" 2>&1 &
fi

echo "Script launched. Check $LOG_FILE for output."
echo "You can view the progress using: tail -f $LOG_FILE"
echo "To stop the script, find its process ID (PID) using 'ps aux | grep run_debate.py' and then 'kill <PID>'."