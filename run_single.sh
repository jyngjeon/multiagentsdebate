#!/bin/bash

# --- Configuration Variables (Adjust these to fit your environment) ---

# Path to your CIAR questions JSON file (e.g., /path/to/your/ciar_questions.json)
INPUT_JSON_FILE="data/CounterintuitiveQA/CIAR.json" 

# Directory to save evaluation results (e.g., /path/to/your/single_agent_results)
OUTPUT_DIR="results/single_results"

# URL of your local VLLM server (e.g., http://0.0.0.0:8000)
LOCAL_LLM_URL="http://0.0.0.0:8000" 

# Model name to use (must match the model loaded on your VLLM server)
MODEL_NAME="Qwen/Qwen3-14B" 

# Sampling temperature for LLM responses (lower values are more deterministic)
TEMPERATURE=0 

# System prompt for the single agent.
# Ensure this matches the strict format requirements for evaluation.
SYSTEM_PROMPT="You are a helpful and accurate assistant. Provide a direct answer to the question. The final answer MUST be a single numerical value (integer, decimal, or fraction like 'X/Y') WITHOUT any units, text, or explanations. For percentages, output as a decimal (e.g., 9.09% should be 0.0909). For quantities with units (e.g., 500 kg, 5 minutes), convert to a pure numerical value in the most standard base unit for direct comparison (e.g., for 500 kg, output 500; for 5 minutes, output 5). If the question implies a specific unit, ensure the final answer is a pure number for that implied unit. Only output the final answer."

# Log file path
LOG_FILE="logs/single.log"

# --- Execute the Script ---

echo "Starting single agent evaluation script in background..."
echo "Logs will be saved to: $LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR"
echo "--------------------------------------------------"

# Use nohup to run the Python script in the background.
# -u: Unbuffered binary stdout and stderr. Ensures logs are written immediately.
# > "$LOG_FILE" 2>&1: Redirects both standard output and standard error to $LOG_FILE.
# &: Runs the command in the background.

nohup python -u run_single.py \
    --input-json-file "$INPUT_JSON_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --local-llm-url "$LOCAL_LLM_URL" \
    --model-name "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --system-prompt "$SYSTEM_PROMPT" \
    > "$LOG_FILE" 2>&1 &

echo "Script launched. Check $LOG_FILE for output."
echo "You can view the progress using: tail -f $LOG_FILE"
echo "To stop the script, find its process ID (PID) using 'ps aux | grep run_single.py' and then 'kill <PID>'."