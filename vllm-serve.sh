nohup vllm serve Qwen/Qwen3-14B \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 > logs/vllm_server3.log 2>&1 &