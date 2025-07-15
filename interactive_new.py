# interactive_new.py

import os
import json
import argparse
from tqdm import tqdm # batch_processing에서만 쓰이지만, interactive.py에도 추가해두면 코드 복사 붙여넣기 편리
from datetime import datetime

from code.utils.debate import DebatePlayer, Debate, NAME_LIST # NAME_LIST도 이제 debate_core에서 가져옴

if __name__ == "__main__":
    # --- Add argparse for VLLM URL and model name ---
    parser = argparse.ArgumentParser(description="Run an interactive multi-agent debate with LLMs.")
    parser.add_argument("-lu", "--local-llm-url", type=str, 
                        default="http://0.0.0.0:8000", # Default VLLM URL
                        help="URL of the local LLM server (e.g., http://0.0.0.0:8000)")
    parser.add_argument("-m", "--model-name", type=str, 
                        default="Qwen/Qwen3-14B", # Default VLLM model name
                        help="Model name to use for the debate (e.g., Qwen/Qwen3-14B).")
    parser.add_argument("-t", "--temperature", type=float, 
                        default=0, # Default temperature
                        help="Sampling temperature for LLM responses.")
    parser.add_argument("-k", "--api-key", type=str, 
                        default=None, # Optional OpenAI API key
                        help="OpenAI API key (only if not using local LLM or for fallback).")
    parser.add_argument("-c", "--config-prompt-path", type=str,
                        default="code/utils/config4all.json", # 프롬프트 템플릿 파일 경로
                        help="Path to the JSON file containing prompt templates (e.g., config4all.json).")

    args = parser.parse_args()

    current_script_path = os.path.abspath(__file__)
    MAD_path_for_config = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트의 디렉토리
    config_full_path = os.path.join(MAD_path_for_config, args.config_prompt_path) # 상대 경로를 절대 경로로

    while True:
        debate_topic_input = "" # 사용자 입력 변수명 변경 (config의 debate_topic과 구분)
        while debate_topic_input == "":
            debate_topic_input = input(f"\nEnter your debate topic: ")
            
        # 프롬프트 템플릿 로드 (매번 새로 로드하여 변경 가능성 방지)
        current_config = json.load(open(config_full_path, "r", encoding='utf-8'))
        
        # 사용자 입력 질문을 config에 'question'과 'debate_topic'으로 설정
        current_config['question'] = debate_topic_input # Debate 클래스의 init_prompt가 'question'을 사용하도록 변경되었으므로
        current_config['debate_topic'] = debate_topic_input # 호환성을 위해 유지하거나 제거 가능

        # Debate 인스턴스 생성 및 실행
        debate = Debate(num_players=3, 
                        openai_api_key=args.api_key, 
                        local_llm_url=args.local_llm_url, 
                        model_name=args.model_name, 
                        config=current_config, # 현재 질문이 포함된 config 전달
                        temperature=args.temperature, 
                        sleep_time=0)
        
        # run() 메서드가 이제 최종 config를 반환하므로 받아서 출력
        final_debate_config = debate.run()

        # 인터랙티브 모드에서는 print_answer()를 여기서 직접 호출하여 최종 결과 출력
        # print_answer()는 Debate 클래스 내부에 있으므로, Debate 인스턴스를 통해 호출해야 합니다.
        # 그러나 Debate.run()이 이미 print_answer()를 호출하고 있다면 중복입니다.
        # debate_core.py의 Debate.run()에서 print_answer()를 제거했으므로, 여기서 호출합니다.
        debate.print_answer() # 최종 결과를 출력