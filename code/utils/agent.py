# code/utils/agent.py

import openai
import backoff
import time
import random
import re # normalize_answer 함수에서 사용
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout # requests 관련 예외

# openai_utils.py에서 필요한 함수 및 딕셔너리 임포트
from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context

# VLLM 서버와 통신할 때 사용하는 모델 이름 목록.
# VLLM에 로드된 모델 이름 (예: Qwen/Qwen3-14B)을 포함해야 합니다.
# OpenAI 모델도 필요하다면 여기에 추가됩니다.
support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', 'Qwen/Qwen3-14B', 'Qwen/Qwen2.5-14B-Instruct']

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0, local_llm_url: str = None, openai_api_key: str = None) -> None:
        """
        AI 에이전트를 생성합니다. 이 에이전트는 LLM과 통신하고 대화 기록을 관리합니다.

        Args:
            model_name (str): 사용할 LLM의 이름 (예: 'Qwen/Qwen3-14B', 'gpt-4').
            name (str): 에이전트의 고유 이름.
            temperature (float): LLM의 응답 다양성을 제어하는 값 (높을수록 다양).
            sleep_time (float): API 요청 간 대기 시간 (초).
            local_llm_url (str, optional): 로컬 VLLM 서버의 URL (예: "http://localhost:8000").
                                            이 값이 주어지면 로컬 LLM을 사용합니다.
            openai_api_key (str, optional): OpenAI API 키. local_llm_url이 없을 때 사용됩니다.
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []  # 대화 기록 저장
        self.sleep_time = sleep_time
        self.local_llm_url = local_llm_url
        self.openai_api_key = openai_api_key # OpenAI API 키를 인스턴스 변수로 저장

        # OpenAI 클라이언트 인스턴스 초기화.
        # 로컬 LLM URL이 제공되면 해당 URL을 base_url로 사용합니다.
        # OpenAI API Key는 로컬 LLM(VLLM)에서는 보통 필요 없지만, openai.OpenAI 클라이언트의 인자로 요구될 수 있어 더미 키를 사용합니다.
        if self.local_llm_url:
            base_url_for_client = self.local_llm_url.rstrip('/') 
            self.openai_client = openai.OpenAI(base_url=f"{base_url_for_client}/v1", api_key="sk-no-key-required") 
        else:
            self.openai_client = None

    @backoff.on_exception(backoff.expo, 
                          (RequestException, HTTPError, ConnectionError, Timeout, openai.APIError), # backoff 재시도 예외 목록
                          max_tries=20) # 최대 20번 재시도
    def query(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """
        LLM에 질의를 보내 응답을 받습니다.

        Args:
            messages (list[dict]): OpenAI ChatCompletion API 형식의 대화 기록 (role, content).
            max_tokens (int): 생성할 답변의 최대 토큰 수.
            temperature (float): 응답 다양성 제어 온도.

        Returns:
            str: LLM으로부터 받은 응답 텍스트.

        Raises:
            OutOfQuotaException: OpenAI API 할당량 초과 시.
            AccessTerminatedException: OpenAI API 접근이 종료되었을 시.
            Exception: 그 외 API 통신 또는 응답 처리 중 발생한 오류.
        """
        time.sleep(self.sleep_time) # Rate Limit 회피를 위한 대기

        # local_llm_url이 설정되어 있으면 로컬 VLLM 서버를 통해 통신합니다.
        if self.local_llm_url:
            client_to_use = self.openai_client # __init__에서 초기화된 클라이언트 사용
            try:
                response = client_to_use.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                gen = response.choices[0].message.content # OpenAI v1.0+ 응답 형식
                return gen
            # VLLM과의 통신 실패(네트워크, HTTP 오류) 또는 OpenAI 클라이언트 자체 오류 처리
            except openai.APIError as e: # openai 클라이언트에서 발생하는 API 오류
                print(f"Error calling local LLM (OpenAI APIError path): {e}")
                raise e 
            except RequestException as e: # requests 라이브러리 (또는 httpx) 관련 네트워크/HTTP 오류
                print(f"Error calling local LLM (RequestException path): {e}")
                raise e
            except Exception as e: # 그 외 예상치 못한 모든 예외
                print(f"An unexpected error occurred during local LLM call: {e}")
                raise e
        
        # local_llm_url이 설정되지 않았으면 원격 OpenAI API를 사용합니다.
        else:
            if not self.openai_api_key:
                raise ValueError("OpenAI API Key is not provided for remote OpenAI API calls.")
            
            # 원격 OpenAI API 호출을 위한 클라이언트 생성
            # 매 호출마다 클라이언트를 생성하는 방식 또는 Agent.__init__에서 한 번 생성 후 재사용하는 방식 중 선택 가능.
            # 여기서는 API 키 유동성 때문에 매번 생성하는 방식으로 유지.
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                gen = response.choices[0].message.content
                return gen

            # OpenAI API 오류 처리: RateLimitError, 그 외 APIError 등
            except openai.APIError as e:
                error_message = str(e) 
                if "quota" in error_message.lower():
                    raise OutOfQuotaException(self.openai_api_key)
                elif "terminated" in error_message.lower() or "banned" in error_message.lower():
                    raise AccessTerminatedException(self.openai_api_key)
                else:
                    print(f"OpenAI API Error (remote LLM client path): {e}")
                    raise e
            except RequestException as e: # OpenAI 클라이언트가 내부적으로 httpx를 사용하므로 RequestException도 발생 가능
                print(f"Network/Request Error (remote LLM client path): {e}")
                raise e
            except Exception as e: # 그 외 예상치 못한 모든 예외
                print(f"An unexpected error occurred during remote OpenAI API call: {e}")
                raise e

    def set_meta_prompt(self, meta_prompt: str):
        """
        에이전트의 시스템 프롬프트(페르소나/역할 지시)를 설정합니다.
        이는 대화 기록의 첫 번째 메시지가 됩니다.
        """
        # 기존 시스템 프롬프트가 있다면 덮어쓰기 (또는 업데이트)
        if self.memory_lst and self.memory_lst[0]['role'] == 'system':
            self.memory_lst[0]['content'] = meta_prompt
        else:
            self.memory_lst.insert(0, {"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        """
        사용자(또는 다른 에이전트)로부터의 새로운 대화 이벤트를 메모리에 추가합니다.
        """
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        """
        에이전트 자신의 발언을 메모리에 추가합니다.
        모델의 긴 사고 과정 (<think> 태그 내부)은 저장하지 않고 제거하여 토큰 길이를 관리합니다.
        """
        # <think>...</think> 태그 안의 내용을 찾습니다.
        thought_match = re.search(r'<think>(.*?)</think>', memory, re.DOTALL)
        
        # 메모리에 저장할 내용은 <think> 태그 부분을 제거한 나머지 텍스트입니다.
        if thought_match:
            memory_to_store = re.sub(r'<think>.*?</think>', '', memory, flags=re.DOTALL).strip()
        else:
            memory_to_store = memory.strip() # <think> 태그가 없으면 전체 저장

        # 빈 문자열은 저장하지 않습니다 (예: 생각만 있고 답변이 없는 경우)
        if memory_to_store:
            self.memory_lst.append({"role": "assistant", "content": memory_to_store})
        
        # 출력은 원본 그대로 (사용자/개발자가 사고 과정을 볼 수 있도록)
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float=None):
        """
        현재 대화 기록을 바탕으로 LLM에 질의하여 답변을 받습니다.
        컨텍스트 길이 관리를 위해 필요한 경우 오래된 메시지를 잘라냅니다.
        """
        # VLLM 서버의 실제 max_model_len 값을 설정합니다.
        # 이 값은 VLLM 서버 시작 시 `--max-model-len` 인자로 설정한 값과 일치해야 합니다.
        # 예를 들어 16384 또는 8192 등으로 설정.
        vllm_actual_model_len = 16384 # <-- 이 값을 VLLM 서버 설정에 맞게 변경하세요.
        model_total_max_context = model2max_context.get(self.model_name, vllm_actual_model_len) 

        # 모델이 최소한으로 생성할 답변 토큰 수 (이 공간은 항상 확보하려고 시도합니다).
        min_completion_tokens = 50 

        # 클라이언트 토크나이저와 서버 토크나이저 간의 불일치, 모델 내부 오버헤드 등을 위한 안전 마진.
        # 이 값을 충분히 크게 설정하여 컨텍스트 초과 오류를 방지하는 것이 중요합니다.
        safety_margin_tokens = 100 

        # 현재 memory_lst에 있는 모든 메시지들의 총 토큰 수를 계산합니다.
        num_current_messages_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])

        # 허용 가능한 최대 메시지 토큰 수 계산: (전체 컨텍스트 - 최소 답변 토큰 - 안전 마진)
        allowed_max_message_tokens = model_total_max_context - min_completion_tokens - safety_margin_tokens

        messages_to_send = self.memory_lst # LLM에 실제로 전송할 메시지 리스트 (기본값: 전체 메모리)
        current_tokens_after_truncation = num_current_messages_token # 전송할 메시지의 토큰 수 (기본값: 전체 메모리 토큰 수)

        # 메시지 길이가 허용치를 초과하는 경우, 오래된 메시지부터 잘라냅니다.
        if num_current_messages_token > allowed_max_message_tokens:
            print(f"Warning: Current message tokens ({num_current_messages_token}) exceed allowed max ({allowed_max_message_tokens}). Truncating memory for {self.name}.")
            
            truncated_memory_lst = []
            current_tokens_after_truncation = 0 # 트렁케이션 시작 시 토큰 카운트 초기화

            # 1. 시스템 프롬프트 (첫 번째 메시지)는 대화의 핵심이므로 최대한 유지합니다.
            if self.memory_lst and self.memory_lst[0]['role'] == 'system':
                truncated_memory_lst.append(self.memory_lst[0])
                current_tokens_after_truncation += num_tokens_from_string(self.memory_lst[0]['content'], self.model_name)

            # 2. 시스템 프롬프트 다음부터의 메시지들을 최신 순서대로 추가하여 컨텍스트를 채웁니다.
            #    (이를 통해 오래된 대화 기록부터 삭제하는 효과를 얻습니다.)
            messages_to_consider = self.memory_lst[len(truncated_memory_lst):]
            
            # 임시 리스트에 역순으로 추가 (효율성을 위해)
            temp_reverse_list = []
            for m in reversed(messages_to_consider):
                m_tokens = num_tokens_from_string(m["content"], self.model_name)
                if current_tokens_after_truncation + m_tokens <= allowed_max_message_tokens:
                    temp_reverse_list.append(m)
                    current_tokens_after_truncation += m_tokens
                else:
                    break # 더 이상 메시지를 추가할 수 없으면 중단
            
            # temp_reverse_list를 다시 뒤집어 정순으로 만든 후, 시스템 프롬프트 뒤에 붙입니다.
            truncated_memory_lst.extend(reversed(temp_reverse_list))
            
            # 최종적으로 전송할 메시지 리스트를 업데이트합니다.
            messages_to_send = truncated_memory_lst
            
            # 잘라낸 후에도 토큰 수가 여전히 허용치를 초과하는 경우 (매우 드물지만, 대비)
            if current_tokens_after_truncation > allowed_max_message_tokens:
                 print(f"Error: Truncation logic failed for {self.name}. Forcing system and last message only.")
                 messages_to_send = []
                 if self.memory_lst and self.memory_lst[0]['role'] == 'system':
                     messages_to_send.append(self.memory_lst[0])
                 if len(self.memory_lst) > (1 if messages_to_send else 0): # 시스템 메시지 외에 다른 메시지가 있다면
                     messages_to_send.append(self.memory_lst[-1]) # 마지막 메시지 추가
                 current_tokens_after_truncation = sum([num_tokens_from_string(m["content"], self.model_name) for m in messages_to_send])
                
        # 생성할 답변의 최대 토큰 수 계산: (모델 총 컨텍스트 - 실제 보낼 메시지 토큰 - 안전 마진)
        max_output_tokens = model_total_max_context - current_tokens_after_truncation - safety_margin_tokens

        # 계산된 max_output_tokens가 최소값보다 작으면 최소값으로 설정
        if max_output_tokens < min_completion_tokens:
            print(f"Warning: Calculated max_output_tokens ({max_output_tokens}) for {self.name} is less than min_completion_tokens ({min_completion_tokens}). Setting to min_completion_tokens.")
            max_output_tokens = min_completion_tokens 
        
        # 최종적으로 생성할 토큰 수가 음수이거나 0이 되는 것을 방지
        if max_output_tokens <= 0:
            print(f"Error: max_output_tokens calculated as {max_output_tokens} for {self.name}. Forcing to 50 for a minimal response.")
            max_output_tokens = 50 

        # 최종 메시지 리스트와 계산된 max_output_tokens, 그리고 온도를 query 메서드에 전달합니다.
        return self.query(messages_to_send, max_output_tokens, temperature=temperature if temperature else self.temperature)