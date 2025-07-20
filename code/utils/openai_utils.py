import tiktoken
from transformers import AutoTokenizer

model2max_context = {
    "gpt-4": 7900,
    "gpt-4-0314": 7900,
    "gpt-3.5-turbo-0301": 3900,
    "gpt-3.5-turbo": 3900,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "Qwen/Qwen3-14B": 16384, # 16384로 바꿔도 되는데 위에 gpt도 길게 안씀
    "Qwen/Qwen2.5-14B-Instruct": 16384
}

# added
_tokenizers_cache = {}

class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def num_tokens_from_string(string: str, model_name: str) -> int:
    # """Returns the number of tokens in a text string."""
    # encoding = tiktoken.encoding_for_model(model_name)
    # num_tokens = len(encoding.encode(string))
    # return num_tokens
    """Returns the number of tokens in a text string for a given model."""

    # OpenAI 모델인 경우 기존 tiktoken 사용
    if model_name.startswith("gpt-"): # 'gpt-3.5-turbo', 'gpt-4' 등
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 특정 모델에 대한 encoding_for_model이 실패하면 일반적인 인코딩을 시도
            encoding = tiktoken.get_encoding("cl100k_base") 
        return len(encoding.encode(string))

    # Qwen 모델인 경우 Hugging Face transformers 토크나이저 사용
    elif model_name.startswith("Qwen/"): # 'Qwen/Qwen3-14B'와 같은 경우
        if model_name not in _tokenizers_cache:
            try:
                # Qwen 모델의 AutoTokenizer 로드
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                _tokenizers_cache[model_name] = tokenizer
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {model_name}. Falling back to approximation. Error: {e}")
                # 토크나이저 로드 실패 시 근사치 반환 (매우 부정확할 수 있음)
                # 실제로는 이 경우 오류를 발생시키거나 사용자에게 알리는 것이 좋습니다.
                return len(string.split()) # 간단히 공백 기준으로 단어 수 세기 (정확하지 않음)

        tokenizer = _tokenizers_cache[model_name]
        return len(tokenizer.encode(string, add_special_tokens=False)) # special token 제외

    else:
        # 그 외 알 수 없는 모델의 경우 (임시 방편)
        print(f"Warning: Unknown model_name '{model_name}'. Falling back to approximate token counting.")
        try:
            # 가능한 경우 tiktoken의 기본 인코딩 시도
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(string))
        except Exception:
            return len(string.split()) # 최후의 수단으로 공백 기준 단어 수
