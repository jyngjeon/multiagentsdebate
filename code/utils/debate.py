# code/utils/debate.py

import os
import json
import random
import re
# random.seed(0) # 필요한 경우 주석 해제
from code.utils.agent import Agent

NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature:float, openai_api_key: str = None, local_llm_url: str = None, sleep_time: float = 0) -> None:
        """
        디베이트에 참여하는 플레이어(에이전트)를 생성합니다.
        Agent 클래스를 상속받아 LLM과의 통신 기능을 사용합니다.

        Args:
            model_name (str): 사용할 LLM의 이름.
            name (str): 플레이어의 이름.
            temperature (float): LLM 응답 다양성.
            openai_api_key (str, optional): OpenAI API 키.
            local_llm_url (str, optional): 로컬 LLM 서버 URL.
            sleep_time (float): API 호출 간 대기 시간.
        """
        # Agent.__init__에 local_llm_url과 openai_api_key 모두 전달
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time, 
                                           local_llm_url=local_llm_url, openai_api_key=openai_api_key)
        # self.openai_api_key = openai_api_key # Agent 클래스에서 이미 저장하므로 여기서 중복 저장하지 않습니다.


class Debate:
    def __init__(self,
            model_name: str='Qwen/Qwen3-14B', # VLLM Qwen 모델을 기본값으로 설정
            temperature: float=0, 
            num_players: int=3, 
            openai_api_key: str=None,
            local_llm_url: str=None,  # 로컬 LLM URL 인자 추가
            config: dict=None,        # 질문 데이터 및 프롬프트 템플릿을 포함하는 config 딕셔너리
            max_round: int=3,
            sleep_time: float=0,
            question_id: int=None     # 현재 처리 중인 문제의 ID 추가 (로그 및 저장용)
        ) -> None:
        """
        멀티 에이전트 디베이트를 생성하고 관리합니다.

        Args:
            model_name (str): 사용할 LLM의 이름.
            temperature (float): LLM 응답 다양성.
            num_players (int): 참여할 플레이어 수 (Affirmative, Negative, Moderator).
            openai_api_key (str, optional): OpenAI API 키.
            local_llm_url (str, optional): 로컬 LLM 서버 URL.
            config (dict): 현재 질문 데이터와 프롬프트 템플릿을 포함하는 딕셔너리.
            max_round (int): 디베이트의 최대 라운드 수.
            sleep_time (float): API 호출 간 대기 시간.
            question_id (int, optional): 현재 디베이트 중인 문제의 고유 번호.
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.local_llm_url = local_llm_url
        self.config = config # 질문 데이터를 config로 사용
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.question_id = question_id 

        self.aff_ans = "" 
        self.neg_ans = ""
        self.mod_ans = {"debate_answer": ""}

        self.init_prompt() 
        self.create_agents()
        self.init_agents()


    def init_prompt(self):
        """
        self.config에 로드된 프롬프트 템플릿의 플레이스홀더를 현재 질문 내용으로 대체합니다.
        """
        # 프롬프트 템플릿 키 목록
        prompt_keys = [
            "player_meta_prompt", "moderator_meta_prompt", "affirmative_prompt",
            "negative_prompt", "debate_prompt", "moderator_prompt",
            "judge_prompt_last1", "judge_prompt_last2"
        ]
        
        # 각 프롬프트 템플릿에서 '##debate_topic##' 플레이스홀더를 실제 질문으로 대체
        for key in prompt_keys:
            if key in self.config and isinstance(self.config[key], str):
                self.config[key] = self.config[key].replace("##debate_topic##", self.config.get("question", ""))
                # 나머지 플레이스홀더 (##aff_ans##, ##neg_ans## 등)는 런타임에 add_event에서 대체됩니다.


    def create_agents(self):
        """
        디베이트에 참여할 Affirmative, Negative, Moderator 플레이어 인스턴스를 생성합니다.
        """
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, 
                         openai_api_key=self.openai_api_key, local_llm_url=self.local_llm_url, sleep_time=self.sleep_time) 
            for name in NAME_LIST
        ]
        # 각 플레이어 인스턴스를 개별 변수로 할당하여 쉽게 접근할 수 있게 합니다.
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        """
        디베이트를 시작하는 초기 단계를 수행합니다.
        메타 프롬프트 설정, 1라운드 발언 및 Moderator의 초기 판단을 포함합니다.
        """
        # 시스템 메타 프롬프트 설정
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        
        # 현재 문제 번호를 출력 (배치 실행 시 유용)
        if self.question_id is not None:
            print(f"\n===== Debating Question ID: {self.question_id} =====")
        print(f"===== Debate Round-1 =====")
        
        # 긍정 측의 첫 발언
        self.affirmative.add_event(self.config['affirmative_prompt'])
        raw_aff_ans = self.affirmative.ask() # 모델의 원본 응답 받기
        self.affirmative.add_memory(raw_aff_ans) # 원본 응답을 메모리에 저장
        self.config['base_answer'] = raw_aff_ans # Affirmative의 첫 답변을 'base_answer'로 저장
        self.config['affirmative_raw_output_round1'] = raw_aff_ans # raw output 별도 저장

        # 부정 측의 첫 발언 (긍정 측 답변 참조)
        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', raw_aff_ans))
        raw_neg_ans = self.negative.ask() # 모델의 원본 응답 받기
        self.negative.add_memory(raw_neg_ans) # 원본 응답을 메모리에 저장
        self.config['negative_raw_output_round1'] = raw_neg_ans # raw output 별도 저장

        # Moderator의 1라운드 판단 (양측 답변 참조)
        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', raw_aff_ans).replace('##neg_ans##', raw_neg_ans).replace('##round##', 'first'))
        raw_mod_ans = self.moderator.ask() # 모델의 원본 응답 받기
        self.moderator.add_memory(raw_mod_ans) # 원본 응답을 메모리에 저장
        # Moderator의 응답에서 JSON 부분을 파싱
        self.mod_ans = self._parse_moderator_response(raw_mod_ans, "first")
        self.config['moderator_raw_output_round1'] = raw_mod_ans # raw output 별도 저장


    def _parse_moderator_response(self, raw_response: str, round_id: str) -> dict:
        """
        LLM의 원본 응답(Moderator, Judge)에서 JSON 부분을 추출하여 파싱합니다.
        <think> 태그 및 JSON 앞뒤의 불필요한 텍스트를 제거합니다.
        """
        # 1. <think> 태그와 그 내용을 제거합니다. (re.DOTALL은 개행 문자도 포함)
        cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        # 2. JSON 객체의 시작 { 을 기준으로 그 이전 텍스트를 제거합니다.
        json_start_char_index = cleaned_response.find('{')
        if json_start_char_index != -1:
            cleaned_response = cleaned_response[json_start_char_index:]
        else:
            # JSON 시작 괄호를 찾지 못하면, 유효한 JSON이 아니므로 경고 후 기본값 반환
            print(f"Warning: No opening curly brace '{{' found for JSON parsing in round {round_id}. Raw response (cleaned): '{cleaned_response}'")
            return {"Whether there is a preference": "No", "Supported Side": "", "Reason": f"JSON parse failed (no {{ found) in round {round_id}.", "debate_answer": ""}

        try:
            # 3. 추출된 JSON 문자열을 파이썬 딕셔너리로 로드합니다.
            return json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from moderator in round {round_id}: {e}\nRaw response (cleaned): '{cleaned_response}'")
            return {"Whether there is a preference": "No", "Supported Side": "", "Reason": f"JSON decoding error in round {round_id}.", "debate_answer": ""}
        except Exception as e:
            print(f"Unexpected error processing moderator response in round {round_id}: {e}\nRaw response (cleaned): '{cleaned_response}'")
            return {"Whether there is a preference": "No", "Supported Side": "", "Reason": f"Unexpected error in round {round_id}.", "debate_answer": ""}

    def round_dct(self, num: int):
        """라운드 번호를 텍스트 (first, second 등)로 변환합니다."""
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]
            
    def print_answer(self):
        """
        현재 디베이트의 최종 요약 답변을 터미널에 출력합니다.
        (주로 interactive.py에서 사용되거나, run_debate.py에서 로그 파일에 기록)
        """
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config.get("question", "N/A")) 
        print("\n----- Base Answer (Affirmative's first answer) -----")
        print(self.config.get("base_answer", "N/A")) 
        print("\n----- Debate Answer (Moderator/Judge's final answer) -----")
        print(self.config.get("debate_answer", "N/A"))
        print("\n----- Debate Reason -----")
        print(self.config.get("Reason", "N/A"))
        print("\n----- Correct Answers (from JSON) -----")
        print(self.config.get("answer", "N/A"))


    def broadcast(self, msg: str):
        """모든 플레이어에게 공지 메시지를 브로드캐스트합니다."""
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """특정 플레이어가 다른 모든 플레이어에게 메시지를 브로드캐스트합니다."""
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        """플레이어에게 질문하고, 답변을 받아 메모리에 추가한 후 다른 플레이어에게 브로드캐스트합니다."""
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)


    def run(self):
        """
        디베이트의 메인 루프를 실행합니다.
        Moderator가 최종 답변을 내거나 최대 라운드에 도달할 때까지 진행합니다.
        """
        for round_num in range(self.max_round - 1): # Round 1은 init_agents에서 이미 처리되었습니다.
            if self.mod_ans.get("debate_answer", '') != '': # Moderator가 최종 답변을 냈다면 토론 종료
                break
            else:
                print(f"===== Debate Round-{round_num+2} =====")
                # 긍정 측 발언
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                raw_aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(raw_aff_ans)
                self.aff_ans = str(raw_aff_ans) if raw_aff_ans is not None else ""

                # 부정 측 발언
                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', raw_aff_ans))
                raw_neg_ans = self.negative.ask()
                self.negative.add_memory(raw_neg_ans)
                self.neg_ans = str(raw_neg_ans) if raw_neg_ans is not None else ""

                # Moderator 판단
                self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', raw_aff_ans).replace('##neg_ans##', raw_neg_ans).replace('##round##', self.round_dct(round_num+2)))
                raw_mod_ans = self.moderator.ask()
                self.moderator.add_memory(raw_mod_ans)
                self.mod_ans = self._parse_moderator_response(raw_mod_ans, self.round_dct(round_num+2))
                self.config[f'moderator_raw_output_round{round_num+2}'] = raw_mod_ans # raw output 저장

        # 디베이트 종료 후 최종 처리
        if self.mod_ans.get("debate_answer", '') != '':
            # Moderator가 최종 답변을 성공적으로 냈을 경우
            self.config.update(self.mod_ans) # Moderator의 최종 판단 (Reason, debate_answer 등)으로 config 업데이트
            self.config['success'] = True
        else: 
            # Moderator가 max_round까지 최종 답변을 내지 못했을 경우, Judge 호출
            print(f"\n===== Moderator did not conclude. Calling Judge for final decision. =====")
            # Judge 플레이어 생성 (Judge도 DebatePlayer를 사용)
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, 
                                        openai_api_key=self.openai_api_key, local_llm_url=self.local_llm_url, sleep_time=self.sleep_time)
            # Judge 플레이어를 self.players 리스트에 추가하여 메모리도 함께 저장될 수 있도록 합니다.
            self.players.append(judge_player) 
            
            # 마지막 라운드의 양측 발언을 Judge에게 전달
            aff_ans_latest = self.affirmative.memory_lst[-1]['content'] if self.affirmative.memory_lst else ""
            neg_ans_latest = self.negative.memory_lst[-1]['content'] if self.negative.memory_lst else ""

            # Judge에게도 Moderator와 유사한 메타 프롬프트 설정
            judge_player.set_meta_prompt(self.config['moderator_meta_prompt']) 

            # Judge Stage 1: 답변 후보 추출
            judge_player.add_event(self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans_latest).replace('##neg_ans##', neg_ans_latest))
            raw_ans_judge1 = judge_player.ask() # 모델의 원본 응답
            judge_player.add_memory(raw_ans_judge1)
            ans_stage1 = self._parse_moderator_response(raw_ans_judge1, "judge_stage1") # JSON 파싱
            self.config['judge_raw_output_stage1'] = raw_ans_judge1 # raw output 저장

            # Judge Stage 2: 최종 답변 선택
            judge_player.add_event(self.config['judge_prompt_last2'])
            raw_ans_judge2 = judge_player.ask() # 모델의 원본 응답
            judge_player.add_memory(raw_ans_judge2)
            ans_stage2 = self._parse_moderator_response(raw_ans_judge2, "judge_stage2") # JSON 파싱
            self.config['judge_raw_output_stage2'] = raw_ans_judge2 # raw output 저장
            
            # 최종 답변은 Judge Stage 2의 결과로 판단
            ans = ans_stage2 
            if ans and ans.get("debate_answer", '') != '':
                self.config['success'] = True
            else:
                self.config['success'] = False
                # Stage 2가 실패했다면 Stage 1의 결과를 fallback으로 시도
                if ans_stage1 and ans_stage1.get("debate_answer", '') != '':
                    ans = ans_stage1
                    self.config['success'] = True
                else: # Judge도 최종 답변을 내지 못했을 경우
                    ans = {"debate_answer": "", "Reason": "Judge failed to provide final answer.", "Supported Side": ""}

            self.config.update(ans) # Judge의 최종 판단으로 config 업데이트


        # --- 모든 플레이어의 최종 메모리(대화 기록)를 config에 저장합니다. ---
        self.config['players'] = {} 
        for player in self.players:
            self.config['players'][player.name] = player.memory_lst.copy() # 메모리 리스트 복사하여 저장
        # --- 저장 로직 끝 ---

        # 디베이트 최종 요약 답변을 터미널/로그 파일에 출력
        self.print_answer() # <-- 여기에 print_answer 호출을 다시 추가합니다.

        return self.config # 최종 config를 반환하여 run_debate.py에서 저장할 수 있게 합니다.