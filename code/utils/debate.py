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
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time,
                                           local_llm_url=local_llm_url, openai_api_key=openai_api_key)


class Debate:
    def __init__(self,
                 model_name: str='Qwen/Qwen3-14B',
                 temperature: float=0,
                 num_players: int=3,
                 openai_api_key: str=None,
                 local_llm_url: str=None,
                 config: dict=None,
                 max_round: int=3,
                 sleep_time: float=0,
                 question_id: int=None
                 ) -> None:
        """
        멀티 에이전트 디베이트를 생성하고 관리합니다.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.local_llm_url = local_llm_url
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.question_id = question_id

        self.aff_ans = ""
        self.neg_ans = ""
        self.mod_ans = {"debate_answer": ""}

        self.init_prompt()
        self.create_agents()
        self.init_agents()

    def _strip_think_tag(self, raw_response: str) -> str:
        """
        LLM의 원본 응답에서 <think>...</think> 태그와 그 내용을 제거합니다.
        """
        if not isinstance(raw_response, str):
            return ""
        # re.DOTALL은 개행 문자도 포함하여 검색합니다.
        return re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

    def init_prompt(self):
        """
        self.config에 로드된 프롬프트 템플릿의 플레이스홀더를 현재 질문 내용으로 대체합니다.
        """
        prompt_keys = [
            "player_meta_prompt", "moderator_meta_prompt", "affirmative_prompt",
            "negative_prompt", "debate_prompt", "moderator_prompt",
            "judge_prompt_last1", "judge_prompt_last2"
        ]
        for key in prompt_keys:
            if key in self.config and isinstance(self.config[key], str):
                self.config[key] = self.config[key].replace("##debate_topic##", self.config.get("question", ""))

    def create_agents(self):
        """
        디베이트에 참여할 Affirmative, Negative, Moderator 플레이어 인스턴스를 생성합니다.
        """
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature,
                         openai_api_key=self.openai_api_key, local_llm_url=self.local_llm_url, sleep_time=self.sleep_time)
            for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        """
        디베이트를 시작하는 초기 단계를 수행합니다.
        """
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])

        if self.question_id is not None:
            print(f"\n===== Debating Question ID: {self.question_id} =====")
        print(f"===== Debate Round-1 =====")

        # 긍정 측의 첫 발언
        self.affirmative.add_event(self.config['affirmative_prompt'])
        raw_aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(raw_aff_ans)
        self.config['affirmative_raw_output_round1'] = raw_aff_ans # 전체 원본 응답 저장
        
        # <<< 수정: Moderator에게 전달하기 전에 생각 제거 >>>
        cleaned_aff_ans = self._strip_think_tag(raw_aff_ans)
        self.aff_ans = cleaned_aff_ans # 정제된 답변을 상태로 저장
        self.config['base_answer'] = cleaned_aff_ans

        # 부정 측의 첫 발언 (정제된 긍정 측 답변 참조)
        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', cleaned_aff_ans))
        raw_neg_ans = self.negative.ask()
        self.negative.add_memory(raw_neg_ans)
        self.config['negative_raw_output_round1'] = raw_neg_ans
        
        # <<< 수정: Moderator에게 전달하기 전에 생각 제거 >>>
        cleaned_neg_ans = self._strip_think_tag(raw_neg_ans)
        self.neg_ans = cleaned_neg_ans

        # Moderator의 1라운드 판단 (정제된 양측 답변 참조)
        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', cleaned_aff_ans).replace('##neg_ans##', cleaned_neg_ans).replace('##round##', 'first'))
        raw_mod_ans = self.moderator.ask()
        self.moderator.add_memory(raw_mod_ans)
        self.mod_ans = self._parse_moderator_response(raw_mod_ans, "first")
        self.config['moderator_raw_output_round1'] = raw_mod_ans

    def _parse_moderator_response(self, raw_response: str, round_id: str) -> dict:
        """
        Moderator/Judge의 응답에서 JSON 부분을 추출하여 파싱합니다.
        """
        # 먼저 사회자 자신의 <think> 태그 제거
        cleaned_response = self._strip_think_tag(raw_response)
        
        # 마크다운 코드 블록 및 JSON 앞뒤 텍스트 제거
        json_str = self._extract_json_string(cleaned_response)

        if not json_str:
            print(f"Warning: No valid JSON found for parsing in round {round_id}. Raw response: '{raw_response}'")
            return {"Whether there is a preference": "No", "Supported Side": "", "Reason": f"JSON parse failed in round {round_id}.", "debate_answer": ""}

        try:
            # 백슬래시 문제 회피를 위한 전처리
            corrected_json_str = json_str.replace('\\', '\\\\')
            return json.loads(corrected_json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from moderator in round {round_id}: {e}\nCleaned response string: '{json_str}'")
            return {"Whether there is a preference": "No", "Supported Side": "", "Reason": f"JSON decoding error in round {round_id}.", "debate_answer": ""}

    def _extract_json_string(self, text: str) -> str | None:
        """
        텍스트에서 JSON 객체 문자열을 추출하는 헬퍼 함수.
        """
        match = re.search(r'```json\s*(\{.*\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def round_dct(self, num: int):
        dct = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}
        return dct.get(num, f'{num}th')
            
    def print_answer(self):
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

    def run(self):
        """
        디베이트의 메인 루프를 실행합니다.
        """
        for round_num in range(self.max_round - 1):
            if self.mod_ans.get("debate_answer", ''):
                break
            else:
                print(f"===== Debate Round-{round_num+2} =====")
                # 긍정 측 발언
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                raw_aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(raw_aff_ans)
                self.aff_ans = self._strip_think_tag(raw_aff_ans) # <<< 수정

                # 부정 측 발언
                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                raw_neg_ans = self.negative.ask()
                self.negative.add_memory(raw_neg_ans)
                self.neg_ans = self._strip_think_tag(raw_neg_ans) # <<< 수정

                # Moderator 판단 (정제된 답변 사용)
                self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round_num+2)))
                raw_mod_ans = self.moderator.ask()
                self.moderator.add_memory(raw_mod_ans)
                self.mod_ans = self._parse_moderator_response(raw_mod_ans, self.round_dct(round_num+2))
                self.config[f'moderator_raw_output_round{round_num+2}'] = raw_mod_ans

        if self.mod_ans.get("debate_answer", ''):
            self.config.update(self.mod_ans)
            self.config['success'] = True
        else:
            print(f"\n===== Moderator did not conclude. Calling Judge for final decision. =====")
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature,
                                         openai_api_key=self.openai_api_key, local_llm_url=self.local_llm_url, sleep_time=self.sleep_time)
            self.players.append(judge_player)
            
            # <<< 수정: Judge에게 전달하기 전에 생각 제거 >>>
            aff_ans_latest_raw = self.affirmative.memory_lst[-1]['content'] if self.affirmative.memory_lst else ""
            neg_ans_latest_raw = self.negative.memory_lst[-1]['content'] if self.negative.memory_lst else ""
            aff_ans_latest_cleaned = self._strip_think_tag(aff_ans_latest_raw)
            neg_ans_latest_cleaned = self._strip_think_tag(neg_ans_latest_raw)

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])
            
            # Judge Stage 1
            judge_player.add_event(self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans_latest_cleaned).replace('##neg_ans##', neg_ans_latest_cleaned))
            raw_ans_judge1 = judge_player.ask()
            judge_player.add_memory(raw_ans_judge1)
            self.config['judge_raw_output_stage1'] = raw_ans_judge1

            # Judge Stage 2
            judge_player.add_event(self.config['judge_prompt_last2'])
            raw_ans_judge2 = judge_player.ask()
            judge_player.add_memory(raw_ans_judge2)
            ans_stage2 = self._parse_moderator_response(raw_ans_judge2, "judge_stage2")
            self.config['judge_raw_output_stage2'] = raw_ans_judge2
            
            ans = ans_stage2
            if ans and ans.get("debate_answer", ''):
                self.config['success'] = True
            else:
                self.config['success'] = False
                ans = {"debate_answer": "", "Reason": "Judge failed to provide final answer.", "Supported Side": ""}
            self.config.update(ans)

        self.config['players'] = {}
        for player in self.players:
            self.config['players'][player.name] = player.memory_lst.copy()

        self.print_answer()
        return self.config