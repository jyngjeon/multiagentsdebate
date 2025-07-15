"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import json
import random
# random.seed(0) 
from code.utils.agent import Agent
import argparse

openai_api_key = "Your-OpenAI-Api-Key"

NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature:float, openai_api_key: str, local_llm_url: str, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time, local_llm_url=local_llm_url)
        self.openai_api_key = openai_api_key


class Debate:
    def __init__(self,
            model_name: str='Qwen/Qwen3-14B', 
            temperature: float=0, 
            num_players: int=3, 
            openai_api_key: str=None,
            local_llm_url: str=None,  # added
            config: dict=None,
            max_round: int=3,
            sleep_time: float=0
        ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            openai_api_key (str): As the parameter name suggests
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.local_llm_url = local_llm_url
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time

        self.init_prompt()

        # creat&init agents
        self.creat_agents()
        self.init_agents()


    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, openai_api_key=self.openai_api_key,  local_llm_url=self.local_llm_url, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        
        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.config['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)
        self.config['base_answer'] = self.aff_ans

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)

        # self.mod_ans = eval(self.mod_ans)
        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        
        raw_mod_ans = self.moderator.ask() # LLM의 원본 응답을 받습니다.
        self.moderator.add_memory(raw_mod_ans) # 원본 응답을 메모리에 저장합니다.

        # LLM 응답에서 JSON 부분만 추출하고 파싱합니다.
        try:
            # LLM이 JSON 응답을 ````json ... ```` 또는 `{ ... }` 형태로 줄 수 있습니다.
            # 가장 바깥쪽의 { 부터 } 까지를 찾아서 파싱합니다.
            json_start = raw_mod_ans.find('{')
            json_end = raw_mod_ans.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = raw_mod_ans[json_start : json_end + 1]
                self.mod_ans = json.loads(json_string) # json.loads를 사용하여 더 안전하게 파싱합니다.
            else:
                # JSON을 찾을 수 없는 경우 예외 처리 또는 기본값 설정
                print(f"Warning: Moderator response did not contain a valid JSON object: {raw_mod_ans}")
                self.mod_ans = {"Whether there is a preference": "No", "Supported Side": "", "Reason": "Moderator could not parse response.", "debate_answer": ""}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from moderator: {e}\nRaw response: {raw_mod_ans}")
            self.mod_ans = {"Whether there is a preference": "No", "Supported Side": "", "Reason": "JSON decoding error from moderator.", "debate_answer": ""}
        except Exception as e:
            print(f"Unexpected error processing moderator response: {e}\nRaw response: {raw_mod_ans}")
            self.mod_ans = {"Whether there is a preference": "No", "Supported Side": "", "Reason": "Unexpected error processing moderator response.", "debate_answer": ""}

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)


    def run(self):

        for round in range(self.max_round - 1):

            if self.mod_ans["debate_answer"] != '':
                break
            else:
                print(f"===== Debate Round-{round+2} =====\n")
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(self.aff_ans)

                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.negative.ask()
                self.negative.add_memory(self.neg_ans)

                self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round+2)))
                self.mod_ans = self.moderator.ask()
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True

        # ultimate deadly technique.
        else:
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, openai_api_key=self.openai_api_key, local_llm_url=self.local_llm_url, sleep_time=self.sleep_time)
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            # extract answer candidates
            judge_player.add_event(self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = judge_player.ask()
            judge_player.add_memory(ans)

            # select one from the candidates
            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = judge_player.ask()
            judge_player.add_memory(ans)
            
            ans = eval(ans)
            if ans["debate_answer"] != '':
                self.config['success'] = True
                # save file
            self.config.update(ans)
            self.players.append(judge_player)

        self.print_answer()


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

    args = parser.parse_args()

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    while True:
        debate_topic = ""
        while debate_topic == "":
            debate_topic = input(f"\nEnter your debate topic: ")
            
        config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        config['debate_topic'] = debate_topic

        debate = Debate(num_players=3, 
                        openai_api_key=args.api_key, # Pass the optional API key
                        local_llm_url=args.local_llm_url, # Pass the VLLM URL
                        model_name=args.model_name, # Pass the VLLM model name
                        config=config, 
                        temperature=args.temperature, # Use temperature from args
                        sleep_time=0)
        debate.run()

