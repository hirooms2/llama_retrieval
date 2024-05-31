import json
import os
import random
from copy import deepcopy
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "args")

    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.args = args
        file_name = os.path.join(args.home, "templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_instructions(self,
                              mode,
                              dataset_input: list,
                              dataset_output: list):
        instructions = []

        for data, label in zip(dataset_input, dataset_output):

            # predicted_goal = data['predicted_goal'][0]
            if self.args.query:
                predicted_goal = data['query']
            else:
                predicted_goal = data['predicted_goal'][0]

            predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
            if self.args.topic_conf != 1:
                predicted_topic_conf_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])
                cum_prob = 0
                candidate_topic_entities = []
                for p_topic, p_conf in zip(predicted_topic_list, predicted_topic_conf_list):
                    if cum_prob < self.args.topic_conf:
                        candidate_topic_entities.append(p_topic)
                        cum_prob += p_conf
                predicted_topic_list = deepcopy(candidate_topic_entities)

            if 'predicted_know' in data and 'P' in self.args.prompt:
                if len(predicted_topic_list) > 1 or self.args.combined_top1:  # self.args.combined:
                    partition = int(len(data['predicted_know']) / 2)
                    n_partition_negative = int(self.args.n_docs / 2)

                    top1_negative_candidates = data['predicted_know'][:partition]
                    top2_negative_candidates = data['predicted_know'][partition:]

                    top1_negative_candidates = [i for i in top1_negative_candidates if i != '']
                    top2_negative_candidates = [i for i in top2_negative_candidates if i != '']

                    # Filtering code
                    if self.args.filtering:
                        top1_negative_candidates = [i for i in top1_negative_candidates if data['predicted_topic'][0].lower().strip() in i.lower().strip()]
                        top2_negative_candidates = [i for i in top2_negative_candidates if data['predicted_topic'][1].lower().strip() in i.lower().strip()]

                    top1_negative_candidates = top1_negative_candidates[:n_partition_negative]
                    top2_negative_candidates = top2_negative_candidates[:n_partition_negative]

                    # Tagging code
                    # top1_negative_candidates = [f"{data['predicted_topic'][0]}|{i}" for i in top1_negative_candidates]
                    # top2_negative_candidates = [f"{data['predicted_topic'][1]}|{i}" for i in top2_negative_candidates]
                    predicted_know = top1_negative_candidates + top2_negative_candidates
                    # top_negative_candidates = [top1_negative_candidates, top2_negative_candidates]
                    # predicted_know = []
                    # for i in range(len(predicted_topic_list)):
                    #     predicted_know += top_negative_candidates[i]
                else:  # 사용되는 topic이 무조건 top-1인 경우
                    predicted_know = data['predicted_know']
                    predicted_know = [i for i in predicted_know if i != '']

                    if self.args.filtering:
                        predicted_know = [i for i in predicted_know if data['predicted_topic'][0].lower().strip() in i.lower().strip()]
                    predicted_know = predicted_know[:self.args.n_docs]

                if len(predicted_know) == 0:  # 가끔 싹 다 필터링되는 상황이 있음
                    predicted_know = data['predicted_know'][:self.args.n_docs]

                relevant_idx = predicted_know.index(label) if label in predicted_know else -1
                label = f"{relevant_idx + 1}. {label}"
                predicted_know = '\n'.join([f"{idx + 1}. {know}" for idx, know in enumerate(predicted_know)])

            if 'UD2I' in self.args.prompt:
                instructions.append(
                    self.generate_prompt(instruction=data['dialog'], input=data['user_profile'], label=label,
                                         mode=mode))
            elif 'DP2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'D2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))
            elif 'DGI2R' in self.args.prompt:
                # num_items = 2 if mode == 'train' else 1
                guide = f"Goal: {data['predicted_goal'][0]} | Topic: {' or '.join(data['predicted_topic'][:1])}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=guide, label=label, mode=mode))
            elif 'DG2P' in self.args.prompt:
                # num_items = 2 if mode == 'train' else 1
                # mode choices = ['test', 'train_test'] -> train_test 옵션일 때 적용 하려면 구조 변경 필요
                # if self.args.mode == 'test' and self.args.item_selection == 'conf':
                #     # adaptive item selection (topic conf) if mode tests
                #     cum_prob = 0
                #     candidate_topic_entities = []
                #     predicted_topic_list = deepcopy(data['predicted_topic'])
                #     predicted_topic_conf_list = deepcopy(data['predicted_topic_confidence'])
                #     for p_topic, p_conf in zip(predicted_topic_list, predicted_topic_conf_list):
                #         if cum_prob < self.args.topic_conf:
                #             candidate_topic_entities.append(p_topic)
                #             cum_prob += p_conf
                #     guide = f"Goal: {data['predicted_goal'][0]} | Topic: {' or '.join(candidate_topic_entities)}"
                # else:
                guide = f"Goal: {predicted_goal} | Topic: {' or '.join(predicted_topic_list)}"

                # instructions.append(self.generate_prompt(instruction=data['dialog'],  label=label, mode=mode))
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=guide, label=label, mode=mode))
            elif 'UDP2GP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=data['user_profile'], label=label, mode=mode))
            elif 'DGIP2GIP' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know, label=label, mode=mode))
            elif 'DGP2P' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_know, label=label, mode=mode))
            elif 'UDGIP2P' == self.args.prompt or 'UDGIP2GIP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2GI' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGI2GI' == self.args.prompt:
                if mode == 'train':
                    label = f"Goal: {data['goal']}\nTopic: {data['topic']}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=data['user_profile'], label=label, mode=mode))
            elif 'DP2GP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'DP2I' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'UDP2I' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=data['user_profile'], label=label, mode=mode))
            elif self.args.prompt == 'pretrain':
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))
            elif 'D2P' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'D2V' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'DI2P' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=data['topic'], label=label, mode=mode))
        return instructions

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            input2: Union[None, str] = None,
            input3: Union[None, str] = None,
            input4: Union[None, str] = None,
            label: Union[None, str] = None,
            mode: str = 'test') -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input and not input2:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        elif input and input2 and not input3:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2
            )
        elif input and input2 and input3 and not input4:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2, input3=input3
            )
        elif input and input2 and input3 and input4:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2, input3=input3, input4=input4
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if mode == 'train':
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
