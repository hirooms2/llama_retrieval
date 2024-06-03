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
                if self.args.combined:  # self.args.combined:
                    partition = int(len(data['predicted_know']) / 2)
                    n_partition_negative = self.args.n_sampled_negative  # int(self.args.n_docs / 2)

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

                    # predicted_know = top1_negative_candidates + top2_negative_candidates
                    top_negative_candidates = [top1_negative_candidates, top2_negative_candidates]
                    predicted_know = ""
                    for i in range(len(predicted_topic_list)):
                        prefix = f"Here are the candidate passages about Topic {i + 1}. {predicted_topic_list[i]}"
                        candidate_passages = '\n'.join([f"Passage {i * n_partition_negative + idx + 1}. {know}" for idx, know in enumerate(top_negative_candidates[i])])
                        predicted_know += f"{prefix}\n{candidate_passages}\n\n"

                else:  # 사용되는 topic이 무조건 top-1인 경우
                    n_partition_negative = int(self.args.n_docs / 1)
                    predicted_know = data['predicted_know']
                    predicted_know = [i for i in predicted_know if i != '']

                    if self.args.filtering:
                        predicted_know = [i for i in predicted_know if data['predicted_topic'][0].lower().strip() in i.lower().strip()]
                    predicted_know = predicted_know[:self.args.n_docs]
                    predicted_know = '\n'.join([f"{idx + 1}. {know}" for idx, know in enumerate(predicted_know)])

            if 'UD2I' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=data['user_profile'], label=label, mode=mode))
            elif 'DP2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'D2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))
            elif 'DGI2R' in self.args.prompt:
                # num_items = 2 if mode == 'train' else 1
                guide = f"Goal: {data['predicted_goal'][0]} | Topic: {' or '.join(data['predicted_topic'][:1])}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=guide, label=label, mode=mode))
            elif 'DG2P' in self.args.prompt:
                guide = f"Goal: {predicted_goal} | Topic: {' or '.join(predicted_topic_list)}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=guide, label=label, mode=mode))
            elif 'UDP2GP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=data['user_profile'], label=label, mode=mode))
            elif 'DGIP2I' in self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])

                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know, label=label, mode=mode))
            elif 'DGP2P' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_know, label=label, mode=mode))
            elif 'UDGIP2P' == self.args.prompt or 'UDGIP2GIP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2I_new' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2GIP_new' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2GI' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'DGIP2GIP' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know, label=label, mode=mode))
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
