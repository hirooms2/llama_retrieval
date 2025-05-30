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

            topk_topic = self.args.topk_topic
            predicted_topic_list = deepcopy(data['predicted_topic'][:topk_topic])
            if self.args.topic_conf != 1:
                predicted_topic_conf_list = deepcopy(data['predicted_topic_confidence'][:topk_topic])
                cum_prob = 0
                candidate_topic_entities = []
                for p_topic, p_conf in zip(predicted_topic_list, predicted_topic_conf_list):
                    if cum_prob < self.args.topic_conf:
                        candidate_topic_entities.append(p_topic)
                        cum_prob += p_conf
                predicted_topic_list = deepcopy(candidate_topic_entities)

            if self.args.selected_topic and 'selected_topic' in dataset_input[0]:
                predicted_topic = data['selected_topic']
            else:
                if self.args.inspired and data['predicted_topic'] == []:
                    predicted_topic = data['predicted_topic']
                else:
                    predicted_topic = data['predicted_topic'][0]

            if 'predicted_know' in data and 'P' in self.args.prompt:
                top_negative_candidates = deepcopy(data['predicted_know'])
                random_candidates = deepcopy(data['predicted_know'])
                if self.args.combined:  # self.args.combined:
                    for idx, top_passages in enumerate(top_negative_candidates):
                        top_negative_candidates[idx] = [i for i in top_passages if i != '']
                    for idx, random_passages in enumerate(random_candidates):
                        tmp = [i for i in random_passages if i != '']
                        random.shuffle(tmp)
                        random_candidates[idx] = tmp

                    # Filtering code
                    if self.args.filtering:
                        for idx, top_passages in enumerate(top_negative_candidates):
                            top_negative_candidates[idx] = [i for i in top_passages if data['predicted_topic'][idx].lower().strip() in i.lower().strip()]

                    for idx, top_passages in enumerate(top_negative_candidates):
                        top_negative_candidates[idx] = top_negative_candidates[idx][:self.args.n_sampled_negative]

                    for idx, random_passages in enumerate(random_candidates):
                        random_candidates[idx] = random_candidates[idx][:self.args.n_sampled_negative]

                    predicted_know = ""
                    for i in range(len(predicted_topic_list)):
                        if self.args.inspired or self.args.redial:
                            prefix = f"Here are the candidate passages about Item {i + 1}. {predicted_topic_list[i]}"
                        else:
                            prefix = f"Here are the candidate passages about Topic {i + 1}. {predicted_topic_list[i]}"
                        candidate_passages = '\n'.join([f"Passage {i * self.args.n_sampled_negative + idx + 1}. {know}" for idx, know in enumerate(top_negative_candidates[i])])
                        random_passages = '\n'.join([f"Passage {i * self.args.n_sampled_negative + idx + 1}. {know}" for idx, know in enumerate(random_candidates[i])])
                        if self.args.all_passages:
                            predicted_know += f"{candidate_passages}\n\n"
                        else:
                            if self.args.random_passages:
                                predicted_know += f"{prefix}\n{random_passages}\n\n"
                            else:
                                predicted_know += f"{prefix}\n{candidate_passages}\n\n"

                else:  # 사용되는 topic이 무조건 top-1인 경우
                    predicted_know = [i for i in top_negative_candidates[0] if i != '']

                    if self.args.filtering:

                        predicted_know_filtered = [i for i in predicted_know if predicted_topic.lower().strip() in i.replace('\xa0', ' ').strip().lower().strip()]
                        predicted_know_unfiltered = [i for i in predicted_know if predicted_topic.lower().strip() not in i.replace('\xa0', ' ').strip().lower().strip()]

                        if len(predicted_know_filtered) < self.args.n_sampled_negative:
                            predicted_know_filtered = predicted_know_filtered + predicted_know_unfiltered[:self.args.n_sampled_negative - len(predicted_know_filtered)]
                        predicted_know = predicted_know_filtered
                    if self.args.target:
                        predicted_know = data['target_knowledge']
                        predicted_know = f"Passage 1. {predicted_know}\n"
                    else:
                        predicted_know = predicted_know[:self.args.n_sampled_negative]
                        predicted_know = '\n'.join([f"Passage {idx + 1}. {know}" for idx, know in enumerate(predicted_know)])

            if mode == 'train':
                random.shuffle(predicted_topic_list)

            if 'UD2I' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=data['user_profile'], label=label, mode=mode))
            if 'D2I' == self.args.prompt or 'D2I_cot' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))
            if 'DI2I' == self.args.prompt or 'DI2I_cot' == self.args.prompt:
                if self.args.redial or self.args.inspired:
                    candidate_topics = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                else:
                    candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=candidate_topics, label=label, mode=mode))
            elif 'UDGI2I' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=data['user_profile'], label=label, mode=mode))
            elif 'UDGI2I_cot' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=data['user_profile'], label=label, mode=mode))
            elif 'UDG2I' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=data['user_profile'], label=data['topic'], mode=mode))
            elif 'UDG2I_cot' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=data['user_profile'], label=data['topic'], mode=mode))
            elif 'DP2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif 'DGP2R' == self.args.prompt:
                if predicted_know == '':
                    instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=' ', label=label, mode=mode))
                else:
                    instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_know, label=label, mode=mode))
            elif 'DGIP2R' == self.args.prompt:
                if predicted_know == '':
                    instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_topic, input3=' ', label=label, mode=mode))
                else:
                    instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_topic, input3=predicted_know, label=label, mode=mode))
            elif 'DIP2R' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_topic, input2=predicted_know, label=label, mode=mode))
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
            elif 'DIP2I' == self.args.prompt or 'DIP2I_cot' == self.args.prompt:
                if self.args.redial or self.args.inspired:
                    candidate_topics = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                else:
                    candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=candidate_topics, input2=predicted_know, label=label, mode=mode))
            elif 'DIP2I_redial' == self.args.prompt or 'DIP2I_redial_cot' == self.args.prompt:
                if self.args.redial or self.args.inspired:
                    candidate_topics = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                else:
                    candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=candidate_topics, input2=predicted_know, label=label, mode=mode))
            elif 'DGP2P' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_know, label=label, mode=mode))
            elif 'UDGIP2P' == self.args.prompt or 'UDGIP2GIP' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic_list), input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2I_new' == self.args.prompt or 'UDGIP2I_cot' == self.args.prompt or 'UDGIP2IP_cot' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'UDGIP2P_new' == self.args.prompt or 'UDGIP2P_cot' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know,
                                                         input4=data['user_profile'], label=label, mode=mode))
            elif 'DGIP2P_new' == self.args.prompt or 'DGIP2P_cot' == self.args.prompt or 'DGIP2P_cot_new' == self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                if self.args.selected_topic and 'selected_topic' in dataset_input[0]:
                    candidate_topics = f"Topic 1. {data['selected_topic']}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=candidate_topics, input3=predicted_know,
                                                         label=label, mode=mode))
            elif 'DIP2P' in self.args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic_list)])
                if self.args.selected_topic and 'selected_topic' in dataset_input[0]:
                    candidate_topics = f"Topic 1. {data['selected_topic']}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=candidate_topics, input2=predicted_know,
                                                         label=label, mode=mode))
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

        # instructions = [i.replace('\xa0', ' ').strip() for i in instructions]
        instructions = [i.replace('\xa0', ' ').replace('  ', ' ').strip() for i in instructions]
        print(len(instructions))
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
