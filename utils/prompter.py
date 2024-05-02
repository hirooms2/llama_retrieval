import json
import os
import random
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
            if 'predicted_know' in data:
                predicted_know = data['predicted_know'][:self.args.n_docs]
                if mode == 'train':
                    random.shuffle(predicted_know)
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
                guide = f"Goal: {data['predicted_goal'][0]} | Topic: {' or '.join(data['predicted_topic'][:2])}"
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=guide, label=label, mode=mode))
            elif 'DG2P' in self.args.prompt:
                # num_items = 2 if mode == 'train' else 1
                guide = f"Goal: {data['predicted_goal'][0]} | Topic: {' or '.join(data['predicted_topic'][:2])}"
                # instructions.append(self.generate_prompt(instruction=data['dialog'],  label=label, mode=mode))
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=guide, label=label, mode=mode))
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
            label: Union[None, str] = None,
            mode: str = 'test') -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input and not input2:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        elif input and input2:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2
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
