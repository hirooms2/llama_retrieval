import json
import os
from random import random
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "args")

    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.args = args
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            if args.stage == "crs":
                template_name = "withoutCoT"
            elif args.stage == "quiz":
                template_name = "alpaca_legacy"
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

            if self.args.prompt == 'UD2I':
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=data['user_profile'], label=label, mode=mode))
            elif self.args.prompt == 'DP2R':
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif self.args.prompt == 'DP2I':
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode))
            elif self.args.prompt == 'UDP2I':
                instructions.append(self.generate_prompt(instruction=data['dialog'], input=data['user_profile'], input2=predicted_know, label=label, mode=mode))
            elif self.args.prompt == 'pretrain':
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))

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
