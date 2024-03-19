import argparse
from datetime import datetime
import json
import pickle
from typing import Union
import os

import torch
from loguru import logger
from pytz import timezone
from tqdm import tqdm


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

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
            isNew: bool = False,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()


def load_dataset(args):
    print('LLAMA_DATASET')
    train_file_path = os.path.join(args.home, 'data/train_pred_aug_dataset.pkl')
    with open(file=train_file_path, mode='rb') as f:
        train_dataset = pickle.load(f)

    test_file_path = os.path.join(args.home, 'data/test_pred_aug_dataset.pkl')
    with open(file=test_file_path, mode='rb') as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset


def prepare_dataset(dataset):
    instructions = []
    labels = []
    for data in dataset:
        instruction = data['dialog'].replace('[SEP]', '\n')
        label = data['candidate_knowledges'][0]
        instructions.append(instruction)
        labels.append(label)

    return instructions, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--peft', type=str, default="lora")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--prompt', type=str, default="D2P")
    parser.add_argument('--peft_weights', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)

    parser.add_argument('--cutoff', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    parser.add_argument('--prompt_template_name', type=str, default='D2P')

    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--num_beams', type=int, default=5)

    args = parser.parse_args()
    args.num_device = torch.cuda.device_count()

    return args


def dir_init(default_args):
    from copy import deepcopy
    """ args 받은다음, device, Home directory, data_dir, log_dir, output_dir, 들 지정하고, Path들 체크해서  """
    args = deepcopy(default_args)
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
        print(args.home)
    elif sysChecker() == "Windows":
        args.home = ''
        # args.batch_size, args.num_epochs = 4, 2
        # args.debug = True
        pass  # HJ local
    else:
        raise Exception("Check Your Platform Setting (Linux-Server or Windows)")

    return args


def createLogFile(args):
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = 'llama_result'
    else:
        log_name = args.log_name
    args.log_name = mdhm + '_' + log_name

    args.output_dir = 'result'
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    result_path = os.path.join(args.output_dir, args.base_model.replace('/', '-'))
    args.result_path = result_path  # os.path.join(args.home, result_path)
    if not os.path.exists(args.result_path): os.mkdir(args.result_path)

    log_file = open(os.path.join(args.result_path, log_name + ".json"), 'a', buffering=1, encoding='UTF-8')
    args.log_file = log_file

    saved_model_path = os.path.join(args.home, 'saved_model')
    args.saved_model_path = saved_model_path

    if not os.path.exists(saved_model_path): os.mkdir(saved_model_path)
    args.peft_weights = os.path.join(saved_model_path, args.peft_weights)

    return args


def cutoffInstruction(tokenizer, instructions, length, reverse=False):
    new_instructions = []
    for data in tqdm(instructions):
        if reverse:
            data = tokenizer.decode(tokenizer(data).input_ids[1:][-length:])
        else:
            data = tokenizer.decode(tokenizer(data).input_ids[1:][:length])
        new_instructions.append(data)
    logger.info('[Finish Cutting-off the instructions]')
    return new_instructions
