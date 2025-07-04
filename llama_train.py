import math
import os
import sys
import random
from copy import deepcopy
from typing import List

import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, LlamaConfig, AutoModelForCausalLM
from utils.prompter import Prompter

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict, prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, TrainerCallback


class QueryEvalCallback(TrainerCallback):
    def __init__(self, args):
        self.log_name = args.log_name
        self.saved_model_path = args.saved_model_path
        self.no_save = args.no_save

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        epoch = state.epoch
        path = os.path.join(self.saved_model_path, self.log_name + '_E' + str(int(epoch)))
        if not self.no_save:
            model.save_pretrained(path)


def llama_finetune(
        args,
        tokenizer,
        instructions: list = None,
        train_know_dataset: list = None,
        labels: list = None,
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 2,
        num_epochs: int = 3,
        learning_rate: float = 4e-4,
        warmup_steps=200,
        val_set_size: int = 0,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        # train_on_inputs: bool = True,  # if False, masks out inputs in loss
        train_only_inputs: bool = True,
        train_only_outputs: bool = False,
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca_legacy",  # The prompt template to use, will default to alpaca.
):
    print('#' * 64)
    print('I\'M TRAINER for Sampling')
    print('#' * 64)

    base_model = args.base_model

    # global_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
    batch_size = args.batch_size
    per_device_batch_size = batch_size // args.num_device
    global_batch_size = args.global_batch_size
    global_batch_size = global_batch_size if global_batch_size > batch_size else batch_size
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * args.num_device)
    print(f"per_device_batch_size: {per_device_batch_size}\n"
          f"global_batch_size: {global_batch_size}\n"
          f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
    learning_rate = args.learning_rate
    resume_from_checkpoint = args.peft_weights
    prompt_template_name = args.prompt
    warmup_steps = args.warmup_steps
    # max_train_steps = num_epochs * math.ceil(math.ceil(len(instructions) / batch_size) / gradient_accumulation_steps)
    # warmup_steps = int(0.1 * max_train_steps)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"per_device_train_batch_size: {batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_only_inputs: {train_only_inputs}\n"
            f"train_only_outputs: {train_only_outputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # gradient_accumulation_steps = batch_size // micro_batch_size

    # device_map = "auto"
    device_map = {"": 0}

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("world_size: %d" % world_size)
    # ddp = world_size != 1
    if world_size != 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        print(device_map)
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                # and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['instruction']
        tokenized_full_prompt = tokenize(full_prompt)

        return tokenized_full_prompt

    if args.bf:
        fp16 = False
        bf16 = True
        dtype = torch.bfloat16
        print('#' * 64)
        print('bf16')
        print('#' * 64)
    else:
        fp16 = True
        bf16 = False
        dtype = torch.float16
        print('#' * 64)
        print('fp16')
        print('#' * 64)

    if args.quantization == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # default
            bnb_4bit_compute_dtype=torch.bfloat16  # divergence if fp16
        )  # 240414 추가
        print('#' * 64)
        print('4 bit')
        print('#' * 64)
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
        # print('#' * 64)
        print('8 bit')
        print('#' * 64)

    data = []
    for inst, lab in zip(train_know_dataset, labels):
        inst['output'] = lab
        data.append(inst) # keys: dialog, topic, predicted_know, candidate_knowledges_gpt

    first_sample = Dataset.from_pandas(pd.DataFrame([data[0]]))
    data = Dataset.from_pandas(pd.DataFrame(data))

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        # generate_and_tokenize_prompt(first_sample[0]) # Debug
        train_data = data.shuffle()  # .map(generate_and_tokenize_prompt)
        val_data = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype, 
        device_map=device_map, 
        quantization_config=quantization_config,
    )

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"  # Allow batched inference
    # tokenizer.add_eos_token = True  # Check eos token

    # model = prepare_model_for_int8_training(model) # Quantization
    model = prepare_model_for_kbit_training(model) # Quantization

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if resume_from_checkpoint != "":
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    else:
        resume_from_checkpoint = None

    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    class D2PDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, dataset):
            self.tokenizer = tokenizer
            self.dataset = dataset
            self.prompter = Prompter(args, args.prompt)
            self.print_result = True

        def prompting(self, data, predicted_goal, predicted_topic, predicted_know, label, mode='train'):
            # Inspired2
            ## Item selection ablation (w/o CoT)
            if 'DIP2I' == args.prompt:
                label = data['topic']
                if args.redial or args.inspired:
                    candidate_topics = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                else:
                    candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=candidate_topics, input2=predicted_know, label=label,
                                                            mode=mode)
            ## Item selection ours
            elif 'DIP2I_cot' == args.prompt:
                rationale = data['topic_cot'].split('Therefore')[0].strip()
                if args.redial or args.inspired:
                    label = f"{rationale} Therefore, the most suitable item is \"{data['topic']}\""
                    candidate_topics = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                else:
                    label = f"{rationale} Therefore, the most suitable topic is \"{data['topic']}\""
                    candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=candidate_topics, input2=predicted_know, label=label, mode=mode)
            ## Feature selection ablation (w/o CoT)
            elif 'DIP2P' == args.prompt:
                if label != '':
                    label = f"{label}"
                else:
                    label = "None of the passages are relevant for generating a response to the given dialog."

                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'],
                                                            input=candidate_topics,
                                                            input2=predicted_know,
                                                            label=label, mode=mode)
            ## Feature selection ours
            elif 'DIP2P_cot' == args.prompt:
                rationale = data['passage_cot'].split('Therefore')[0].strip()
                if label != '':
                    label = f"{rationale} Therefore, the most relevant passages are as follows:\n{label}."
                else:
                    label = "None of the passages are relevant for generating a response to the given dialog."
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'],
                                                            input=candidate_topics,
                                                            input2=predicted_know,
                                                            label=label, mode=mode)
            ## Response generation ours
            elif 'DP2R' in args.prompt:
                label = data['response']
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode)
            
            # DurecDial2
            ## Item selection ablation (w/o CoT)
            elif 'UDGIP2I_new' == args.prompt:
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=candidate_topics,
                                                            input3=predicted_know, input4=data['user_profile'],
                                                            label=data['topic'], mode=mode)
            ## Item selection ours
            elif 'UDGIP2I_cot' == args.prompt:
                rationale = data['topic_cot'].split('Therefore')[0].strip()
                label = f"{rationale} Therefore, the most suitable topic is \"{data['topic']}\""
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=candidate_topics,
                                                            input3=predicted_know, input4=data['user_profile'],
                                                            label=label, mode=mode)
            ## Feature selection ablation (w/o CoT)
            elif 'DGIP2P_new' == args.prompt:
                label = f"{label}."
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=candidate_topics,
                                                            input3=predicted_know,
                                                            label=label, mode=mode)
            ## Feature selection ours
            elif 'DGIP2P_cot_new' == args.prompt:
                rationale = data['passage_cot'].split('Therefore')[0].strip()
                label = f"{rationale} Therefore, the relevant passages are as follow:\n{label}"
                candidate_topics = '\n'.join([f"Topic {idx + 1}. {t}" for idx, t in enumerate(predicted_topic)])
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=candidate_topics,
                                                            input3=predicted_know,
                                                            label=label, mode=mode)
            ## Response generation ours
            elif 'DGP2R' in args.prompt:
                label = data['response']
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=predicted_know,
                                                            label=label, mode=mode)
            else:
                raise ValueError
            return full_prompt

        def __getitem__(self, idx):
            train_only_inputs = args.train_only_inputs
            train_only_outputs = args.train_only_outputs

            if args.weighted_loss:
                if idx % 2 == 0:
                    train_only_outputs = False
                else:
                    train_only_outputs = True

            data = self.dataset[idx]

            if args.positive == 'pseudo':
                target_knowledge = random.choice(data['predicted_know'][:args.n_pseudo])
            elif args.positive == 'highly_relevant':
                target_knowledge = random.choice(data['candidate_knowledges_gpt'])
            elif args.positive == 'gpt_selection':
                target_knowledge = data['gpt_selection']
            elif args.positive == 'target':
                target_knowledge = data['target_knowledge']
            else:
                raise ValueError

            candidate_knowledges_gpt = data['candidate_knowledges_gpt'][:args.n_sampled_positive]

            if args.query:
                predicted_goal = data['query']
            else:
                predicted_goal = data['predicted_goal'][0]

            topk_topic = args.topk_topic if data['combined'] else 1  # multiple items if data['combined']
            topk_topic = min(topk_topic, len(data['predicted_topic']))
            if args.topic_num_shuffle and topk_topic > 1:
                # item이 3개, 2개 섞어 들어감
                topk_topic = random.randint(2, topk_topic)

            if args.item_random_negative:
                topic_idx = [data['predicted_topic'].index(data['topic'])]
                while len(topic_idx) < topk_topic:
                    negative_idx = random.randrange(args.item_random_negative_num)  # idx = {0, 1, 2}
                    if negative_idx not in topic_idx:
                        topic_idx.append(negative_idx)
            else:
                topic_idx = [i for i in range(topk_topic)]
            random.shuffle(topic_idx)  # Doesn't matter if top-1 item only
            if data['predicted_topic']:
                predicted_topic_list = [data['predicted_topic'][i] for i in topic_idx]
            else:
                predicted_topic_list = []

            if args.selected_topic and 'selected_topic' in data:
                predicted_topic_list = [data['selected_topic']]

            if data['predicted_know']:
                top_negative_candidates = deepcopy([data['predicted_know'][i] for i in topic_idx])
                random_candidates = []
                for i in topic_idx:
                    tmp = deepcopy(data['predicted_know'][i])
                    random.shuffle(tmp)
                    random_candidates.append(tmp[:args.n_sampled_negative])
            else:
                top_negative_candidates = []
                random_candidates = []

            if data['combined']:
                for idx, top_passages in enumerate(top_negative_candidates):
                    top_negative_candidates[idx] = [i for i in top_passages if i not in candidate_knowledges_gpt and i != '']

                # Filtering code
                if args.filtering:
                    for idx, top_passages in enumerate(top_negative_candidates):
                        hard_negative_candidates_filtered = [passage for passage in top_passages if data['predicted_topic'][idx].lower().strip() in passage.lower().strip()]  # Filtering
                        hard_negative_candidates_unfiltered = [passage for passage in top_passages if data['predicted_topic'][idx].lower().strip() not in passage.lower().strip()]  # Filtering
                        if len(hard_negative_candidates_filtered) < args.n_sampled_negative:
                            hard_negative_candidates_filtered = hard_negative_candidates_filtered + hard_negative_candidates_unfiltered[:args.n_sampled_negative - len(hard_negative_candidates_filtered)]
                        top_negative_candidates[idx] = hard_negative_candidates_filtered

                for idx, predicted_topic in enumerate(predicted_topic_list):
                    if data['topic'].replace('\xa0', ' ').replace('  ', ' ').strip().lower() == predicted_topic.replace('\xa0', ' ').replace('  ', ' ').strip().lower():
                        top_negative_candidates[idx] = candidate_knowledges_gpt + top_negative_candidates[idx]

                for idx, top_passages in enumerate(top_negative_candidates):
                    if args.n_sampled_negative == -1:
                        top_negative_candidates[idx] = top_negative_candidates[idx][:]
                    else:
                        top_negative_candidates[idx] = top_negative_candidates[idx][:args.n_sampled_negative]

                if args.shuffle:
                    for idx, top_passages in enumerate(top_negative_candidates):
                        random.shuffle(top_negative_candidates[idx])

                predicted_know_list = []
                random_know_list = []

                for i in range(len(predicted_topic_list)):
                    predicted_know_list += top_negative_candidates[i]

                for i in range(len(predicted_topic_list)):
                    random_know_list += random_candidates[i]

                predicted_know = ""
                for i in range(len(predicted_topic_list)):
                    if args.redial or args.inspired:
                        prefix = f"Here are the candidate passages about Item {i + 1}. {predicted_topic_list[i]}"
                    else:
                        prefix = f"Here are the candidate passages about Topic {i + 1}. {predicted_topic_list[i]}"
                    if args.random_passages:
                        predicted_know_list = random_know_list
                        candidate_passages = '\n'.join(
                            [f"Passage {i * args.n_sampled_negative + idx + 1}. {know}" for idx, know in
                             enumerate(random_candidates[i])])
                    else:
                        candidate_passages = '\n'.join(
                            [f"Passage {i * args.n_sampled_negative + idx + 1}. {know}" for idx, know in
                             enumerate(top_negative_candidates[i])])
                    if "P2R" in args.prompt:
                        predicted_know += f"{candidate_passages}\n\n"
                    else:
                        predicted_know += f"{prefix}\n{candidate_passages}\n\n"

            else:
                hard_negative_candidates = [passage for passage in data['predicted_know'][0] if passage not in candidate_knowledges_gpt and passage != '']
                if args.n_hard_negative == -1:
                    n_hard_negative = args.n_sampled_negative - len(candidate_knowledges_gpt)
                else:
                    n_hard_negative = args.n_hard_negative

                if args.filtering:
                    hard_negative_candidates_filtered = [passage for passage in hard_negative_candidates if data['predicted_topic'][0].lower().strip() in passage.lower().strip()]  # Filtering
                    hard_negative_candidates_unfiltered = [passage for passage in hard_negative_candidates if data['predicted_topic'][0].lower().strip() not in passage.lower().strip()]  # Filtering
                    if len(hard_negative_candidates_filtered) < n_hard_negative:
                        hard_negative_candidates_filtered = hard_negative_candidates_filtered + hard_negative_candidates_unfiltered[:n_hard_negative - len(hard_negative_candidates_filtered)]
                    hard_negative_candidates = hard_negative_candidates_filtered

                hard_negative_candidates = hard_negative_candidates[:n_hard_negative]
                if args.shuffle:
                    random.shuffle(hard_negative_candidates)

                hard_negative_candidates = candidate_knowledges_gpt + hard_negative_candidates
                hard_negative_candidates = hard_negative_candidates[:args.n_sampled_negative]
                if args.shuffle:
                    random.shuffle(hard_negative_candidates)

                predicted_know_list = hard_negative_candidates

                predicted_know = '\n'.join([f"Passage {idx + 1}. {know}" for idx, know in enumerate(predicted_know_list)])

            if args.all_passages: # DP(all)2R ablation setting
                label = ""
            else: # MOCHA code
                relevant_idx = predicted_know_list.index(target_knowledge) if target_knowledge in predicted_topic_list else -1
                relevant_idx_list = []
                if not args.random_passages: # random_passages ablation
                    for x in candidate_knowledges_gpt:
                        relevant_idx_list.append(predicted_know_list.index(x))

                if args.candidate_knowledges_gpt:
                    label = '\n'.join([f"Passage {x + 1}. {y}" for x, y in zip(relevant_idx_list, candidate_knowledges_gpt)])

                elif args.target:
                    label = f"{predicted_know}"
                else:
                    label = f"Passage {relevant_idx + 1}. {target_knowledge}"

                if args.combined_top1:
                    if idx % 2 == 0 or args.input_top1:
                        predicted_topic_list = [data['predicted_topic'][0]] if data['topic'] == data['predicted_topic'][0] else [data['predicted_topic'][1]]

            if args.train_only_inputs:
                full_prompt = self.prompting(data, predicted_goal, predicted_topic_list, predicted_know, label, mode='test')
                full_prompt = full_prompt.replace('\xa0', ' ').replace('  ', ' ').strip()
                tokenized_full_prompt = tokenize(full_prompt, add_eos_token=False)
            else:
                full_prompt = self.prompting(data, predicted_goal, predicted_topic_list, predicted_know, label)
                full_prompt = full_prompt.replace('\xa0', ' ').replace('  ', ' ').strip()
                tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)

            if args.debug:
                print(full_prompt)
                print(train_only_outputs)

            if train_only_outputs:
                user_prompt = self.prompting(data, predicted_goal, predicted_topic_list, predicted_know, label, mode='test')

                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably

            return tokenized_full_prompt

        def __len__(self):
            return len(self.dataset)

    """ adamw_bnb_8bit adamw_8bit paged_adamw_32bit paged_adamw_8bit
    ['adamw_hf', 'adamw_torch', 'adamw_torch_fused',
     'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused',
      'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad',
       'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit',
        'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit']
    """

    trainer = Trainer(
        model=model,
        train_dataset=D2PDataset(tokenizer, train_data),
        args=transformers.TrainingArguments(
            num_train_epochs=num_epochs,
            deepspeed=args.deepspeed if args.deepspeed != '' else None,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            # max_steps = 100,
            learning_rate=learning_rate,
            logging_steps=10,
            lr_scheduler_type="cosine",
            output_dir=output_dir,
            optim="adamw_torch",
            # use 32bit adamW optimizer with paging. decoding error if use 4bit optimizer
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="no",
            fp16=False if args.deepspeed != '' else args.fp16_trainarg,  # fp16,
            bf16=bf16,  # True if BF16
            eval_steps=5 if val_set_size > 0 else None,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True),
        callbacks=[QueryEvalCallback(args)]
    )
    model.config.use_cache = False

    # Turn off the setting below
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # output_dir = os.path.join(args.saved_model_path, args.log_name + '_final')
    # model.save_pretrained(output_dir, safe_serialization=True)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
