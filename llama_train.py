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
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, LlamaConfig
from utils.prompter import Prompter

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# JP
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

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        epoch = state.epoch
        path = os.path.join(self.saved_model_path, self.log_name + '_E' + str(int(epoch)))
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
        learning_rate: float = 3e-4,
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
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
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
    train_on_inputs = args.train_on_inputs

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
            f"train_on_inputs: {train_on_inputs}\n"
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
            bnb_4bit_quant_type="nf4",  # nomalized 라 하던데, 그냥 default 로 쓰는 것인듯
            bnb_4bit_compute_dtype=torch.bfloat16  # fp16으로 하면 발산함
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
        data.append(inst)
        # data.append({"dialog": inst['dialog'], "topic": inst['topic'], "predicted_know": inst['predicted_know'], "candidate_knowledges_gpt": inst['candidate_knowledges_gpt'], "output": lab})

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
        # generate_and_tokenize_prompt(first_sample[0])
        train_data = data.shuffle()  # .map(generate_and_tokenize_prompt)
        val_data = None

    # if args.debug:
    #     configuration = LlamaConfig(num_hidden_layers=1)
    #     model = LlamaForCausalLM(configuration)
    # else:
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,  # 의미 없음 -> 오히려 빨라지는 양상? 이거 BF16으로 한번 해보기?
        device_map=device_map,  # {"": 0},  # device_map,  # {"": 0},  # 만일 multi-GPU를 'auto', 240414 추가
        quantization_config=quantization_config,  # 240414 추가
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # "left" 이거 right로 하면 학습안됨?"" # Allow batched inference
    # tokenizer.add_eos_token = True  # 이렇게 했을 때, 마지막에 eos 붙는거 확인.. 위치는 SFTtrainer 안에 _prepare_dataset() 내에서 진행. 240414 추가

    # model = prepare_model_for_int8_training(model)
    model = prepare_model_for_kbit_training(model)  # 얘 하면 시간 더 오래 걸리는데, 어떤 역할을 하는지 모르겠음 -> 어떨때는 또 오래 안걸림

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

        def prompting(self, data, predicted_goal, predicted_topic, predicted_know, label, mode='train'):
            if 'D2P' in args.prompt:
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode)
            elif 'DI2P' in args.prompt:
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=data['topic'], label=label, mode=mode)
            elif 'DP2I' == args.prompt:
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode)
            elif 'DG2P' == args.prompt:
                # num_items = 2 if mode == 'train' else 1
                guide = f"Goal:{predicted_goal} | Topic:{' or '.join(data['topic'])}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=guide, label=label, mode=mode)
            elif 'DP2GP' == args.prompt:
                guide = f"Goal:{predicted_goal}: {data['topic']}"
                label = f"{guide}\nPassage:{label}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, label=label, mode=mode)
            elif 'DGIP2GIP' == args.prompt:
                label = f"Goal:{predicted_goal}\nTopic:{data['topic']}\nPassage:{label}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal,
                                                            input2=", ".join(predicted_topic), input3=predicted_know, label=label, mode=mode)
            elif 'UDGIP2GIP' == args.prompt:
                label = f"Goal:{predicted_goal}\nTopic:{data['topic']}\nPassage:{label}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic),
                                                            input3=predicted_know, input4=data['user_profile'], label=label, mode=mode)
            elif 'UDGIP2P' == args.prompt:
                label = f"Passage:{label}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic),
                                                            input3=predicted_know, input4=data['user_profile'], label=label, mode=mode)
            elif 'DGP2P' == args.prompt:
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=predicted_know, label=label, mode=mode)
            elif 'UDGIP2GI' == args.prompt:
                label = f"Goal:{predicted_goal}\nTopic:{data['topic']}"
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_goal, input2=", ".join(predicted_topic),
                                                            input3=predicted_know, input4=data['user_profile'], label=label, mode=mode)

            elif 'UDP2GP' == args.prompt:
                guide = f"Goal:{predicted_goal}: {data['topic']}"
                label = f"{guide}\nPassage:{label}"
                profile = data['user_profile']
                full_prompt = self.prompter.generate_prompt(instruction=data['dialog'], input=predicted_know, input2=profile, label=label, mode=mode)
            else:
                raise ValueError
            return full_prompt

        def __getitem__(self, idx):
            data = self.dataset[idx]

            predicted_know = []
            if args.positive == 'pseudo':
                target_knowledge = random.choice(data['predicted_know'][:args.n_pseudo])
            elif args.positive == 'highly_relevant':
                target_knowledge = random.choice(data['candidate_knowledges_gpt'])
            elif args.positive == 'gpt_selection':
                target_knowledge = data['gpt_selection']
            else:
                raise ValueError

            topic_idx = [i for i in range(args.topk_topic)]
            random.shuffle(topic_idx)
            predicted_topic = [data['predicted_topic'][i] for i in topic_idx]

            if args.query:
                predicted_goal = data['query']
            else:
                predicted_goal = data['predicted_goal'][0]

            if args.combined:
                n_partition_negative = int(args.n_hard_negative / 2)
                top1_hard_negative_candidates = [item for item in data['predicted_know'][0:0 + n_partition_negative] if item != target_knowledge]
                top2_hard_negative_candidates = [item for item in data['predicted_know'][10:10 + n_partition_negative] if item != target_knowledge]

                top1_hard_negative_candidates = [f"{data['predicted_topic'][0]}|{i}" for i in top1_hard_negative_candidates]
                top2_hard_negative_candidates = [f"{data['predicted_topic'][1]}|{i}" for i in top2_hard_negative_candidates]

                if data['topic'] == data['predicted_topic'][0]:
                    # target_knowledge = f"{data['predicted_topic'][0]}|{target_knowledge}"
                    top1_hard_negative_candidates.append(target_knowledge)
                elif data['topic'] == data['predicted_topic'][1]:
                    # target_knowledge = f"{data['predicted_topic'][1]}|{target_knowledge}"
                    top2_hard_negative_candidates.append(target_knowledge)

                random.shuffle(top1_hard_negative_candidates)
                random.shuffle(top2_hard_negative_candidates)
                top_hard_negative_candidates_list = [top1_hard_negative_candidates, top2_hard_negative_candidates]
                predicted_know = top_hard_negative_candidates_list[topic_idx[0]] + top_hard_negative_candidates_list[topic_idx[1]]
            else:
                hard_negative_candidates = [item for item in data['predicted_know'] if item != data['gpt_selection']]
                hard_negative_candidates = hard_negative_candidates[:args.n_hard_negative]
                predicted_know.append(target_knowledge)

                if len(set(hard_negative_candidates)) + 1 < args.n_sampled_negative:
                    n_sampled_negative = len(set(hard_negative_candidates)) + 1
                else:
                    n_sampled_negative = args.n_sampled_negative
                while len(predicted_know) < n_sampled_negative:
                    selected = random.choice(hard_negative_candidates)
                    if selected not in predicted_know:
                        predicted_know.append(selected)
                random.shuffle(predicted_know)

            relevant_idx = predicted_know.index(target_knowledge)
            predicted_know = '\n'.join([f"{idx + 1}. {know}" for idx, know in enumerate(predicted_know)])
            label = f"{relevant_idx + 1}. {target_knowledge}"

            full_prompt = self.prompting(data, predicted_goal, predicted_topic, predicted_know, label)

            tokenized_full_prompt = tokenize(full_prompt)
            # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # item['labels'] = torch.tensor(self.labels[idx])
            if not train_on_inputs:
                user_prompt = self.prompting(data, predicted_goal, predicted_topic, predicted_know, label, mode='test')

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
        # eval_dataset=val_data,
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
            # paging 기법이 적용된 adamW optimizer 를 쓰는데, 32 bit 씀. 이거 4bit로 하면 decoding 할 때 에러나는 경우가 있음. paged_adamw_32bit???
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="no",
            fp16=False if args.deepspeed != '' else args.fp16_trainarg,  # fp16,  # ,
            bf16=bf16,  # BF16으로 하는 거면 True
            eval_steps=5 if val_set_size > 0 else None,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[QueryEvalCallback(args)]
    )
    model.config.use_cache = False

    # 이거 켜놓으면 절대 안됨
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
