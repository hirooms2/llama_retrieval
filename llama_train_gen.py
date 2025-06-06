import os
import sys
from typing import List

import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, LlamaConfig

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
        batch_size: int = 128,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        warmup_steps=100,
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
    print('I\'M TRAINER for GEN')
    print('#' * 64)

    base_model = args.base_model
    # train_on_inputs = args.train_on_inputs

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
            f"per_device_train_batch_size: {per_device_batch_size}\n"
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
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16  # fp16 --> divergence
        )
        print('#' * 64)
        print('4 bit')
        print('#' * 64)
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
        # print('#' * 64)
        print('8 bit')
        print('#' * 64)

    # data = []
    # for inst, lab in zip(train_know_dataset, labels):
    #     data.append({"instruction": inst, "input": "", "output": lab})
    data = [{"instruction": i} for i in instructions]

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
        generate_and_tokenize_prompt(first_sample[0])
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # if args.debug:
    #     configuration = LlamaConfig(num_hidden_layers=1)
    #     model = LlamaForCausalLM(configuration)
    # else:
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # tokenizer.add_eos_token = True  # Check eos

    # model = prepare_model_for_int8_training(model)
    model = prepare_model_for_kbit_training(model)

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

    # train_args = transformers.TrainingArguments(
    #     per_device_train_batch_size=per_device_train_batch_size,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     warmup_steps=warmup_steps,
    #     num_train_epochs=num_epochs,
    #     learning_rate=learning_rate,
    #     fp16=True,
    #     logging_steps=10,
    #     optim="adamw_torch",
    #     evaluation_strategy="steps" if val_set_size > 0 else "no",
    #     save_strategy="steps",
    #     eval_steps=5 if val_set_size > 0 else None,
    #     save_steps=200,
    #     output_dir=output_dir,
    #     save_total_limit=3,
    #     load_best_model_at_end=True if val_set_size > 0 else False,
    #     ddp_find_unused_parameters=False if ddp else None,
    #     group_by_length=group_by_length,
    #     report_to="wandb" if use_wandb else None,
    #     # run_name=args.wandb_run_name if use_wandb else None,
    # ),
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
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
            # use 32bit adamW optimizer with paging. decoding error if use 4bit optimizer
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="no",
            fp16=False if args.deepspeed != '' else args.fp16_trainarg,  # fp16,
            bf16=bf16,  # True if BF16
            eval_steps=5 if val_set_size > 0 else None,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
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

    print(tokenizer.decode(train_data[0]['input_ids']))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # output_dir = os.path.join(args.saved_model_path, args.log_name + '_final')
    # model.save_pretrained(output_dir, safe_serialization=True)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
