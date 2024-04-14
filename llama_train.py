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
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
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
    print('#' * 16)
    print('I\'M TRAINER!!!!!!!!!!!!!!!!!!!!!!')

    base_model = args.base_model
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.num_device  # update the model's weights once every gradient_accumulation_steps batches instead of updating the weights after every batch.
    per_device_train_batch_size = batch_size // args.num_device
    resume_from_checkpoint = args.peft_weights
    prompt_template_name = args.prompt

    # if args.warmup != 0:
    #     max_train_steps = num_epochs * math.ceil(math.ceil(len(instructions) / batch_size) / gradient_accumulation_steps)
    #     warmup_steps = int(args.warmup * max_train_steps)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
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

    device_map = "auto"
    # device_map = "cuda"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("world_size: %d" % world_size)
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",  # nomalized 라 하던데, 그냥 default 로 쓰는 것인듯
    #     bnb_4bit_compute_dtype=torch.bfloat16  # fp16으로 하면 발산함
    # )  # 240414 추가

    data = []
    for inst, lab in zip(instructions, labels):
        data.append({"instruction": inst, "input": "", "output": lab})

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
        torch_dtype=torch.float16,  # 의미 없음 -> 오히려 빨라지는 양상?
        device_map={"": 0},  # 만일 multi-GPU를 'auto', 240414 추가
        quantization_config=quantization_config,  # 240414 추가
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # "left" 이거 right로 하면 학습안됨?"" # Allow batched inference
    tokenizer.add_eos_token = True  # 이렇게 했을 때, 마지막에 eos 붙는거 확인.. 위치는 SFTtrainer 안에 _prepare_dataset() 내에서 진행. 240414 추가

    model = prepare_model_for_int8_training(model)
    # model = prepare_model_for_kbit_training(model)  # 얘 하면 시간 더 오래 걸리는데, 어떤 역할을 하는지 모르겠음 -> 어떨때는 또 오래 안걸림

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

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

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
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            # max_steps = 100,
            learning_rate=learning_rate,
            logging_steps=10,
            output_dir=output_dir,
            optim="adamw_torch",  # paging 기법이 적용된 adamW optimizer 를 쓰는데, 32 bit 씀. 이거 4bit로 하면 decoding 할 때 에러나는 경우가 있음.
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="no",
            fp16=True,
            bf16=False,  # BF16으로 하는 거면 True
            eval_steps=5 if val_set_size > 0 else None,
            report_to="none",
            # gradient_checkpointing=True,  # 이거 없으면 메모리 엄청 먹음.
            # gradient_checkpointing_kwargs={"use_reentrant": False},  # 얘는 위에거랑 세트
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[QueryEvalCallback(args)]
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
    #
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # output_dir = os.path.join(args.saved_model_path, args.log_name + '_final')
    # model.save_pretrained(output_dir, safe_serialization=True)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
