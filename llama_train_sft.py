import os
import sys
from typing import List

import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, LlamaConfig
from trl import SFTTrainer

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
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
        model.save_pretrained(path, safe_serialization=False)


def llama_finetune_sft(
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

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
    compute_dtype = getattr(torch, 'float16')
    if compute_dtype == torch.float16:  # and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)  # 만일 아래 메세지가 뜨면 bf16을 쓰고, 아니면 fp16을 쓰는게 좋을 듯 -> 밑에 SFTTrainer에 fp16, bf16 중 하나 True해야할 듯 -> 안해도 돌아가긴 하는데, 해야 맞을듯
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 몇 비트 연산을 할 것인가에 대한 것인데, 최근에는 아래 4비트로 연산하는 듯. 오히려 8비트가 에러가 존재하는 듯? -> 실제적인 메모리 사용량 차이는 크게 안나는 듯
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # nomalized 라 하던데, 그냥 default 로 쓰는 것인듯
        bnb_4bit_compute_dtype=torch.bfloat16  # torch.bfloat16, # torch.float 이 조금 더 빠름. 설명상으로는 bfloat16이 조금 더 정교한 방법이라 함. 실제적인 정확도 차이는 모르겠음 (실험 필요)
    )

    data = []
    for inst, lab in zip(instructions, labels):
        data.append({"instruction": inst, "label": lab})

    first_sample = Dataset.from_pandas(pd.DataFrame([data[0]]))
    data = Dataset.from_pandas(pd.DataFrame(data))

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle()  # .map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle()  # .map(generate_and_tokenize_prompt)
        )
        train_data = train_val["train"].shuffle()
        val_data = train_val["test"].shuffle()
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
        # torch_dtype=torch.float16, # 의미 없음 -> 오히려 빨라지는 양상?
        device_map=device_map,
        quantization_config=quantization_config,
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference + SFT 쓰면 무조건 얘 하라고 메세지 뜸

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)  # 얘 하면 시간 더 오래 걸리는데, 어떤 역할을 하는지 모르겠음 -> 어떨때는 또 오래 안걸림

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,  # 'all-linear'로 할 시 학습 파라미터 수 증가 -> 시간/메모리 더 오래 걸림 & 근데 아마 정확도는 더 오를 듯
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # model = get_peft_model(model, peft_config)
    model = get_peft_model(model, peft_config)
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

    # tokenizer.pad_token = tokenizer.eos_token # 많은 코드들이 이렇게 하는데, 이러면 EOS 학습이 안되지 않나?
    tokenizer.add_eos_token = True  # 이렇게 했을 때, 마지막에 eos 붙는거 확인.. 위치는 SFTtrainer 안에 _prepare_dataset() 내에서 진행.
    torch.cuda.empty_cache()  # 이거 쓰면 좋을게 있는지 모르겠는데, 일단 사용
    print(f"Train_data input_ids[0] contents \n{train_data[0]}\n")
    print(per_device_train_batch_size)
    print(gradient_accumulation_steps)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="instruction",  # 사실상 얘가 tokenize 돼서, input이자 labels가 됨
        peft_config=peft_config,
        args=transformers.TrainingArguments(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            # max_steps = 100,
            learning_rate=learning_rate,
            logging_steps=10,
            output_dir=output_dir,
            optim="paged_adamw_8bit",  # paging 기법이 적용된 adamW optimizer 를 쓰는데, 8bit 연산을 함 
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            bf16=True,
            eval_steps=5 if val_set_size > 0 else None,
            save_steps=200,
            report_to="none",
            gradient_checkpointing=True,  # 이거 없으면 메모리 엄청 먹음.
            gradient_checkpointing_kwargs={"use_reentrant": False},  # 얘는 위에거랑 세트
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, mlm=False),  # pad_to_multiple_of=8은 텐서를 8의 배수의 크기로 맞춘다는 것인데, 메모리 (혹은 연산속도) 상 이점이 있다 함
        callbacks=[QueryEvalCallback(args)],

    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference! -> 필요한지 잘 모르겠음. 대세엔 영향 없어보이긴 함
    trainer.train()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
