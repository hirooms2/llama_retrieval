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
    set_peft_model_state_dict, prepare_model_for_kbit_training
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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    data = []
    for inst, lab in zip(instructions, labels):
        data.append({"instruction": inst, "label": lab})

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
        train_data = data.shuffle()
        val_data = None

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map='auto',
        quantization_config=quantization_config,
    )

    # if args.debug:
    #     configuration = LlamaConfig(num_hidden_layers=1)
    #     model = LlamaForCausalLM(configuration)
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         torch_dtype=torch.float16,
    #         device_map=device_map,
    #         quantization_config=quantization_config,
    #     )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules='all-linear',
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    torch.cuda.empty_cache()
    print(f"Train_data input_ids[0] contents \n{train_data[0]}\n")
    print(per_device_train_batch_size)
    print(gradient_accumulation_steps)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="instruction",
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
            optim="paged_adamw_8bit",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=5 if val_set_size > 0 else None,
            save_steps=200,
            # load_best_model_at_end=True,
            report_to="none",
            # compute_metrics = compute_metrics_gen, compute_met
            gradient_checkpointing=True,  # Leads to reduction in memory at slighly decrease in speed
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, mlm=False),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        callbacks=[QueryEvalCallback(args)],

    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_data,
    #     dataset_text_field="text",
    #     peft_config=peft_config,
    #     args=transformers.TrainingArguments(
    #         num_train_epochs=num_epochs,
    #         per_device_train_batch_size=per_device_train_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=warmup_steps,
    #         learning_rate=learning_rate,
    #         fp16=False,
    #         bf16=False,
    #         logging_steps=10,
    #         optim="paged_adamw_32bit",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=5 if val_set_size > 0 else None,
    #         save_steps=200,
    #         output_dir=output_dir,
    #         save_total_limit=3,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         # run_name=args.wandb_run_name if use_wandb else None,
    #     ),
    #     callbacks=[QueryEvalCallback(args)],
    #     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # )

    # trainer = Trainer(
    #     model=model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=per_device_train_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=warmup_steps,
    #         num_train_epochs=num_epochs,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=10,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=5 if val_set_size > 0 else None,
    #         save_steps=200,
    #         output_dir=output_dir,
    #         save_total_limit=3,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         # run_name=args.wandb_run_name if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    #     callbacks=[QueryEvalCallback(args)]
    # )
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
    # trainer.train()

    # model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
