import json
import os

from transformers import LlamaTokenizer

from llama_test import LLaMaEvaluator
# from llama_train import llama_finetune
from utils.parser import parse_args
from utils.prompter import Prompter
from utils.utils import dir_init, createLogFile, load_dataset, prepare_dataset, merge_dataset_passages, augment_dataset
import pickle

if __name__ == "__main__":

    # fire.Fire(llama_finetune)
    args = parse_args()
    args = dir_init(args)
    args = createLogFile(args)
    print(args)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    train_raw_dataset, test_raw_dataset = load_dataset(args)

    train_know_dataset = merge_dataset_passages(args, train_raw_dataset, mode='train')
    test_know_dataset = merge_dataset_passages(args, test_raw_dataset, mode='test')

    train_know_dataset, train_labels, train_topics = prepare_dataset(args, tokenizer, train_know_dataset)
    test_know_dataset, test_labels, test_topics = prepare_dataset(args, tokenizer, test_know_dataset)

    if 'D2P' in args.prompt and args.positive == 'highly_relevant':
        train_know_dataset, train_labels, train_topics = augment_dataset(train_know_dataset, train_labels, train_topics)
        # test_know_dataset, test_labels, test_topics = augment_dataset(test_know_dataset, test_labels, test_topics)

    prompter = Prompter(args, args.prompt)
    train_instructions = prompter.generate_instructions('train', train_know_dataset, train_labels)
    test_instructions = prompter.generate_instructions('test', test_know_dataset, test_labels)

    prompt_list = ["D2P", "DI2P", "DP2I", "UDP2I"]

    if args.mode == 'train':
        if args.prompt in prompt_list: from llama_train import llama_finetune
        else: from llama_train_gen import llama_finetune
        llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, train_know_dataset=train_know_dataset, labels=train_labels, num_epochs=args.epoch)
    elif args.mode == 'test':
        LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, topics=test_topics, prompt_template_name=args.prompt).test()
    elif args.mode == 'train_test':
        if args.sft:
            from llama_train_sft import llama_finetune_sft
            llama_finetune_sft(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
        else:
            if args.prompt in prompt_list: from llama_train import llama_finetune
            else: from llama_train_gen import llama_finetune
            llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, train_know_dataset=train_know_dataset, labels=train_labels, num_epochs=args.epoch)

        for e in range(args.epoch):
            args.peft_weights = os.path.join(args.saved_model_path, args.log_name + '_E' + str(int(e + 1)))
            print(f"loading peft model: {args.peft_weights}")
            LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, topics=test_topics, prompt_template_name=args.prompt).test(epoch=e+1)
