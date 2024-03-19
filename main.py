import os

from transformers import LlamaTokenizer

from llama_test import LLaMaEvaluator
from llama_train import llama_finetune
from llama_util import parse_args, dir_init, createLogFile, load_dataset, prepare_dataset, cutoffInstruction, Prompter

if __name__ == "__main__":
    # fire.Fire(llama_finetune)
    args = parse_args()
    args = dir_init(args)
    args = createLogFile(args)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    train_dataset, test_dataset = load_dataset(args)
    train_dialogs, train_profiles, train_labels = prepare_dataset(args, tokenizer, train_dataset)
    test_dialogs, train_profiles, test_labels = prepare_dataset(args, tokenizer, test_dataset)

    prompter = Prompter(args, args.prompt)
    train_instructions = prompter.generate_instructions('train', train_dialogs, train_profiles, train_labels)
    test_instructions = prompter.generate_instructions('test', train_dialogs, train_profiles, train_labels)
    #
    # train_instructions = [prompter.generate_prompt(instruction=instruction) for instruction in self.instructions]
    # train_dialogs = cutoffInstruction(tokenizer, train_dialogs, args.cutoff, reverse=True)
    # test_dialogs = cutoffInstruction(tokenizer, test_dialogs, args.cutoff, reverse=True)

    if 'train' in args.mode:
        llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
    if 'test' in args.mode:
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, prompt_template_name=args.prompt).test()
