import os

from transformers import LlamaTokenizer

from llama_test import LLaMaEvaluator
from llama_train import llama_finetune
from llama_util import parse_args, dir_init, createLogFile, load_dataset, prepare_dataset, cutoffInstruction

if __name__ == "__main__":
    # fire.Fire(llama_finetune)
    args = parse_args()
    args = dir_init(args)
    args = createLogFile(args)

    train_dataset, test_dataset = load_dataset(args)
    train_instructions, train_labels = prepare_dataset(train_dataset)
    test_instructions, test_labels = prepare_dataset(test_dataset)

    if args.debug:
        train_instructions = train_instructions[:100]
        train_labels = train_labels[:100]
        test_instructions = test_instructions[:100]
        test_labels = test_labels[:100]

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    train_instructions = cutoffInstruction(tokenizer, train_instructions, args.cutoff, reverse=True)
    test_instructions = cutoffInstruction(tokenizer, test_instructions, args.cutoff, reverse=True)

    if 'train' in args.mode:
        llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
    if 'test' in args.mode:
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, prompt_template_name=args.prompt).test()
