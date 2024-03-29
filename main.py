from transformers import LlamaTokenizer

from llama_test import LLaMaEvaluator
from llama_train import llama_finetune
# from llama_train_sft import llama_finetune
from utils.parser import parse_args
from utils.prompter import Prompter
from utils.utils import dir_init, createLogFile, load_dataset, prepare_dataset, merge_dataset_passages

if __name__ == "__main__":

    # fire.Fire(llama_finetune)
    args = parse_args()
    args = dir_init(args)
    args = createLogFile(args)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    train_raw_dataset, test_raw_dataset = load_dataset(args)

    train_know_dataset = merge_dataset_passages(args, train_raw_dataset, mode='train')
    test_know_dataset = merge_dataset_passages(args, test_raw_dataset, mode='test')

    train_know_dataset, train_labels = prepare_dataset(args, tokenizer, train_know_dataset)
    test_know_dataset, test_labels = prepare_dataset(args, tokenizer, test_know_dataset)

    prompter = Prompter(args, args.prompt)
    train_instructions = prompter.generate_instructions('train', train_know_dataset, train_labels)
    test_instructions = prompter.generate_instructions('test', test_know_dataset, test_labels)

    if args.mode == 'train':
        llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
    elif args.mode == 'test':
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, prompt_template_name=args.prompt).test()
    elif args.mode == 'train_test':
        llama_finetune(args, tokenizer=tokenizer, instructions=train_instructions, labels=train_labels, num_epochs=args.epoch)
        for e in range(args.epoch):
            evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, insturctions=test_instructions, labels=test_labels, prompt_template_name=args.prompt).test(epoch=e)
