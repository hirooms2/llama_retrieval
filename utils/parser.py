import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)

    parser.add_argument('--log_name', type=str, default="")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--write', type=bool, default=True)
    parser.add_argument('--sft', type=bool, default=False)
    parser.add_argument('--bf', type=bool, default=False)
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--quantization', type=str, default="8bit")

    parser.add_argument('--peft', type=str, default="lora")
    parser.add_argument('--prompt', type=str, default="D2P", choices=['D2P', 'DP2R', 'UDP2I', 'pretrain', 'DP2R_original', 'DP2R_new', 'DP2R_v4', 'DP2R_v3', 'DP2R_v5'])
    parser.add_argument('--train_know_file', type=str, default="espresso")
    parser.add_argument('--test_know_file', type=str, default="espresso")
    parser.add_argument('--peft_weights', type=str, default="")

    # For training config
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--n_docs', type=int, default=2)
    parser.add_argument('--n_pseudo', type=int, default=2)
    parser.add_argument('--positive', type=str, default='pseudo', choices=['only_pseudo', 'pseudo', 'highly_relevant'])

    parser.add_argument('--cutoff', type=int, default=256)
    parser.add_argument('--passage_cutoff', type=int, default=128)

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    # parser.add_argument('--prompt_template_name', type=str, default='D2P')

    # For generation config
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--device', '--gpu', default='0', type=str, help='GPU Device')

    args = parser.parse_args()
    args.num_device = torch.cuda.device_count()

    if 'I' in args.prompt.split('2')[-1]:
        args.task = 'topic'
    elif 'P' in args.prompt.split('2')[-1]:
        args.task = 'know'
    elif 'R' in args.prompt.split('2')[-1]:
        args.task = 'resp'
    elif args.prompt == 'pretrain':
        args.task = 'pretrain'
    else:
        raise ValueError

    return args
