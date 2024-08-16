import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--test_continue', type=int, default=0)

    parser.add_argument('--log_name', type=str, default="")
    parser.add_argument('--train_data', type=str, default="new")
    parser.add_argument('--test_data', type=str, default="new")

    parser.add_argument('--log_dir', default='logs', type=str, help="logging file directory")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--write', type=bool, default=True)
    parser.add_argument('--sft', type=bool, default=False)
    parser.add_argument('--bf', type=bool, default=False)
    parser.add_argument('--fp16_trainarg', type=bool, default=False)

    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--quantization', type=str, default="8bit")

    parser.add_argument('--peft', type=str, default="lora")
    parser.add_argument('--prompt', type=str, default="D2P")
    parser.add_argument('--train_know_file', type=str, default="espresso")
    parser.add_argument('--test_know_file', type=str, default="espresso")
    parser.add_argument('--peft_weights', type=str, default="")
    parser.add_argument('--local_rank', type=int, default=-1)

    # For training config
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--global_batch_size', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=200)

    parser.add_argument('--n_docs', type=int, default=2)
    parser.add_argument('--n_pseudo', type=int, default=2)
    parser.add_argument('--n_hard_negative', type=int, default=-1)
    parser.add_argument('--n_sampled_negative', type=int, default=3)

    parser.add_argument('--positive', type=str, default='pseudo', choices=['only_pseudo', 'pseudo', 'highly_relevant', 'gpt_selection', 'target'])
    parser.add_argument("--pseudo", action='store_true', help="combine pseudo")
    parser.add_argument("--combined", action='store_true', help="combine pseudo")
    parser.add_argument("--combined_top1", action='store_true', help="combine combined but top-1")
    parser.add_argument("--input_top1", action='store_true', help="combine combined but top-1")
    parser.add_argument("--target", action='store_true',help="Input target knowledge in response generation task")
    parser.add_argument('--redial', action='store_true', help="Input redial dataset")
    parser.add_argument('--inspired',action='store_true',help='Input inspired dataset')
    parser.add_argument('--force_gpt', action='store_true', help="Insert topic in force")
    parser.add_argument('--force_topic', action='store_true', help="Insert topic in force")
    parser.add_argument('--force_split', action='store_true', help="split in dp2r task if topic in top-3")

    parser.add_argument("--partition", action='store_true', help="combine pseudo")
    parser.add_argument("--query", action='store_true', help="original goal vs. refined goal")
    parser.add_argument("--filtering", action='store_true', help="filter-out if a passage does not contain the predicted topic")
    parser.add_argument("--shuffle", action='store_true', help="shuffle passages if task is passage selection")
    parser.add_argument("--selected_topic", action='store_true', help="selected_topic")
    parser.add_argument("--candidate_knowledges_gpt", action='store_true', help="selected_topic")
    parser.add_argument("--n_sampled_positive", type=int, default=3)

    parser.add_argument("--postfix", action='store_true', help="Add postfix in GEN")
    parser.add_argument("--topic_num_shuffle", action='store_true', help="Variable number of topics")
    parser.add_argument('--weighted_loss', action='store_true', help="Add postfix in GEN")


    parser.add_argument('--cutoff', type=int, default=256)
    parser.add_argument('--passage_cutoff', type=int, default=50)
    parser.add_argument('--train_only_outputs', action='store_true', help="Disable to learn input")
    parser.add_argument('--train_only_inputs', action='store_true', help="Disable to learn input")

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', choices=['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf'])
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    parser.add_argument('--deepspeed', type=str, default='')

    # parser.add_argument('--prompt_template_name', type=str, default='D2P')

    # adpative item selection
    parser.add_argument('--item_selection', type=str, default='top', choices=['top', 'conf'], help='Adaptive item selection')
    parser.add_argument('--topic_conf', type=float, default=1.0, help='Minimum threshold for topic confidence')
    parser.add_argument('--topk_topic', type=int, default=1)
    parser.add_argument('--item_random_negative_num', type=int, default=3)
    parser.add_argument('--item_random_negative', action='store_true', help='random select negative sample')

    # For generation config
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--device', '--gpu', default='0', type=str, help='GPU Device')

    args = parser.parse_args()
    args.num_device = torch.cuda.device_count()

    # if 'I' in args.prompt.split('2')[-1]:
    #     args.task = 'topic'
    # elif 'P' in args.prompt.split('2')[-1]:
    #     args.task = 'know'
    # elif 'R' in args.prompt.split('2')[-1]:
    #     args.task = 'resp'
    # elif args.prompt == 'pretrain':
    #     args.task = 'pretrain'
    # else:
    #     raise ValueError

    return args
