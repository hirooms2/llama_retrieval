from copy import deepcopy
from datetime import datetime
import json
import pickle
import os
from loguru import logger
from pytz import timezone
from tqdm import tqdm


def load_dataset(args):
    print('LLAMA_DATASET')
    train_file_path = os.path.join(args.home, 'data/train_pred_aug_dataset_new.pkl')
    with open(file=train_file_path, mode='rb') as f:
        train_dataset = pickle.load(f)

    test_file_path = os.path.join(args.home, 'data/test_pred_aug_dataset_new.pkl')
    with open(file=test_file_path, mode='rb') as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset


def augment_dataset(args, know_dataset, labels, topics):
    new_know_dataset, new_labels, new_topics = [], [], []
    for i, j, k in zip(know_dataset, labels, topics):
        if i['topic'] in i['predicted_topic'][:args.topk_topic]:
            if i['topic'].replace('\xa0', ' ').strip().lower() == i['topic_cot_long'].split('Therefore')[-1].lower().strip():
                new_know_dataset.append(i)
                new_labels.append(j)
                new_topics.append(k)
        # else:
        # if i['gpt_selection'] != '' and i['topic'] in i['predicted_topic'][:args.topk_topic]:
        #     i['candidate_knowledges'] = [j for j in i['candidate_knowledges'] if j != i['gpt_selection']]
        #     new_know_dataset.append(i)
        #     new_labels.append(j)
        #     new_topics.append(k)
    return new_know_dataset, new_labels, new_topics


def merge_dataset_passages(args, dataset, mode='train', know_file_path='', combined=False):
    if mode == 'train' and know_file_path == '':
        know_file_path = args.train_know_file
    elif mode == 'test' and know_file_path == '':
        know_file_path = args.test_know_file

    know_file_path = os.path.join(args.home, f'data/know/en_{mode}_know_{know_file_path}.json')
    know_dataset = json.load(open(know_file_path, 'r', encoding='utf-8'))

    ## Duplicate the raw dataset in case the size of the knowledge dataset is larger.
    if len(dataset) != len(know_dataset):
        dataset = [data for data in dataset for _ in range(int(len(know_dataset) / len(dataset)))]

    for idx, know_data in enumerate(know_dataset):
        dataset[idx]['predicted_know'] = know_data['predicted_know']
        dataset[idx]['combined'] = combined

    return deepcopy(dataset)


def prepare_dataset(args, tokenizer, dataset):
    labels, topics = [], []

    if args.debug:
        dataset = dataset[:50] + dataset[-50:]

    if args.test_continue != 0:
        dataset = dataset[args.test_continue:]

    for data in tqdm(dataset):
        dialog = data['dialog'].replace('[SEP]', '\n')[:-1]
        dialog = tokenizer.decode(tokenizer(dialog).input_ids[1:][-args.cutoff:])
        data['dialog'] = dialog
        # ['accepted food', 'accepted music', 'accepted movies', 'accepted celebrities', ]

        user_profile = data['user_profile']
        filtered_user_profile = []
        for profile in user_profile.split('|'):
            if 'accepted' in profile.lower() or 'rejected' in profile.lower():
                filtered_user_profile.append(profile.strip())
        user_profile = " | ".join(filtered_user_profile).strip()
        user_profile = tokenizer.decode(tokenizer(user_profile).input_ids[1:][:200])
        data['user_profile'] = user_profile

        data['response'] = data['response'].replace('[SEP]', '')
        topics.append(data['topic'])

        for idx1, top_passages in enumerate(data['predicted_know']):
            for idx2, passage in enumerate(top_passages):
                data['predicted_know'][idx1][idx2] = tokenizer.decode(
                    tokenizer(passage).input_ids[1:][:args.passage_cutoff]).strip()

        # if 'I' in args.prompt.split('2')[-1]:
        #     labels.append(data['topic'])
        if 'R' in args.prompt.split('2')[-1]:
            labels.append(data['response'].replace('\xa0', ' ').strip())
        elif 'P' in args.prompt.split('2')[-1]:
            labels.append(
                tokenizer.decode(tokenizer(data['target_knowledge']).input_ids[1:][:args.passage_cutoff]).strip().replace('\xa0', ' ').strip())
        elif 'I' in args.prompt.split('2')[-1]:
            labels.append(data['topic'].replace('\xa0', ' ').strip())
        elif args.prompt == 'pretrain':
            labels.append(data['response'].replace('\xa0', ' ').strip())
        else:
            raise ValueError

    return dataset, labels, topics


def dir_init(default_args):
    from copy import deepcopy
    """ args 받은다음, device, Home directory, data_dir, log_dir, output_dir, 들 지정하고, Path들 체크해서  """
    args = deepcopy(default_args)
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(os.path.dirname(__file__))
        print(args.home)
    elif sysChecker() == "Windows":
        args.home = ''
        # args.batch_size, args.num_epochs = 4, 2
        # args.debug = True
        pass  # HJ local
    else:
        raise Exception("Check Your Platform Setting (Linux-Server or Windows)")

    # Check path
    return args


def checkPath(*args) -> None:
    for path in args:
        if not os.path.exists(path): os.makedirs(path)


def createLogFile(args):
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    md = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d'))

    if args.log_name == '':
        if args.peft_weights != '' and args.mode == 'test':
            log_name = args.peft_weights
        else:
            log_name = 'llama_result'
    else:
        log_name = args.log_name
    args.log_name = mdhm + '_' + log_name
    args.log_dir = os.path.join(args.home, 'logs')
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

    args.output_dir = os.path.join(args.home, 'result')
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    result_path = os.path.join(args.output_dir, args.base_model.replace('/', '-'))
    if not os.path.exists(result_path): os.mkdir(result_path)

    result_path = os.path.join(result_path, md)
    args.result_path = result_path  # os.path.join(args.home, result_path)
    if not os.path.exists(args.result_path): os.mkdir(args.result_path)

    log_file = open(os.path.join(args.result_path, args.log_name + ".json"), 'a', buffering=1, encoding='UTF-8')
    args.log_file = log_file

    saved_model_path = os.path.join(args.home, 'saved_model')
    args.saved_model_path = saved_model_path

    checkPath(args.saved_model_path)
    if args.peft_weights != '':
        args.peft_weights = os.path.join(saved_model_path, args.peft_weights)

    return args


def cutoffInstruction(tokenizer, instructions, length, reverse=False):
    new_instructions = []
    for data in tqdm(instructions):
        if reverse:
            data = tokenizer.decode(tokenizer(data).input_ids[1:][-length:])
        else:
            data = tokenizer.decode(tokenizer(data).input_ids[1:][:length])
        new_instructions.append(data)
    logger.info('[Finish Cutting-off the instructions]')
    return new_instructions
