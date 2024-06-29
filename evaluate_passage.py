import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import re

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
train_raw_data = pickle.load(open("data/train_pred_aug_dataset_new.pkl", 'rb'))
test_raw_data = pickle.load(open("data/test_pred_aug_dataset_new.pkl", 'rb'))

results1 = json.load(open('result/0629091953_llama2_DGIP2P_cotnew_deepspeed_4e-4_nsampled4_rel_clean_npos3_E2.json', 'r', encoding='utf-8'))

en_test_know_combined3 = json.load(open('data/know/en_test_know_combined3.json', 'r', encoding='utf-8'))

for (i, j, x) in tqdm(zip(results1, test_raw_data, en_test_know_combined3)):
    i['response'] = j['response']
    # if i['response'] == "System: It's suitable for eating Steamed Chicken with Chili Sauce in such weather.[SEP]":
    #     print('fuck')
    i['goal'] = j['goal']
    i['topic'] = j['topic']
    i['predicted_topic'] = j['predicted_topic']
    i['GEN'] = i['GEN'].split("###Response:")[-1]
    i['selected_topic'] = j['selected_topic']
    i['predicted_topic_confidence'] = j['predicted_topic_confidence']

    predicted_know = deepcopy(x['predicted_know'][0])

    for idx, psg in enumerate(predicted_know):
        predicted_know[idx] = tokenizer.decode(tokenizer(psg).input_ids[1:][:50]).strip()

    # predicted_know = [y for y in predicted_know if y != '']
    predicted_know_filtered = [idx for idx, y in enumerate(predicted_know) if j['selected_topic'].replace('\xa0', ' ').strip().lower() in y.replace('\xa0', ' ').strip().lower()]
    predicted_know_unfiltered = [idx for idx, y in enumerate(predicted_know) if j['selected_topic'].replace('\xa0', ' ').strip().lower() not in y.replace('\xa0', ' ').strip().lower() and y != '']
    if len(predicted_know_filtered) < 4:
        predicted_know_filtered = predicted_know_filtered + predicted_know_unfiltered[:4 - len(predicted_know_filtered)]
    predicted_know = predicted_know_filtered[:4]

    i['predicted_know'] = [x['predicted_know'][0][i] for i in predicted_know]  # x['predicted_know'][0] #

    i['target_knowledge'] = j['target_knowledge']
    i['candidate_knowledges'] = j['candidate_knowledges']
    i['CONTEXT'] += ("\n" + i['GEN'])

predicted_know_list = []
print()
hits = [0, 0, 0, 0]
for idx, (data, raw) in enumerate(zip(results1, train_raw_data)):
    passages_results = data['GEN'].split('relevant passage is ')[-1].split('as follow:\n')[-1].split("\n")
    selected_passages_idx = [int(data['GEN'][m.start() + len('Passage ')]) - 1 for m in re.finditer('Passage', data['GEN'])]
    selected_passages = [data['predicted_know'][i] for i in selected_passages_idx]
    predicted_know_list.append({'predicted_know': selected_passages})

    reranked_passages = selected_passages + [i for i in data['predicted_know'][:4] if i not in selected_passages]
    if len(reranked_passages) != 4:
        print('fuck')
    target_knowledge = data['target_knowledge']
    for topk in range(len(hits)):
        if target_knowledge in reranked_passages[:topk + 1]:
            hits[topk] += 1

hits = ["%.4f" % (i / 3711) for i in hits[:3]]
print('\t'.join(hits))
