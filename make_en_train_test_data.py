import pickle
import json
import random
from tqdm import tqdm 
from utils import *
import os
import platform


pseudo_train_name = 'en_train_pseudo_BySamples3711.txt'
pseudo_train_path = os.path.join(dataPath(), pseudo_train_name)
pseudo_train_context = read_json(pseudo_train_path)

pseudo_test_name = 'en_test_pseudo_BySamples3711.txt'
pseudo_test_path = os.path.join(dataPath(), pseudo_test_name)
pseudo_test_context = read_json(pseudo_test_path)

train_raw_pkl_file_name = 'data/train_pred_aug_dataset.pkl'
train_raw_pkl_path = os.path.join(dataPath(), train_raw_pkl_file_name)
train_raw = read_pkl(train_raw_pkl_path)


test_raw_pkl_file_name = 'data/test_pred_aug_dataset.pkl'
test_raw_pkl_path = os.path.join(dataPath(), test_raw_pkl_file_name)
test_raw = read_pkl(test_raw_pkl_path)


train_txt_file_name = 'en_train_know_3711.txt'
train_txt_path = os.path.join(dataPath(), train_txt_file_name)
train = read_json(train_txt_path)


pred_txt_file_name = 'en_test_know_3711.txt'
pred_txt_path = os.path.join(dataPath(), pred_txt_file_name)
pred = read_json(pred_txt_path)


test_json_file_name = 'all_knowledgeDB.json'
test_json_path = os.path.join(dataPath(), test_json_file_name)
test_know = read_json(test_json_path)

train_json_file_name = 'train_knowledgeDB.json'
train_json_path = os.path.join(dataPath(), train_json_file_name)
train_know = read_json(train_json_path)

new_pred_list = []

# for i, j in zip(test_raw, pred):
#     target_knowledge = i['target_knowledge']
#     candidates = [target_knowledge]
#     while len(candidates) < 5:
#         selected = random.choice(test_know)
#         if selected not in candidates:
#             candidates.append(selected)
#     new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

# result = "en_test_know_3711_random.txt"
# result_path = os.path.join(resultPath(), result)
# f = open(result_path, 'w+', encoding='utf-8')

# for i in new_pred_list: # range(len(new_pred_list)):
#     f.write(json.dumps(i, ensure_ascii=False) + '\n')
# f.close()


# for i, j in tqdm(zip(test_raw, pred)):
#     target_knowledge = i['target_knowledge']
#     candidates = [target_knowledge]
#     topic = i['topic']
#     topic_related_know = [i for i in test_know if topic.lower() in i.lower()]

#     while len(candidates) < min(5, len(topic_related_know)):
#         selected = random.choice(topic_related_know)
#         if selected not in candidates:
#             candidates.append(selected)

#     while len(candidates) < 5:
#         selected = random.choice(test_know)
#         if selected not in candidates:
#             candidates.append(selected)
#     new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

# result = "en_test_know_3711_random_sametopic.txt"
# result_path = os.path.join(resultPath(), result)
# f = open(result_path, 'w+', encoding='utf-8')

# for i in new_pred_list: # range(len(new_pred_list)):
#     f.write(json.dumps(i, ensure_ascii=False) + '\n')
# f.close()

n_pseudo = 2
for i, j in zip(train_raw, pseudo_train_context):
    # candidates = [i['target_knowledge']]
    # hard_negative_candidates = j['candidate_knowledges'][n_pseudo:20]
    # hard_negative_candidates = [x for x in hard_negative_candidates if x != i['target_knowledge']]
    # hard_negative = random.choice(hard_negative_candidates)
    # candidates.append(hard_negative)
    # while len(candidates) < 5:
    #     selected = random.choice(train_know)
    #     if selected not in candidates:
    #         candidates.append(selected)
    # new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

    for target_knowledge in j['candidate_knowledges'][:n_pseudo]:
        candidates = [target_knowledge]
        hard_negative_candidates = j['candidate_knowledges'][n_pseudo:20]
        hard_negative = random.choice(hard_negative_candidates)
        candidates.append(hard_negative)
        while len(candidates) < 5:
            selected = random.choice(train_know)
            if selected not in candidates:
                candidates.append(selected)
        new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

result = f"en_train_pseudo_{n_pseudo}_hard_1_random_3.json"
result_path = os.path.join(resultPath(), result)
f = open(result_path, 'w+', encoding='utf-8')
# f.write(json.dumps(new_pred_list, ensure_ascii=False))

for i in new_pred_list: # range(len(new_pred_list)):
    f.write(json.dumps(i, ensure_ascii=False) + '\n')
f.close()


