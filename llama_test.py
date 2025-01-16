import json
import os
import sys

import numpy as np
import torch
# import wandb
from nltk.translate.bleu_score import sentence_bleu

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM
from peft import PeftModel

from utils.prompter import Prompter
from utils.utils import load_dataset

import re
import time
import datetime
from copy import deepcopy

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Textdataset(Dataset):
    def __init__(self, args, instructions, labels, topics, tokenizer):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.topics = topics
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.instructions[idx], self.labels[idx], self.topics[idx]

    def __len__(self):
        return len(self.instructions)


class LLaMaEvaluator:
    def __init__(self, args, tokenizer, insturctions, labels, topics, prompt_template_name: str = ""):
        self.args = args
        # self.dataset = dataset
        self.instructions = insturctions  # [i['context_tokens'] for i in dataset]
        self.labels = labels  # [i['item'] for i in dataset]
        # self.negItems = dataset['negItems']
        self.topics = topics  # [i['explanation'] for i in dataset]
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)

        # self.candidate_scores = candidate_scores
        self.prompter = Prompter(args, prompt_template_name)

        self.dataloader = self.prepare_dataloader()
        self.metric = {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0,
                       'sample_bleu1': 0, 'sample_bleu2': 0, 'sample_bleu3': 0, 'sample_bleu4': 0,
                       'dist1': set(), 'dist2': set(), 'dist3': set(), 'dist4': set(),
                       'hitgen': 0,
                       'hit1': 0, 'hit2': 0, 'hit3': 0, 'hit4': 0, 'hit5': 0,
                       'cnt': 0}
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def compute_hit(self, pred, label):
        for j, k in enumerate([1, 2, 3, 4, 5]):
            output = '| '.join(pred[:k])
            if label.strip().lower() in output.strip().lower():
                self.metric[f'hit{k}'] += 1
            # if f"Passage:{label[0]}" in output.strip().lower():
            #     self.metric[f'hit{k}'] += 1

    def compute_hitgen(self, pred, topic):
        if topic.lower() in pred.lower():
            self.metric[f'hitgen'] += 1

    def compute_bleu(self, pred, label):
        pred, label = pred.split(), [label.split()]
        for k in range(4):
            weights = [0] * 4
            weights[k] = 1
            self.metric[f'bleu{k + 1}'] += sentence_bleu(label, pred, weights)
            self.metric[f'sample_bleu{k + 1}'] = sentence_bleu(label, pred, weights)

    def print_score(self, outputs):
        prompt = self.args.prompt
        task = prompt.split('2')[-1]
        train_data, test_data = load_dataset(self.args)
        test_know_file_path = self.args.test_know_file
        test_know_file_path = os.path.join(self.args.home, f"data/know/en_test_know_{test_know_file_path}.json")
        test_know = json.load(open(test_know_file_path, 'r', encoding='utf-8'))
        # goalTopic_path = os.path.join(self.args.home)
        # goalTopicDic = json.load(open("/home/submission/junpyo/KEMGCRS/data/2/durecdial/DuRec_topicDic.json", 'r', encoding='utf-8'))
        # topicList = list(goalTopicDic['str'].keys())

        total = [(o,t) for o,t in zip(outputs, test_data) if t['topic']!='Q&A' and t['topic']!='Music recommendation']
        # result = [r for r,t in total if t['topic'].replace('  ',' ').replace('\xa0',' ').lower().strip() in r['GEN'].split('suitable topic is ')[-1].replace('  ',' ').replace('\xa0',' ').lower().strip()]
        if 'I' in task:
            if self.args.inspired:
                cnt = len([i for i in outputs if i['ANSWER'] in i['GEN'].split('Therefore')[-1]])
                score = cnt / len(test_data)
                print()
                print(f"Item hit ratio: hit@1")
                print(score)
                print()
            else:
                cnt = 0
                pattern = r'topic \d+\. '
                for r,t in total:
                    answer = t['topic'].replace('  ', ' ').replace('\xa0', ' ').lower().strip()
                    # gen = r['GEN'].split('topic is ')[-1].replace('  ',' ').replace('\xa0',' ').replace('"','').lower().strip()
                    gen = r['GEN'].split('topic is ')[-1].replace('  ', ' ').replace('\xa0', ' ').replace('"', '').lower().strip()
                    if re.search(pattern, gen):
                        gen = gen[re.search(pattern, gen).end():]
                        if answer == gen:
                            cnt += 1
                    else:
                        if answer in gen.split(' | ')[:1]:
                            cnt+=1

                score = cnt / len(total)
                print()
                print(f"Item hit ratio: hit@1")
                print(score)
                print()

        elif 'P' in task:
            for (i, j, x) in tqdm(zip(outputs, test_data, test_know)):
                i['response'] = j['response']
                i['goal'] = j['goal']
                i['topic'] = j['topic']
                i['predicted_topic'] = j['predicted_topic']
                i['GEN'] = i['GEN'].split("###Response:")[-1]
                i['selected_topic'] = j['selected_topic']
                i['predicted_topic_confidence'] = j['predicted_topic_confidence']
                predicted_know = deepcopy(x['predicted_know'][0])
                predicted_know = [idx for idx, y in enumerate(predicted_know) if y != '']
                predicted_know = predicted_know[:4]
                i['predicted_know'] = [x['predicted_know'][0][i] for i in predicted_know]  # x['predicted_know'][0] #
                i['target_knowledge'] = j['target_knowledge']
                i['candidate_knowledges'] = j['candidate_knowledges']
                i['CONTEXT'] += ("\n" + i['GEN'])

            predicted_know_list = []
            hits = [0, 0, 0, 0]
            for idx, data in tqdm(enumerate(outputs)):
                passages_results = data['GEN'].split('relevant passage is ')[-1].split('as follow:\n')[-1].split("\n")
                selected_passages_idx = [int(data['GEN'][m.start() + len('Passage ')]) - 1 for m in re.finditer('Passage', data['GEN'])]
                selected_passages = [data['predicted_know'][i] for i in selected_passages_idx]
                predicted_know_list.append({'predicted_know': selected_passages})
                reranked_passages = selected_passages + [i for i in data['predicted_know'][:4] if i not in selected_passages]
                # if len(reranked_passages) != 4:
                #     print('Check again')
                target_knowledge = data['target_knowledge']
                for topk in range(len(hits)):
                    if target_knowledge in reranked_passages[:topk + 1]:
                        hits[topk] += 1
            hits = ["%.4f" % (i / len(test_data)) for i in hits[:3]]
            score = '\t'.join(hits)
            print()
            print(f'Passage hit ratio: hit@1 | hit@2 | hit@3')
            print(score)
            print()

        elif 'R' in task:
            last = outputs[-1]
            score = last['bleu_scores']
            print()
            print(f"Generation bleu score: bleu@1 | bleu@2 | bleu@3 | bleu@4")
            print(score)
            print()
        else:
            print("Check prompt")
    def prepare_model(self,
                      base_model: str = "",
                      load_8bit: bool = False,
                      lora_weights: str = "",
                      server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                      share_gradio: bool = False, ):
        print('prepare new model for evaluating')
        base_model = self.args.base_model
        peft_weights = self.args.peft_weights

        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        if self.args.bf:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=dtype,  #
                # device_map='auto' # 이거 auto로 하니가 왜 인지 모르는데, 가끔식 GPU 할당이 이상하게 됌. 특정 GPU로 고정 할당하니까 문제 해결된 듯?
            ).to("cuda")  # MultiGPU 써보려고

            # todo: For evaluating the PEFT model
            if peft_weights != "":
                model = PeftModel.from_pretrained(
                    model,
                    peft_weights,
                    torch_dtype=dtype,
                )
        else:
            raise ValueError
            # model = LlamaForCausalLM.from_pretrained(
            #     base_model, device_map={"": device}, low_cpu_mem_usage=True
            # )
            # if self.args.lora_weights != "lora-alpaca":
            #     model = PeftModel.from_pretrained(
            #         model,
            #         lora_weights,
            #         device_map={"": device},
            #     )
        # unwind broken decapoda-research config

        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id

        model.config.bos_token_id = self.tokenizer.bos_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_eos_token = False
        # if not load_8bit and not self.args.sft:
        #     model.half()  # seems to fix bugs for some users. # bfloat16()
        #     # model.bfloat16()  # bf16로 학습시킨거면, 무조건 이거 써야 함... 근데 애초에 이 코드가 필요한 부분인가? 위에서 설정해주는데??

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'

        instruction_dataset = Textdataset(self.args, self.instructions, self.labels, self.topics, self.tokenizer)
        dataloader = DataLoader(instruction_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        return dataloader

    def evaluate(self,
                 input_ids,
                 attention_mask,
                 model,
                 input=None,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=1,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=128,
                 **kwargs):
        generation_config = GenerationConfig(
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
            # do_sample=True,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        # scores = generation_output.sequences_scores
        logits = generation_output.logits
        if num_beams > 1:
            sequences_scores = torch.softmax(generation_output.sequences_scores.view(-1, num_beams), dim=1).cpu().tolist()
        else:
            sequences_scores = [[0] for _ in range(input_ids.size(0))]
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        return [self.prompter.get_response(i) for i in output], sequences_scores  # , scores.to('cpu').numpy()

    def test(self, epoch=None):
        model = self.prepare_model()
        if epoch is not None:
            log_file = open(os.path.join(self.args.result_path, f'{self.args.log_name}_E{int(epoch)}.json'), 'a',
                            buffering=1, encoding='UTF-8')
            self.args.log_file = log_file
        # elif epoch is None:
        #     self.args.log_file = open(os.path.join(self.args.result_path, f'{self.args.log_name}.json'), 'a',
        #                               buffering=1, encoding='UTF-8')

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        # start_time = time.time()
        outputs = []
        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            batched_inputs = self.tokenizer(batch[0], padding=True, return_tensors="pt")
            input_ids = batched_inputs["input_ids"].to("cuda")
            attention_mask = batched_inputs["attention_mask"].to("cuda")
            batch_size = attention_mask.size(0)

            # if self.args.prompt == 'DGIP2P_cot':
            #     responses, logits = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens, num_beams=1)
            #     tokenized_response = self.tokenizer(responses, add_special_tokens=False).input_ids
            #
            #     rationales = [i.split(' "Passage')[0] + " \"Passage " for i in responses]
            #     tokenized_rationales = self.tokenizer(rationales, add_special_tokens=False, padding=True, return_tensors="pt").attention_mask
            #     tokenized_rationales_idx = torch.sum(tokenized_rationales, dim=-1)
            #
            #     output_list = self.tokenizer.convert_tokens_to_ids([str(idx + 1) for idx in range(self.args.n_sampled_negative)])
            #     output_list = torch.LongTensor(output_list).to('cuda')
            #
            #     logits_outputs = torch.stack(logits, dim=1)[torch.arange(batch_size).to('cuda'), tokenized_rationales_idx.to('cuda')]  # [B, V]
            #     logits_outputs = torch.nn.functional.softmax(logits_outputs, dim=-1)
            #     logits_outputs = logits_outputs[:, output_list].detach().tolist()  # [B, n_sample]
            #
            #     for i in range(batch_size):
            #         responses[i] = responses[i] + "\n\nProbs:" + '|'.join(["%.4f" % i for i in logits_outputs[i]])
            #
            # else:
            responses, sequences_scores = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens, num_beams=self.args.num_beams)

            responses = np.reshape(responses, (-1, self.args.num_beams)).tolist()  # [B, beam]

            dialogs, labels, topics = batch[0], batch[1], batch[2]

            for dialog, response, label, topic, scores in zip(batch[0], responses, labels, topics, sequences_scores):
                self.metric['cnt'] += 1
                self.compute_bleu(response[0], label)  # output type이 response일때
                self.compute_hitgen(response[0], topic)  # output type이 response일때
                self.compute_hit(response, label)  # output type이 topic or passage 일때

                bleu1 = self.metric['bleu1'] / self.metric['cnt']
                bleu2 = self.metric['bleu2'] / self.metric['cnt']
                bleu3 = self.metric['bleu3'] / self.metric['cnt']
                bleu4 = self.metric['bleu4'] / self.metric['cnt']

                sample_bleu1 = self.metric['sample_bleu1']
                sample_bleu2 = self.metric['sample_bleu2']
                sample_bleu3 = self.metric['sample_bleu3']
                sample_bleu4 = self.metric['sample_bleu4']

                hitgen = self.metric['hitgen'] / self.metric['cnt']

                hit1 = self.metric['hit1'] / self.metric['cnt']
                hit2 = self.metric['hit2'] / self.metric['cnt']
                hit3 = self.metric['hit3'] / self.metric['cnt']
                hit4 = self.metric['hit4'] / self.metric['cnt']
                hit5 = self.metric['hit5'] / self.metric['cnt']

                output = {'CONTEXT': dialog, 'GEN': ' | '.join(response), 'ANSWER': label,
                                    'hitgen': '%.4f' % hitgen,
                                    'hit_scores': '|'.join(['%.4f' % i for i in [hit1, hit2, hit3, hit4, hit5]]),
                                    'bleu_scores': '|'.join(['%.4f' % i for i in [bleu1, bleu2, bleu3, bleu4]]),
                                    'sample_bleu_scores': '|'.join(['%.4f' % i for i in [sample_bleu1, sample_bleu2, sample_bleu3, sample_bleu4]]),
                                    'contain': response[0].strip() in dialog.strip(),
                                    'llama_hit': label.strip() in response[0].strip(),
                                    'beam_scores': '|'.join(['%.4f' % i for i in scores]),
                                    'espresso_hit': label.strip() in dialog.strip()}
                outputs.append(output)
                if self.args.write or self.metric['cnt'] <= 100:
                    self.args.log_file.write(
                        json.dumps(output, ensure_ascii=False) + '\n')

        self.print_score(outputs)
        # end_time = time.time()
        # sec = (end_time - start_time)
        # result = str(datetime.timedelta(seconds=sec)).split(".")
        # json.dump(result,open(f"{self.args.log_name}_time_comp.json", "w", encoding='utf-8'))

        if not self.args.write:
            self.args.log_file.write(f'\n---Accuracy results for {self.args.log_name} at epoch {epoch}---\n')
            self.args.log_file.write(json.dumps({'hitgen': '%.4f' % hitgen,
                                                 'hit_scores': '|'.join(['%.4f' % i for i in [hit1, hit2, hit3, hit4, hit5]]),
                                                 'bleu_scores': '|'.join(
                                                     ['%.4f' % i for i in [bleu1, bleu2, bleu3, bleu4]]),
                                                 'sample_bleu_scores': '|'.join(
                                                     ['%.4f' % i for i in [sample_bleu1, sample_bleu2, sample_bleu3, sample_bleu4]])}) + '\n')
