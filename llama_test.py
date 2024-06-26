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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
                       'dist1': set(), 'dist2': set(), 'dist3': set(), 'dist4': set(),
                       'hitgen': 0,
                       'hit1': 0, 'hit3': 0, 'hit5': 0,
                       'cnt': 0}
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def compute_hit(self, pred, label):
        for j, k in enumerate([1, 3, 5]):
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
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        return [self.prompter.get_response(i) for i in output]  # , scores.to('cpu').numpy()

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

        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            batched_inputs = self.tokenizer(batch[0], padding=True, return_tensors="pt")
            input_ids = batched_inputs["input_ids"].to("cuda")
            attention_mask = batched_inputs["attention_mask"].to("cuda")

            if self.args.prompt == 'DGIP2P_cot':
                responses = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens, num_beams=1)
                rationales = [i.split(' "Passage')[0] for i in responses]

                batched_inputs_with_rationale = [i + j + " \"Passage " for i, j in zip(batch[0], rationales)]  # [B, L]
                batched_inputs_with_rationale = self.tokenizer(batched_inputs_with_rationale, padding=True, return_tensors="pt")
                input_ids = batched_inputs_with_rationale["input_ids"].to("cuda")
                attention_mask = batched_inputs_with_rationale["attention_mask"].to("cuda")

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)

                output_list = self.tokenizer.convert_tokens_to_ids([str(idx + 1) for idx in range(self.args.n_sampled_negative)])
                output_list = torch.LongTensor(output_list).to("cuda")
                probs_output_list = probs[:, output_list]

                # responses = self.evaluate(input_ids_with_rationale, attention_mask_with_rationale, model, max_new_tokens=self.args.max_new_tokens, num_beams=1)

            else:
                responses = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens, num_beams=self.args.num_beams)

            responses = np.reshape(responses, (-1, self.args.num_beams)).tolist()  # [B, beam]

            dialogs, labels, topics = batch[0], batch[1], batch[2]

            for dialog, response, label, topic in zip(batch[0], responses, labels, topics):
                self.metric['cnt'] += 1
                self.compute_bleu(response[0], label)  # output type이 response일때
                self.compute_hitgen(response[0], topic)  # output type이 response일때
                self.compute_hit(response, label)  # output type이 topic or passage 일때

                bleu1 = self.metric['bleu1'] / self.metric['cnt']
                bleu2 = self.metric['bleu2'] / self.metric['cnt']
                bleu3 = self.metric['bleu3'] / self.metric['cnt']
                bleu4 = self.metric['bleu4'] / self.metric['cnt']

                hitgen = self.metric['hitgen'] / self.metric['cnt']

                hit1 = self.metric['hit1'] / self.metric['cnt']
                hit3 = self.metric['hit3'] / self.metric['cnt']
                hit5 = self.metric['hit5'] / self.metric['cnt']

                if self.args.write or self.metric['cnt'] <= 100:
                    self.args.log_file.write(
                        json.dumps({'CONTEXT': dialog, 'GEN': ' | '.join(response), 'ANSWER': label,
                                    'hitgen': '%.4f' % hitgen,
                                    'hit_scores': '|'.join(['%.4f' % i for i in [hit1, hit3, hit5]]),
                                    'bleu_scores': '|'.join(['%.4f' % i for i in [bleu1, bleu2, bleu3, bleu4]]),
                                    'contain': response[0].strip() in dialog.strip(),
                                    'llama_hit': label.strip() in response[0].strip(),
                                    'espresso_hit': label.strip() in dialog.strip()}, ensure_ascii=False) + '\n')

        if not self.args.write:
            self.args.log_file.write(f'\n---Accuracy results for {self.args.log_name} at epoch {epoch}---\n')
            self.args.log_file.write(json.dumps({'hitgen': '%.4f' % hitgen,
                                                 'hit_scores': '|'.join(['%.4f' % i for i in [hit1, hit3, hit5]]),
                                                 'bleu_scores': '|'.join(
                                                     ['%.4f' % i for i in [bleu1, bleu2, bleu3, bleu4]])}) + '\n')
