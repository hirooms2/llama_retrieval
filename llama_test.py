import json
import os
import sys

import numpy as np
import torch
# import wandb
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from llama_util import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Textdataset(Dataset):
    def __init__(self, args, instructions, labels, tokenizer):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # tokenizer.padding_side = 'left'
        # inputs = self.tokenizer(self.data_samples[idx], padding=True, return_tensors="pt", max_length=args.max_input_length, truncation=True)
        # input_ids = inputs["input_ids"].to(self.args.device_id)
        return self.instructions[idx], self.labels[idx]

    def __len__(self):
        return len(self.instructions)


class LLaMaEvaluator:
    def __init__(self, args, tokenizer, insturctions, labels, explanation=[], prompt_template_name: str = ""):
        self.args = args
        # self.dataset = dataset
        self.instructions = insturctions  # [i['context_tokens'] for i in dataset]
        self.labels = labels  # [i['item'] for i in dataset]
        # self.negItems = dataset['negItems']
        self.explanations = explanation  # [i['explanation'] for i in dataset]
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)

        # self.candidate_scores = candidate_scores
        self.prompter = Prompter(args, prompt_template_name)

        self.dataloader = self.prepare_dataloader()
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

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

        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                # device_map='auto' # 이거 auto로 하니가 왜 인지 모르는데, 가끔식 GPU 할당이 이상하게 됌. 특정 GPU로 고정 할당하니까 문제 해결된 듯?
            ).to("cuda")

            # todo: For evaluating the PEFT model
            if self.args.peft_weights != "":
                model = PeftModel.from_pretrained(
                    model,
                    peft_weights,
                    torch_dtype=torch.float16,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            if self.args.lora_weights != "lora-alpaca":
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": device},
                )
        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'

        instructions = self.instructions  # [self.prompter.generate_prompt(instruction=instruction) for instruction in self.instructions]
        labels = self.labels
        instruction_dataset = Textdataset(self.args, instructions, labels, self.tokenizer)
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
                 num_beams=5,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=50,
                 **kwargs):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
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

    def test(self, model=None, epoch=None):
        if model is None:
            model = self.prepare_model()
        if epoch is not None:
            log_file = open(os.path.join(self.args.result_path, f'{self.args.log_name}_E{int(epoch)}.json'), 'a',
                            buffering=1, encoding='UTF-8')
            self.args.log_file = log_file
        elif epoch is None:
            self.args.log_file = open(os.path.join(self.args.result_path, f'{self.args.log_name}.json'), 'a',
                                      buffering=1, encoding='UTF-8')

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        hit, mentioned_hit, not_mentioned_hit, cnt, mentioned_cnt, not_mentioned_cnt, gen_mentioned_cnt, gen_not_mentioned_cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        hits, cnts = [0, 0, 0], [0, 0, 0]
        idx = 0
        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            generated_results = []
            batched_inputs = self.tokenizer(batch[0], padding=True, return_tensors="pt")
            # input_ids = batched_inputs["input_ids"].to(self.args.device_id)
            # attention_mask = batched_inputs["attention_mask"].to(self.args.device_id)
            input_ids = batched_inputs["input_ids"].to("cuda")
            attention_mask = batched_inputs["attention_mask"].to("cuda")

            responses = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens,
                                      num_beams=self.args.num_beams)
            responses = np.reshape(responses, (-1, self.args.num_beams)).tolist()  # [B, beam]
            # scores = np.reshape(scores, (-1, self.args.num_beams)).tolist()  # [B, beam]

            dialogs, labels = batch[0], batch[1]

            for dialog, response, label in zip(batch[0], responses, labels):
                topk_results = []
                for j, k in enumerate([1, 3, 5]):
                    output = '| '.join(response[:k])
                    if label.lower() in output.lower():
                        # if title == gen_title and year == gen_year:
                        hits[j] += 1.0
                    cnts[j] += 1.0
                    hit_ratio = (hits[j] / cnts[j]) * 100
                    topk_results.append('%.2f' % hit_ratio)

                generated_results.append(
                    {'CONTEXT': dialog, 'GEN': output, 'ANSWER': label, 'AVG_HIT': ', '.join(topk_results)})

            for i in generated_results:
                self.args.log_file.write(json.dumps(i, ensure_ascii=False) + '\n')

            if cnt % 100 == 0 and cnt != 0:
                wandb.log({"hit_ratio": (hit / cnt)})
                print("%.4f" % (hit / cnt))
