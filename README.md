# Leveraging Retrieval-Augmented Language Models for Accurate Item/Feature Selection in Conversational Recommender Systems

This repository provides a reference implementation of MOCHA as described in the following paper:
```
- Leveraging Retrieval-Augmented Language Models for Accurate Item/Feature Selection in Conversational Recommender Systems 
- Taeho Kim, Junpyo Kim, Won-Yong Shin, Sang-Wook Kim.
- Conference name ( )
```

## Authors
- Taeho Kim (hirooms2@hanyang.ac.kr)
- Junpyo Kim (pyo9912@hanyang.ac.kr)
- Won-Yong Shin (wy.shin@yonsei.ac.kr)
- Sang-Wook Kim (wook@hanyang.ac.kr)
  
## Requirements
The code has been tested running under Python 3.8.19  
You can import the conda environment from the provided `environment.yaml`
```
conda create --file environment.yaml
conda activate mocha
```

## Dataset
You can download dataset from the [link](https://drive.google.com/drive/folders/1HjDFGiEK4sUlyaPwj2l8C0TAYcsvW6PT?usp=sharing), which include recommendation and conversation of **Inspired2** and **DurecDial2**. Please put the downloaded dataset into data directory.


## Basic Usage
The run commands are provided in `run.sh` 

Item selection commands
```
## train
deepspeed main.py --mode=train --log_name=inspired2_item_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2I_cot --train_know_file=inspired2_isel --test_know_file=inspired2_isel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --combined --shuffle --inspired --force_gpt --cutoff=128;

## test
python main.py --mode=test --log_name=inspired2_item_selection_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2I_cot --train_know_file=inspired2_isel --test_know_file=inspired2_isel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --combined --shuffle --inspired --force_gpt --cutoff=128 --peft_weights=; # Add your peft_weights
```
Feature selection commands
```
## train
deepspeed main.py --mode=train --log_name=inspired2_feature_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2P_cot --force_gpt --force_topic --train_know_file=inspired2_final --test_know_file=inspired2_fsel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --selected_topic --candidate_knowledges_gpt;
## test
python main.py --mode=test --log_name=inspired2_feature_selection_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2P_cot --force_gpt --force_topic --train_know_file=inspired2_final --test_know_file=inspired2_fsel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --selected_topic --candidate_knowledges_gpt --peft_weights=; # Add your peft_weights
```
Response generation commands
```
## train
deepspeed main.py --mode=train --log_name=inspired2_response_generation --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DP2R_inspired --force_split --train_know_file=inspired2_final --test_know_file=inspired2_rgen --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --inspired --selected_topic --force_topic --force_gpt --cutoff=128;
## test
python main.py --mode=test --log_name=inspired2_response_generation_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DP2R_inspired --force_split --train_know_file=inspired2_final --test_know_file=inspired2_rgen --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --inspired --selected_topic --force_topic --force_gpt --cutoff=128 --peft_weights=; # Add your peft_weights
```

## Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
add proceedings
```