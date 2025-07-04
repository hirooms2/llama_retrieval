### inspired2 dataset

## I-SEL
# train
deepspeed main.py --mode=train --log_name=inspired2_item_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2I_cot --train_know_file=inspired2_isel --test_know_file=inspired2_isel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --combined --shuffle --inspired --force_gpt --cutoff=128;
# test
python main.py --mode=test --log_name=inspired2_item_selection_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2I_cot --train_know_file=inspired2_isel --test_know_file=inspired2_isel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --combined --shuffle --inspired --force_gpt --cutoff=128 --peft_weights=; # Add your peft_weights

## F-SEL
# train
deepspeed main.py --mode=train --log_name=inspired2_feature_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2P_cot --force_topic --train_know_file=inspired2_final --test_know_file=inspired2_fsel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --inspired --selected_topic --candidate_knowledges_gpt;
# test
python main.py --mode=test --log_name=inspired2_feature_selection_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DIP2P_cot --force_gpt --force_topic --train_know_file=inspired2_final --test_know_file=inspired2_fsel --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --inspired --selected_topic --candidate_knowledges_gpt --peft_weights=; # Add your peft_weights

## Response Generation
# train
deepspeed main.py --mode=train --log_name=inspired2_response_generation --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DP2R_inspired --force_split --train_know_file=inspired2_final --test_know_file=inspired2_rgen --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --inspired --selected_topic --force_topic --force_gpt --cutoff=128;
# test
python main.py --mode=test --log_name=inspired2_response_generation_test --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DP2R_inspired --force_split --train_know_file=inspired2_final --test_know_file=inspired2_rgen --train_data=inspired2_final --test_data=inspired2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --inspired --selected_topic --force_topic --force_gpt --cutoff=128 --peft_weights=; # Add your peft_weights


### DuRecDial2 dataset

## I-SEL
# train
deepspeed main.py --mode=train --log_name=durecdial2_item_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=UDGIP2I_cot --train_know_file=durec2_isel --test_know_file=durec2_isel --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --combined --shuffle;
# test
python main.py --mode=test --log_name=durec2_Item_selection_T1_E1 --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=UDGIP2I_cot --train_know_file=durec2_isel --test_know_file=durec2_isel --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=3 --topic_conf=0.7 --combined --shuffle --peft_weights=; # Add your peft_weights

## F-SEL
# train
deepspeed main.py --mode=train --log_name=durec2_feature_selection --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DGIP2P_cot_new --train_know_file=durec2_final --test_know_file=durec2_fsel --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_sampled_positive=3 --n_sampled_negative=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --selected_topic --candidate_knowledges_gpt;
# test
python main.py --mode=test --log_name=durec2_feature_selection_T1_E1 --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=1 --prompt=DGIP2P_cot_new --train_know_file=durec2_final --test_know_file=durec2_fsel --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=4 --n_sampled_negative=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1 --shuffle --selected_topic --candidate_knowledges_gpt --n_sampled_positive=4 --peft_weights=; # Add your peft_weights

## Response Generation
# train
deepspeed main.py --mode=train --log_name=durec2_response_generation --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DGP2R --train_know_file=durec2_final --test_know_file=durec2_rgen --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1  --selected_topic;
# test
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=durec2_response_generation_T1_E1 --base_model=meta-llama/Llama-2-7b-chat-hf --num_beams=4 --prompt=DGP2R --train_know_file=durec2_final --test_know_file=durec2_rgen --train_data=durec2_final --test_data=durec2_final --eval_batch_size=4 --n_pseudo=1  --batch_size=16 --epoch=5 --positive=gpt_selection --n_hard_negative=0 --n_sampled_negative=4 --n_sampled_positive=4 --n_docs=4 --learning_rate=4e-4 --deepspeed=deepspeed.json --topk_topic=1  --selected_topic --peft_weights=; # Add your peft_weights