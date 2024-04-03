#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기


# 240328 실험 세팅 JP
# ## candidate list에 top1과 random sample을 넣는 variation
# # CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --log_name=llama_pretest_dp2r_random_ndocs_5 --prompt=DP2R --train_know_file=espresso --test_know_file=random --batch_size=16 --eval_batch_size=4 --n_docs=5
# CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --log_name=llama_pretest_dp2r_random_ndocs_3 --prompt=DP2R --train_know_file=espresso --test_know_file=random --batch_size=16 --eval_batch_size=4 --n_docs=3
# CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --log_name=llama_pretest_dp2r_random_ndocs_1,--prompt=DP2R --train_know_file=espresso --test_know_file=random --batch_size=16 --eval_batch_size=4 --n_docs=1

# ## candidate list에 top1과 random sample same topic을 넣는 variation
# # CUDA_VISIBLE_DEVICES=1 python main.py --mode=train --log_name=llama_pretest_dp2r_random_sametopic_ndocs_5 --prompt=DP2R --train_know_file=espresso --test_know_file=random_sametopic --batch_size=16 --eval_batch_size=4 --n_docs=5
# CUDA_VISIBLE_DEVICES=1 python main.py --mode=train --log_name=llama_pretest_dp2r_random_sametopic_ndocs_3 --prompt=DP2R --train_know_file=espresso --test_know_file=random_sametopic --batch_size=16 --eval_batch_size=4 --n_docs=3
# CUDA_VISIBLE_DEVICES=1 python main.py --mode=train --log_name=llama_pretest_dp2r_random_sametopic_ndocs_1 --prompt=DP2R --train_know_file=espresso --test_know_file=random_sametopic --batch_size=16 --eval_batch_size=4 --n_docs=1


## Train_test 한번에
## TODO
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_new_E3_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035841_llama_pretest_dp2r_random_new_ndocs=3_E3; 
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_sametopic_new_E3_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035854_llama_pretest_dp2r_random_sametopic_new_ndocs=3_E3;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_new_E4_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035841_llama_pretest_dp2r_random_new_ndocs=3_E4;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_sametopic_new_E4_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035854_llama_pretest_dp2r_random_sametopic_new_ndocs=3_E4;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_new_E5_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035841_llama_pretest_dp2r_random_new_ndocs=3_E5;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_sametopic_new_E5_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035854_llama_pretest_dp2r_random_sametopic_new_ndocs=3_E5;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_new_E6_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035841_llama_pretest_dp2r_random_new_ndocs=3_E6;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_sametopic_new_E6_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035854_llama_pretest_dp2r_random_sametopic_new_ndocs=3_E6;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_new_E7_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035841_llama_pretest_dp2r_random_new_ndocs=3_E7;
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --log_name=llama_pretest_dp2r_random_sametopic_new_E7_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0403035854_llama_pretest_dp2r_random_sametopic_new_ndocs=3_E7



CUDA_VISIBLE_DEVICES=1 python main.py --mode=train_test --log_name=llama_pretest_dp2r_random_new_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0329214427_llama_pretrain_E5 --epoch=7
CUDA_VISIBLE_DEVICES=1 python main.py --mode=train_test --log_name=llama_pretest_dp2r_random_sametopic_new_ndocs=3 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=3 --peft_weights=0329214427_llama_pretrain_E5 --epoch=7


CUDA_VISIBLE_DEVICES=0 python main.py --mode=train_test --log_name=llama_pretest_dp2r_random_sft_kbit_ndocs=1 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random --batch_size=16 --eval_batch_size=4 --n_docs=1 --sft=True --peft_weights=0329214427_llama_pretrain_E5
CUDA_VISIBLE_DEVICES=1 python main.py --mode=train_test --log_name=llama_pretest_dp2r_random_sametopic_ndocs=1 --prompt=DP2R_new --train_know_file=espresso --test_know_file=random_sametopic --batch_size=32 --eval_batch_size=4 --n_docs=1 --peft_weights=0329214427_llama_pretrain_E5

# 240328 실험 세팅 JP
### Test
## candidate list에 top1과 random sample을 넣는 variation
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --debug=True --log_name=llama_pretest_dp2r_random_ndocs_1_Debug --prompt=DP2R --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=1 --peft_weights=0328202624_llama_pretest_dp2r_random_ndocs_1_E5
CUDA_VISIBLE_DEVICES=1 python main.py --mode=test --debug=True --log_name=llama_pretest_dp2r_random_sft_ndocs_1_Debug --prompt=DP2R --train_know_file=espresso --test_know_file=random --batch_size=32 --eval_batch_size=4 --n_docs=1 --peft_weights=0403144349_llama_pretest_dp2r_random_sft_ndocs=1_E1

