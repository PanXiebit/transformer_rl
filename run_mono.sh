#!bin/bash


export CUDA_VISIBLE_DEVICES="0,1,2,3"
export LD_LIBRARY_PATH=/usr/local/nccl_2.3.4/lib:$LD_LIBRARY_PATH

# sharing embedding, v1

python3 main_mono.py \
    --debug false \
    --num_gpus 4 \
    --roll_num 5 \
    --param_set base \
    --data_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v1/gen_data \
    --data_dir_mono /home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v1_mono/gen_data \
    --model_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl_sess/model_save/en-tr/share/base/train_mono \
    --pretrain_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl_sess/model_save/en-tr/share/base/train_base \
    --learning_rate 0.5 \
    --learning_rate_bt 1.0 \
    --batch_size 4900 \
    --max_length 50 \
    --fro src \
    --to tgt \
    --save_checkpoints_secs 20000 \
    --shared_embedding_softmax_weights true \
    --train_steps 250000 \
    --steps_between_evals 1000

# not sharing embedding, v0
#python3 transformer_gan_main.py \
#    --debug false \
#    --num_gpus 2 \
#    --param_set base \
#    --data_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v0/gen_data \
#    --model_dir /home/work/xiepan/xp_dial/gan_nmt/transformer_teach_force/model_save/en-tr \
#    --learning_rate 0.1 \
#    --batch_size 100 \
#    --max_length 50 \
#    --fro src \
#    --to tgt \
#    --save_checkpoints_secs 1200 \
#    --train_steps 250000 \
#    --steps_between_evals 2000
