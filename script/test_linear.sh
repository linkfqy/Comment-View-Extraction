export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --warmup_steps 2000 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/chinese-roberta-wwm-ext-large \
    --task_type test \
    --test_file ./data/test_public.csv \
    --checkpoint ./save/linear_epoch15_loss0.006604_score0.671217_20230418-215427.pt \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name linear_nodev2 \
    # --debug \