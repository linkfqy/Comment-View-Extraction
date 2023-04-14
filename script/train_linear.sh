export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --warmup_steps 2000 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/bert-base-chinese \
    --task_type train \
    --train_file ./data/train_splited.csv \
    --dev_file ./data/dev_splited.csv \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name linear \
    # --debug \