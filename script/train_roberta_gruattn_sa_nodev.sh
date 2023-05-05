export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --warmup_steps_rate 0.05 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/chinese-roberta-wwm-ext \
    --task_type train \
    --model_class BertGruAttnSaModel \
    --train_file ./data/train_data_public.csv \
    --dev_file ./data/train_data_public.csv \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name gruattn_sa_nodev \
    # --debug \