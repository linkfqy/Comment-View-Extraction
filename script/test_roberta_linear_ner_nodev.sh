export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --warmup_steps 2000 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/chinese-roberta-wwm-ext-large \
    --task_type test \
    --model_class BertNerModel \
    --checkpoint ./save/linear_ner_nodev_epoch18_loss0.000471_score0.495831_20230421-195856.pt \
    --test_file ./data/test_public.csv \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name linear_ner_nodev \
    # --debug \