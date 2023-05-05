export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --warmup_steps 2000 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/chinese-roberta-wwm-ext \
    --task_type test \
    --model_class BertGruAttnSaModel \
    --test_file ./data/test_public.csv \
    --checkpoint ./save/20230423-114228/gruattn_sa_nodev_epoch11_loss0.000447_score0.287571.pt \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name gruattn_sa \
    # --debug \