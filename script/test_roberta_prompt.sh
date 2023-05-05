export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
python ./source/run.py \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --warmup_steps_rate 0.1 \
    --max_epoch 20 \
    \
    --pretrained_model ./models/chinese-roberta-wwm-ext-large \
    --task_type test \
    --model_class BertPromptSaModel \
    --dataset_class PromptDataset \
    --test_file ./data/test_public.csv \
    --checkpoint ./save/20230427-162822/prompt_sa_epoch4_loss0.114316_score0.290074.pt \
    \
    --seed 19260817 \
    --max_length 512 \
    --save_name prompt_sa \
    # --debug \