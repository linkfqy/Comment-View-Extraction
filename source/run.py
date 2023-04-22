import argparse
import model
import os
from train import Trainer, Tester
from datetime import datetime


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps_rate", type=float, default=None,
                        help="if given, warmup_steps will be disabled")
    parser.add_argument("--max_epoch", type=int, default=20)

    parser.add_argument("--pretrained_model", type=str,
                        default="models/bert-base-chinese")
    parser.add_argument("--task_type", type=str, default="train")
    parser.add_argument("--model_class", type=str, default="BertLinearModel")
    parser.add_argument("--train_file", type=str, default="data/train_splited.csv")
    parser.add_argument("--dev_file", type=str, default="data/dev_splited.csv")
    parser.add_argument("--test_file", type=str, default="data/test_public.csv")
    parser.add_argument("--checkpoint", type=str, default="save/xxx.pt")

    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_name", type=str, default="draft",
                        help="[NOT PATH] name used to save model and log")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    args.start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(f"./save/{args.start_time}")

    model = getattr(model,args.model_class)(args.__dict__)
    if args.task_type == 'train':
        trainer = Trainer(model, args.__dict__)
        trainer.train()
    else:
        tester = Tester(model, args.__dict__)
        tester.load_checkpoint()
        tester.test()
