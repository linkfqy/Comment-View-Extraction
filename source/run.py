import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_epoch", type=int, default=20)
    
    parser.add_argument("--pretrained_model", type=str, default="model/bert-base-chinese")
    parser.add_argument("--train_file", type=str, default="data/train_splited.csv")
    parser.add_argument("--dev_file", type=str, default="data/dev_splited.csv")
    parser.add_argument("--task_type", default="train", type=str)
    parser.add_argument("--num_warmup_steps", default=2000, type=int)
    
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_file", type=str, default="MedicineExam/save/test.log")
    parser.add_argument("--save_file", type=str, default="MedicineExam/save/test.pt")
    parser.add_argument("--debug", type=bool, default=False)
    
    return parser.parse_args()

if __name__=='__main__':
    args = init_args()
    