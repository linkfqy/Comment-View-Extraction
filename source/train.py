import torch
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from my_log import get_logger
from dataset import MyDataset
from model import MyModel
from utils import AvgCalc
from tqdm import tqdm
from utils import *


class Trainer():
    def __init__(self, model: MyModel, arg_dict):
        self.__dict__.update(arg_dict)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.logger = get_logger(f"save/{self.save_name}_{self.start_time}.log")
        self.logger.info('start training!')
        self.logger.info(arg_dict)

        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.logger.info(f"model name: {self.model.__class__.__name__}")

        self.train_set = MyDataset(self.train_file, self.tokenizer, arg_dict, 'train')
        self.dev_set = MyDataset(self.dev_file, self.tokenizer, arg_dict, 'train')

        self.train_dataloader = self.train_set.get_dataloader()
        self.dev_dataloader = self.dev_set.get_dataloader()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.total_step = len(self.train_dataloader)*self.max_epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.warmup_steps, self.total_step
        )

    def train(self):
        min_loss = float('inf')
        max_score = float('-inf')
        self.logger.info("Start Training")
        for epoch in range(1, self.max_epoch+1):
            loss = self.train_epoch(epoch)
            score, s_ner, s_sa = self.evaluate_epoch()
            self.logger.info(
                f"Epoch[{epoch}/{self.max_epoch}]\t"
                f"loss={float(loss):.6f}\t"
                f"score={score:.6f}\t"
                f"s_ner={s_ner}\t"
                f"s_sa={s_sa}\t"
            )
            min_loss = min(min_loss, loss)
            if score > max_score:
                max_score = score
                self.save_state_dict(
                    f"save/{self.save_name}_epoch{epoch}_loss{loss:.6f}_score{score:.6f}_{self.start_time}.pt"
                )
                self.logger.info(f"saved as checkpoint")
        self.optimizer.zero_grad()
        self.logger.info("Training Finished")

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        avgloss = AvgCalc()
        bar = tqdm(self.train_dataloader)
        for batch in bar:
            self.optimizer.zero_grad()
            output = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                token_type_ids=batch['token_type_ids'].to(self.device),
                labels=batch['BIO_ids'].to(self.device),
                classes=batch['class'].to(self.device),
            )
            # loss为两个任务loss直接相加
            loss = output['ner_loss']+output['sa_loss']
            avgloss.put(loss)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            bar.set_description(f"Train epoch {epoch}, loss {loss:.6f}")
        return avgloss.get_avg()

    @torch.no_grad()
    def evaluate_epoch(self):
        self.model.eval()
        self.model.to(self.device)
        s_set = set()
        g_set = set()
        confusion_matrix = np.zeros((class_number, class_number))

        bar = tqdm(self.dev_dataloader, "Evaluating Epoch")
        for batch in bar:
            output = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                token_type_ids=batch['token_type_ids'].to(self.device),
                predict=True,
            )
            ner, sa = output['ner'], output['sa']
            g_ner, g_sa = batch['BIO_ids'].numpy(), batch['class'].numpy()
            id, raw_len = batch['id'].numpy(), batch['raw_len'].numpy()
            for i in range(len(id)):
                s_set.update(ner2set(id[i], ner[i]))
                g_set.update(ner2set(id[i], g_ner[i]))
                confusion_matrix[sa[i], g_sa[i]] += 1

        score_ner = f1(s_set, g_set)
        score_sa = kappa(confusion_matrix)
        score = (score_ner+score_sa)/2
        return score, score_ner, score_sa

    def save_state_dict(self, filename):
        torch.save({
            "training_state": self.get_training_state(),
            "model": self.model.state_dict(),
        }, filename)

    def get_training_state(self):
        # 待完善
        return {
            "last_lr": self.scheduler.get_last_lr()[0],
        }


class Tester:
    def __init__(self, model: MyModel, arg_dict) -> None:
        self.__dict__.update(arg_dict)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.model = model
        self.tokenizer = model.get_tokenizer()

        self.test_set = MyDataset(self.test_file, self.tokenizer, arg_dict, 'test')
        self.test_dataloader = self.test_set.get_dataloader()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.model.to(self.device)

        id_predict, ner_predict, sa_predict = [], [], []
        for batch in tqdm(self.test_dataloader, "Testing"):
            output = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                token_type_ids=batch['token_type_ids'].to(self.device),
                predict=True,
            )
            ner, sa = output['ner'], output['sa']
            id, raw_len = batch['id'].numpy(), batch['raw_len'].numpy()
            for i in range(len(id)):
                ner_ids = ner[i, 1:raw_len[i]+1]
                ner_res = ' '.join(map(ids2bio, ner_ids))
                id_predict.append(id[i])
                ner_predict.append(ner_res)
                sa_predict.append(sa[i])
        df=pd.DataFrame(
            {
                'id':id_predict,
                'BIO_anno':ner_predict,
                'class':sa_predict,
            }
        ).sort_values(by="id",ascending=True)
        df.to_csv(f"save/{self.save_name}.csv",index=False)

    def load_checkpoint(self):
        checkpoint=torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()