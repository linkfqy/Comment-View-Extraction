from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re
from collections import UserDict
from utils import *


class MyBatch(UserDict):
    '''
    to_model: list of keys whose value is the input of model
    e.g. ['input_ids', 'attention_mask', ...]
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_model = []

    def to(self, device):
        '''
        move all values of 'to_model' to device
        '''
        res = self.copy()
        for k in self.to_model:
            res[k] = res[k].to(device)
        return res


class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, arg_dict, type='train'):
        '''
        可能用到的arg_dict中的参数：batch_size, debug
        type为train时，默认数据中含BIO和class，否则是一个无监督的数据集
        '''
        super().__init__()
        self.__dict__.update(arg_dict)

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.type = type

        self.pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
        self.raw_data = self.read_csv()
        self.dataset = self.tokenize_data()

    def read_csv(self):
        '''
        从csv中读入数据，不做任何处理
        '''
        df = pd.read_csv(self.data_path)
        data = []
        for _, row in df.iterrows():
            row = row.to_dict()
            if self.type == 'train':
                row['BIO_anno'] = row['BIO_anno'].split()
                assert len(row['BIO_anno']) == len(row['text']), f"ERROR id={row['id']}"
            data.append(row)
        if self.debug:
            data = data[:100]
            print(data[0])
        return data

    def tokenize_data(self):
        '''
        使用self.tokenizer将self.raw_data转化成model_inputs（长度不一）
        由于本题的BIO标签是character-based的，
        需要手动将text分为单个字符，而非使用tokenizer分词
        '''
        model_inputs = []
        for data in self.raw_data:
            # 处理脏数据
            newtext = ' '.join(re.sub(r"\r\n|\r|\n| |\t|　|\ufe0f", '，', data['text']))
            input = self.tokenizer(newtext)
            if self.type == 'train':
                input['bio_ids'] = (
                    [bio2ids('O')]
                    + list(map(bio2ids, data['BIO_anno']))
                    + [bio2ids('O')]
                )
                input['sa_ids'] = data['class']
            input['id'] = data['id']
            input['raw_len'] = len(data['text'])
            # 所含字段：input_ids, token_type_ids, attention_mask, (bio_ids, class,) id, raw_len
            model_inputs.append(input)
        if self.debug:
            input = model_inputs[0]
            print(input)
            print(' '.join(self.tokenizer.convert_ids_to_tokens(input['input_ids'])))
        return model_inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_dataloader(self):
        def collate_fn(raw_batch):
            '''
            将同个batch的输入长度对齐，以便输入BERT
            '''
            batch = MyBatch()
            for key in raw_batch[0].keys():
                batch[key] = []
                for data in raw_batch:
                    batch[key].append(tensor(data[key]))

            batch['input_ids'] = pad_sequence(batch['input_ids'], True, self.pad_id)
            batch['token_type_ids'] = pad_sequence(batch['token_type_ids'], True, 0)
            batch['attention_mask'] = pad_sequence(batch['attention_mask'], True, 0)
            batch.to_model = ['input_ids', 'token_type_ids', 'attention_mask']
            batch['id'] = tensor(batch['id'])
            batch['raw_len'] = tensor(batch['raw_len'])
            if self.type == 'train':
                batch['bio_ids'] = pad_sequence(batch['bio_ids'], True, bio2ids('O'))
                batch['sa_ids'] = tensor(batch['sa_ids'])
                batch.to_model.extend(['bio_ids', 'sa_ids'])
                assert batch['input_ids'].size() == batch['bio_ids'].size(), \
                    f"ERROR!!! id={batch['id']}"

            return batch

        return DataLoader(dataset=self, batch_size=self.batch_size, collate_fn=collate_fn)


class PromptDataset(MyDataset):
    def __init__(self, data_path, tokenizer, arg_dict, type='train'):
        super().__init__(data_path, tokenizer, arg_dict, type)
        self.mask_id = tokenizer.mask_token_id
        self.prompt = "这段话的情感是[MASK][MASK]的[SEP]"
        self.prompt_ids = tokenizer(self.prompt, return_tensors='pt').input_ids[0, 1:-1]
        self.prompt_len = len(self.prompt_ids)
        self.mask_pos = 8

    def get_dataloader(self):
        def collate_fn(raw_batch):
            '''
            将同个batch的输入长度对齐，以便输入BERT
            并在开头添加prompt
            '''
            batch = MyBatch()
            for key in raw_batch[0].keys():
                batch[key] = []
                for data in raw_batch:
                    batch[key].append(tensor(data[key]))

            batch['input_ids'] = insertPrompt(
                pad_sequence(batch['input_ids'], True, self.pad_id),
                self.prompt_ids, 1
            )
            batch['token_type_ids'] = insertPrompt(
                pad_sequence(batch['token_type_ids'], True, 0),
                torch.zeros_like(self.prompt_ids), 1
            )
            batch['attention_mask'] = insertPrompt(
                pad_sequence(batch['attention_mask'], True, 0),
                torch.ones_like(self.prompt_ids), 1
            )
            batch.to_model = ['input_ids', 'token_type_ids', 'attention_mask']
            batch['id'] = tensor(batch['id'])
            batch['raw_len'] = tensor(batch['raw_len'])
            if self.type == 'train':
                batch['bio_ids'] = insertPrompt(
                    pad_sequence(batch['bio_ids'], True, bio2ids('O')),
                    torch.ones_like(self.prompt_ids)*bio2ids('O'), 1
                )
                batch['sa_ids'] = tensor(batch['sa_ids'])
                lm_labels = torch.ones_like(batch['input_ids'])*(-100)
                lm_labels[:, self.mask_pos:self.mask_pos+2] = torch.tensor(
                    list(map(ids2class, batch['sa_ids']))
                )
                batch['lm_labels'] = lm_labels
                batch.to_model.append('lm_labels')

            return batch

        return DataLoader(dataset=self, batch_size=self.batch_size, collate_fn=collate_fn)


if __name__ == '__main__':
    arg_dict = {
        'batch_size': 4,
        'debug': True,
    }
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/chinese-roberta-wwm-ext')
    dataset = PromptDataset('data/train_splited.csv', tokenizer, arg_dict, 'train')
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        print(batch)
        exit(0)
