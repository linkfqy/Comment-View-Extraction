from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re
from utils import *


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
                assert len(row['BIO_anno'])==len(row['text']), f"ERROR id={row['id']}"
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
            newtext = ' '.join(re.sub(r"\r\n|\r|\n| |\t",'，',data['text']))
            input = self.tokenizer(newtext)
            if self.type == 'train':
                input['BIO_ids'] = [bio2ids('O')]+list(map(bio2ids, data['BIO_anno']))+[bio2ids('O')]
                input['class'] = data['class']
            input['id'] = data['id']
            input['raw_len']=len(data['text'])
            # 所含字段：input_ids, token_type_ids, attention_mask, (BIO_ids, class,) id, raw_len
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
            batch = {}
            for key in raw_batch[0].keys():
                batch[key] = []
                for data in raw_batch:
                    batch[key].append(tensor(data[key]))

            batch['input_ids'] = pad_sequence(batch['input_ids'], True, self.pad_id)
            batch['token_type_ids'] = pad_sequence(batch['token_type_ids'], True, 0)
            batch['attention_mask'] = pad_sequence(batch['attention_mask'], True, 0)
            batch['id'] = tensor(batch['id'])
            batch['raw_len'] = tensor(batch['raw_len'])
            if self.type == 'train':
                batch['BIO_ids'] = pad_sequence(batch['BIO_ids'], True, bio2ids('O'))
                batch['class'] = tensor(batch['class'])
                assert batch['input_ids'].size()==batch['BIO_ids'].size(), f"ERROR!!! id={batch['id']}"
            
            return batch

        return DataLoader(dataset=self, batch_size=self.batch_size, collate_fn=collate_fn)


if __name__ == '__main__':
    arg_dict = {
        'batch_size': 4,
        'debug': True,
    }
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/bert-base-chinese')
    dataset = MyDataset('data/train_splited.csv', tokenizer, arg_dict, 'train')
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        print(batch)
        exit(0)
