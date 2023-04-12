from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


class MyDataset(Dataset):

    label_list = ["B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", "O",
                  "B-COMMENTS_N", "I-COMMENTS_N", "B-COMMENTS_ADJ", "I-COMMENTS_ADJ",]
    label_number = len(label_list)
    label_dict = {}
    for i, label in enumerate(label_list):
        label_dict[label] = i

    def bio2ids(bio):
        return __class__.label_dict[bio]

    def __init__(self, tokenizer, arg_dict, type='train'):
        super().__init__()
        self.__dict__.update(arg_dict)

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
            id = row['id']
            text = row['text']
            if type(text) == float:
                print("error data:", text)
                continue
            if self.type == 'test':
                bio = []
                class_ = None
            else:
                bio = row['BIO_anno'].split()
                class_ = row['class']
            data.append({
                "id": id,
                "text": text,
                "BIO": bio,
                "class": class_,
            })
        if self.debug:
            data=data[:100]
        return data

    def tokenize_data(self):
        '''
        使用self.tokenizer将self.raw_data转化成model_inputs（长度不一）
        由于本题的BIO标签是character-based的，
        需要手动将text分为单个字符，而非使用tokenizer分词
        '''
        model_inputs = []
        for data in self.raw_data:
            newtext = ' '.join(data['text'])
            input = self.tokenizer(newtext)
            input['BIO_ids'] = list(map(__class__.bio2ids, data['BIO']))
            input['class'] = data['class']
            input['id'] = data['id']
            # 所含字段：input_ids, token_type_ids, attention_mask, BIO_ids, class, id
            model_inputs.append(input)
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
            batch={}
            for key in raw_batch[0].keys():
                batch[key]=[]
                for data in raw_batch:
                    batch[key].append(tensor(data[key]))
            
            batch['input_ids'] = pad_sequence(batch['input_ids'], True, self.pad_id)
            batch['token_type_ids'] = pad_sequence(batch['token_type_ids'], True, 0)
            batch['attention_mask'] = pad_sequence(batch['attention_mask'], True, 0)
            batch['BIO_ids'] = pad_sequence(batch['BIO_ids'], True, __class__.bio2ids('O'))
            batch['class']=tensor(batch['class'])
            batch['id']=tensor(batch['id'])
            return batch

        return DataLoader(dataset=self, batch_size=self.batch_size, collate_fn=collate_fn)

if __name__=='__main__':
    args= {
        'data_path': 'data/dev_splited.csv',
        'batch_size': 4,
        'debug':True,
    }
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('model/bert-base-chinese')
    dataset = MyDataset(tokenizer,args,'train')
    dataloader=dataset.get_dataloader()
    for batch in dataloader:
        print(batch)
        exit(0)