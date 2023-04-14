import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from utils import *


class MyModel(nn.Module):
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_model)


class BertLinearModel(MyModel):
    def __init__(self, arg_dict) -> None:
        '''
        可能用到的arg_dict中的参数：pretrained_model
        '''
        super().__init__()
        self.__dict__.update(arg_dict)

        self.bert = BertModel.from_pretrained(self.pretrained_model)
        self.classifier4NER = nn.Linear(self.bert.config.hidden_size, label_number)
        self.classifier4SA = nn.Linear(self.bert.config.hidden_size, class_number)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, classes=None, predict=False):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        ner_logits = self.classifier4NER(bert_output.last_hidden_state)
        sa_logits = self.classifier4SA(bert_output.pooler_output)

        output = {'ner_logits': ner_logits, 'sa_logits': sa_logits}
        if labels is not None:
            output['ner_loss'] = self.loss_fn(
                ner_logits.view(-1, label_number),
                labels.view(-1)
            )
        if classes is not None:
            output['sa_loss'] = self.loss_fn(sa_logits, classes)
        if predict:
            # shape(batch, seq_len)
            output['ner'] = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            # shape(batch,)
            output['sa'] = torch.argmax(sa_logits, dim=-1).cpu().numpy()

        return output
