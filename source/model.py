import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, BertModel
from utils import *


class MyModel(nn.Module):
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_model)

    def get_params(self):
        '''
        params for create optimizer
        '''
        return self.parameters()

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Tensor = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


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


class BertNerModel(MyModel):
    def __init__(self, arg_dict) -> None:
        '''
        可能用到的arg_dict中的参数：pretrained_model
        '''
        super().__init__()
        self.__dict__.update(arg_dict)

        self.bert = BertModel.from_pretrained(self.pretrained_model)
        self.classifier4NER = nn.Linear(self.bert.config.hidden_size, label_number)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, classes=None, predict=False):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        ner_logits = self.classifier4NER(bert_output.last_hidden_state)

        output = {'ner_logits': ner_logits}
        if labels is not None:
            output['ner_loss'] = self.loss_fn(
                ner_logits.view(-1, label_number),
                labels.view(-1)
            )
            output['sa_loss'] = 0
        if predict:
            # shape(batch, seq_len)
            output['ner'] = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            # shape(batch,)
            output['sa'] = np.zeros((input_ids.size()[0],), dtype=int)

        return output


class BertSaModel(MyModel):
    def __init__(self, arg_dict) -> None:
        '''
        可能用到的arg_dict中的参数：pretrained_model
        '''
        super().__init__()
        self.__dict__.update(arg_dict)

        self.bert = BertModel.from_pretrained(self.pretrained_model)
        self.classifier4SA = nn.Linear(self.bert.config.hidden_size, class_number)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, classes=None, predict=False):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        sa_logits = self.classifier4SA(bert_output.pooler_output)

        output = {'sa_logits': sa_logits}
        if classes is not None:
            output['ner_loss'] = 0
            output['sa_loss'] = self.loss_fn(sa_logits, classes)
        if predict:
            # shape(batch, seq_len)
            size = input_ids.size()
            output['ner'] = np.zeros((size[0], size[1]), dtype=int)
            # shape(batch,)
            output['sa'] = torch.argmax(sa_logits, dim=-1).cpu().numpy()

        return output


class BertGruAttnSaModel(MyModel):
    def __init__(self, arg_dict) -> None:
        '''
        可能用到的arg_dict中的参数：pretrained_model, attn_head, learning_rate
        '''
        super().__init__()
        self.__dict__.update(arg_dict)

        self.bert = BertModel.from_pretrained(self.pretrained_model)
        self.bert_h = self.bert.config.hidden_size
        self.gru = nn.GRU(self.bert_h, self.bert_h//2,
                          batch_first=True, bidirectional=True)
        self.classifier4SA = nn.Linear(self.bert_h*2, class_number)

        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = FocalLoss(
            alpha=torch.tensor([.3, .6, .1]),
            gamma=2,
        )

    def get_params(self):
        return [
            {'params': self.bert.parameters(), 'lr': self.learning_rate},
            {'params': self.gru.parameters(), 'lr': self.learning_rate*5},
            {'params': self.classifier4SA.parameters(), 'lr': self.learning_rate*5},
        ]

    def attention_net(self, gru_output, final_state):
        '''
        gru_output: (batch, seq_len, bert_h)
        final_state: (2, batch, bert_h//2)
        '''
        # shape(batch, bert_h, 1)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # shape(batch, seq_len)
        attn_weights = torch.bmm(gru_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        # shape(batch, bert_h)
        context = torch.bmm(
            gru_output.transpose(1, 2),
            soft_attn_weights.unsqueeze(2)
        ).squeeze(2)
        return context, soft_attn_weights

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, classes=None, predict=False):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        # shape(batch, seq_len, bert_h)
        seq_output = bert_output.last_hidden_state
        # shape(batch, bert_h)
        pool_output = bert_output.pooler_output
        gru_output, h_n = self.gru(seq_output)
        attn_output, _ = self.attention_net(gru_output, h_n)
        concated = torch.cat([pool_output, attn_output], dim=-1)
        sa_logits = self.classifier4SA(concated)
        # print(sa_logits.size())

        output = {'sa_logits': sa_logits}
        if classes is not None:
            output['ner_loss'] = 0
            output['sa_loss'] = self.loss_fn(sa_logits, classes)
        if predict:
            # shape(batch, seq_len)
            size = input_ids.size()
            output['ner'] = np.zeros((size[0], size[1]), dtype=int)
            # shape(batch,)
            output['sa'] = torch.argmax(sa_logits, dim=-1).cpu().numpy()

        return output
