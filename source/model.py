import torch.nn as nn
from transformers import BertTokenizer


class BERTLinearModel(nn.Module):
    def __init__(self, arg_dict) -> None:
        super().__init__()
        self.__dict__.update(arg_dict)

    def get_tokenizer(self):
        return BertTokenizer.from_pretrained(self.pretrained_model)
