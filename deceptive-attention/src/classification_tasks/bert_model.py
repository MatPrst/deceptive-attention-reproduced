# global dependencies
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import transformers
import math
import time

# local dependencies
from bert_attention import BertSelfAttention_Altered

#TODO: write class for BERT model with args to specify how R is calculated, whether mean or max attn head is chosen
class BERTModel(nn.Module):

    def __init__(self, attention_mechanism=None):
        super(BERTModel, self).__init__()
        """
        Args:

        """
        # overwrite self-attention module with local module
        transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention_Altered

        # load pretrained uncased model
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')

        # verify that self attention mechanism is now handled by the local module
        # print(self.encoder._modules['encoder'].layer[0].attention.self)

    def forward(self, x):
        output = self.encoder(input)
        return output