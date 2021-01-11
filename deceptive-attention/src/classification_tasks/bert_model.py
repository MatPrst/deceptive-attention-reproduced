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

    def __init__(self, penalize):
        super(BERTModel, self).__init__()
        """
        Args:

        """
        self.penalize = penalize
        if self.penalize:
            # if we're penalizing the model's attending to impermissible tokens,
            # we want to overwrite the original self-attention module with a local module
            # to ensure the restriction of information flow
            transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention_Altered

        # load pretrained uncased model
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.l1 = nn.Linear(768, 2)

        # verify that self attention mechanism is now handled by the local module
        # print(self.encoder._modules['encoder'].layer[0].attention.self)

    def forward(self, x, head_mask=None):
        if self.penalize:
            output = self.encoder(x, head_mask, output_attentions=True)
        else:
            output = self.encoder(x, output_attentions=True)

        y = self.l1(output.pooler_output)
        return y, output.attentions