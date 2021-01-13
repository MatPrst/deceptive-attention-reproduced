# Global dependencies
import argparse
# 3rd party transformer code
import transformers
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
# PyTorch Modules
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from torch.optim import Adam
from torch.cuda import device_count

import sys

# Local dependencies
from bert_attention import BertSelfAttention_Altered
from bert_util_pl import GenericDataModule

class BERTModel(LightningModule):

    def __init__(self, penalize):
        super().__init__()
        """
        Args:
            penalize: flag to toggle attention manipulation and information flow restriction
            tokenizer: transformers object used to convert raw text to tensors
        """
        self.penalize = penalize
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.loss_fn = nn.CrossEntropyLoss()
        # if self.penalize:
        #     # if we're penalizing the model's attending to impermissible tokens,
        #     # we want to overwrite the original self-attention module in the transformers module with a local module
        #     # which has been adapted to ensure the restriction of information flow between
        #     # permissible and impermissible tokens
        #     transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention_Altered

        # load pretrained uncased model
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.l1 = nn.Linear(in_features=768, out_features=2)

        # verify that self attention mechanism is now handled by the local module
        # print(self.encoder._modules['encoder'].layer[0].attention.self)

    def configure_optimizers(self):
        "This method handles optimization of params for PyTorch lightning"
        return Adam(self.parameters(), lr=1e-3)

    def forward(self, x, head_mask=None):
        "This method defines how the data is passed through the net"
        if self.penalize:
            output = self.encoder(x, head_mask, output_attentions=True)
        else:
            output = self.encoder(x, output_attentions=True)
        # extract [CLS] representation and feed through linear layer
        y = self.l1(output.pooler_output)
        # return predictions. attention tensor might be of use later.
        return y, output.attentions

    def training_step(self, batch, batch_idx):
        "This method handles the training loop logic for PyTorch Lightning"

        # extract label, sentence and set of impermissible words per sentence from batch
        labels = batch['label']
        sents = batch['sentence']
        impermissible = batch['impermissible']

        # # Tokenize batch of sentences
        tokenized_sents = self.tokenizer(sents, padding=True, truncation=False, return_tensors="pt")
        # print(tokenized_sents["input_ids"].type())

        # Tokenize batch of impermissibles
        tokenized_impermissible = self.tokenizer(impermissible, padding=False, truncation=True, return_tensors="pt")

        # TODO: using sentence_ids and impermissible words, generate self-attention matrix per sentence
        if self.penalize:

            # Generate self attention mask based on permissible and impermissible token ids
            # self_attention_masks = self.generate_mask(tokenized_sents["input_ids"],
            #                                      tokenized_impermissible["input_ids"])

            print(tokenized_sents["input_ids"].type())
            print(labels.type())
            print(tokenized_impermissible.type())
            sys.exit()

            # Feed data through model, along with self-attn masks
            preds, attentions = self(tokenized_sents["input_ids"])

            # Compute loss w.r.t. predictions and labels
            loss = self.loss_fn(preds, labels)

            # Add penalty R to loss
            # TODO compute R component

        # Log loss to Tensorboard
        self.log('loss',loss)
        return loss

    # TODO: implement this function
    def generate_mask(self, tokenized_sents_ids, tokenized_impermissible_ids):
        """
        This function generates the self-attention mask M required to ensure
        a restriction in the information flow between permissible and impermissible tokens.
        """
        batch_size, seq_length = tokenized_sents_ids.shape

        # for i in range(batch_size):
        #
        #     # print(tokenized_sents_ids[i])
        #
        #     print(tokenized_impermissible_ids[i])
        #
        #     for ids in tokenized_impermissible_ids[i]:
        #         print(AutoTokenizer.from_pretrained('bert-base-uncased').decode(ids))
        #
        #     # id 101 = [CLS]
        #     # id 102 = [SEP]
        #
        #     sys.exit()
        return 2

def main(args):

    print('\n -------------- Classification with BERT -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))

    print('no of devices found: {}'.format(device_count()))

    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    # Logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(penalize=config.penalize)

    # Crude logic to freeze all the BERT parameters
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    trainer = Trainer(gpus=config.gpus, logger=tb_logger,overfit_batches=1)
    dm = GenericDataModule(task=config.task,
                                   anonymization=config.anon,
                                   batch_size=config.batch_size)
    dm.setup()
    trainer.fit(model, dm)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=None)

    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')

    config = parser.parse_args()

    main(config)