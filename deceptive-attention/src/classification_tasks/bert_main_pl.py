# Global dependencies
import argparse
# 3rd party transformer code
import transformers
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
# PyTorch Modules
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from torch.optim import Adam
from torch.cuda import device_count

import sys
import time



# Local dependencies
from bert_attention import BertSelfAttention_Altered
from bert_util_pl import GenericDataModule

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERTModel(LightningModule):

    def __init__(self, penalize, lambeda, penalty_fn):
        super().__init__()
        """
        Args:
            penalize: flag to toggle attention manipulation and information flow restriction
            tokenizer: transformers object used to convert raw text to tensors
        """
        self.penalize = penalize
        self.lambeda = lambeda
        self.penalty_fn = penalty_fn
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #TODO: use a faster tokenizer?

        if self.penalize:
            # if we're penalizing the model's attending to impermissible tokens, we want to overwrite the original
            # self-attention class in the transformers library with a local class which has been adapted to ensure
            # the restriction of information flow between permissible and impermissible tokens
            transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention_Altered

        # load pretrained, uncased model
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def configure_optimizers(self):
        "This method handles optimization of params for PyTorch lightning"
        return Adam(self.parameters(), lr=config.lr)

    def forward(self, x, attention_mask, labels, head_mask=None):
        "This method defines how the data is passed through the net"
        if self.penalize:
            output = self.encoder(x, labels=labels, attention_mask=attention_mask, head_mask=head_mask, output_attentions=True)
        else:
            output = self.encoder(x, labels=labels, attention_mask=attention_mask, output_attentions=True)

        return output

    def training_step(self, batch, batch_idx):
        """
        This method implements the training step in PyTorch Lightning
        """

        # extract labels, sentences and vector masks m
        labels = batch['labels']
        sentences = batch['sentences']
        attention_masks = batch['attention_masks']
        mask_vectors = batch['mask_vectors']

        print('first sentence ids: ')
        print(sentences[0].shape)
        print(sentences[0])

        # flag to toggle manipulation of attention maps
        if self.penalize:

            # TODO: Generate self attention mask for info flow restriction
            # self_attention_masks = self.generate_mask(tokenized_sents["input_ids"],
            #                                      tokenized_impermissible["input_ids"])

            # Feed sentences through network
            outputs = self(x=sentences, attention_mask=attention_masks, labels=labels)
            # Compute loss w.r.t. predictions and labels
            loss = outputs.loss

            # TODO compute R component to add to loss
            # loss = loss + <R>
            R = self.compute_R(outputs, mask_vectors, self.lambeda, self.penalty_fn).sum()

            print(R)


            loss += R


        # Log loss to Tensorboard
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        This method implements the training step in PyTorch Lightning
        """

        # extract labels, sentences and vector masks m
        labels = batch['labels']
        sentences = batch['sentences']
        mask_vectors = batch['mask_vectors']
        attention_masks = batch['attention_masks']

        # flag to toggle manipulation of attention maps
        if self.penalize:

            # TODO: Generate self attention mask for info flow restriction
            # self_attention_masks = self.generate_mask(tokenized_sents["input_ids"],
            #                                      tokenized_impermissible["input_ids"])

            # Feed sentences through network
            outputs = self(x=sentences, attention_mask=attention_masks, labels=labels)
            # Compute loss w.r.t. predictions and labels
            loss = outputs.loss

            # TODO compute R component to add to loss
            # loss = loss + <R>

            # # extract self-attention maps of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
            # attention_matrices = outputs.attentions
            #
            # # convert attention matrices to vectors
            # attention_vectors = self.convert2vectors(attention_matrices)

        # Log loss to Tensorboard
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return None

    def test_step(self, batch, batch_idx):
        "This method handles the test logic in PyTorch Lightning"

        # extract label, sentence and set of impermissible words per sentence from batch
        labels = batch['label']
        sents = batch['sentence']
        impermissible = batch['impermissible']

        # # Tokenize batch of sentences
        tokenized_sents = self.tokenizer(sents, padding=True, truncation=False, return_tensors="pt")

        # Tokenize batch of impermissibles
        tokenized_impermissible = self.tokenizer(impermissible, padding=False, truncation=True, return_tensors="pt")

        # TODO: using sentence_ids and impermissible words, generate self-attention matrix per sentence
        if self.penalize:

            # Generate self attention mask based on permissible and impermissible token ids
            # self_attention_masks = self.generate_mask(tokenized_sents["input_ids"],
            #                                      tokenized_impermissible["input_ids"])

            outputs = self(x=tokenized_sents['input_ids'], labels=labels)
            # Compute loss w.r.t. predictions and labels
            loss = outputs.loss

            # Add penalty R to loss
            # TODO compute R component

        # Log loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return None

    def convert2vectors(self, matrices):
        """
        This function takes as its input a tuple with self-attention matrices,
        of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length],
        and returns a tensor with self-attention vectors, of shape
        [batch_size, num_layers, num_heads, sequence_length], by taking the max along the row-dimension.
        """

        # convert tuple of tensors to tensor of tensors (= a tensor)
        matrices = torch.stack(matrices)

        # for some reason, torch.stack switches the layer and batch dimension, which we do not want
        matrices = matrices.permute(1, 0, 2, 3, 4)

        # take max of n x n attention maps along row-dimension
        matrices, _ = torch.max(matrices, dim=3, out=None)

        # or, take diagonal of attention maps?
        # matrices = torch.diagonal(matrices, dim1=3, dim2=4)

        return matrices

    def compute_R(self, outputs, mask_vectors, lambeda, penalty_fn):
        """
        This function computes the R component, which serves as the penalizing mechanism as described in the paper
        """

        # extract self-attention maps of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
        attention_matrices = outputs.attentions

        print('attention matrix for first sample, first layer, first head')
        print(attention_matrices[0][0][0].shape)
        print(attention_matrices[0][0][0])

        #TODO: attention matrices hebben ook non-negative waarden op de allerbuitenste indices

        # convert attention matrices to vectors
        attention_vectors = self.convert2vectors(attention_matrices)

        sequence_length = attention_vectors.shape[-1]
        if penalty_fn == 'mean':

            print('attention vector shape for batch of size 8:')
            print(attention_vectors.shape)
            print('attention vector for first sample, first layer, first head')
            print(attention_vectors[0][0][0])
            print('summing over sequence-length dimension: ')
            print(torch.sum(attention_vectors[0][0][0]))

            #TODO: explore possibility of head_mask argument

            sys.exit()

            # reshape batch_size * 12 x 12 (12 layers, 12 heads) x seq_length to batch_size, 144, seq_length
            attention_vectors = attention_vectors.reshape(config.batch_size, 12**2, sequence_length)

            # add implicit dimension to mask_vectors such that it becomes a rank-4 tensor
            mask_vectors = mask_vectors.unsqueeze(1)

            # compute impermissible attention tensor of shape (batch_size, 12**2, seq_length)
            impermissible_attention = attention_vectors * mask_vectors

            # Sum over last dim to get impermissible attention per head
            impermissible_attention = torch.sum(impermissible_attention, dim=2)

            # Compute the complement or permissible attention
            permissible_attention = 1 - impermissible_attention

            # log permissible attention per head
            log_permissible_attention = torch.log(permissible_attention)

            # Compute R value using 'mean' method
            R = (-lambeda/144)*(torch.sum(log_permissible_attention, dim=1))

            return R


def main(args):

    print('\n -------------- Classification with BERT -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print('no of devices found: {}'.format(device_count()))

    # Logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(penalize=config.penalize, lambeda=config.lambeda, penalty_fn=config.penalty_fn)

    # Crude logic to freeze all the BERT parameters
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # Main Lightning code
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(gpus=config.gpus, logger=tb_logger)
    dm = GenericDataModule(task=config.task,
                                   anonymization=config.anon,
                                   max_length=config.max_length,
                                   batch_size=config.batch_size)
    # dm.setup()
    trainer.fit(model, dm)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Learning specific args
    parser.add_argument('--gpus', default=None)

    parser.add_argument('--batch_size', default=16, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')

    parser.add_argument('--max_length', default=180, type=int,
                        help='max no of tokens for tokenizer')

    # Task specific args
    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')
    parser.add_argument('--lambeda', default=0.1, type=float,
                        help='penalty coefficient')
    parser.add_argument('--penalty_fn', default='mean', type=str,
                        help='penalty fn [options: mean or max')

    config = parser.parse_args()

    main(config)