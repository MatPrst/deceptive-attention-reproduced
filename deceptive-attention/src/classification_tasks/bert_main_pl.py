# Global modules
import argparse
import sys
import time
import os

# PyTorch modules
import torch
from torch.optim import Adam
from torch.cuda import device_count
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics, seed_everything
from pytorch_lightning.callbacks import Callback

# Local dependencies
from bert_util_pl import GenericDataModule

# Local transformer dependencies
# sys.path.append(os.path.join(os.getcwd(), 'classification_tasks')) #
from transformers_editted.src.transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers_editted.src.transformers.models.auto.tokenization_auto import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERTModel(LightningModule):

    def __init__(self, dropout, lr, penalize, lambeda, penalty_fn):
        super().__init__()
        """
        Args:
            penalize: flag to toggle attention manipulation and information flow restriction
            tokenizer: transformers object used to convert raw text to tensors
        """
        self.dropout = dropout
        self.lr = lr
        self.penalize = penalize
        self.lambeda = lambeda
        self.penalty_fn = penalty_fn

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #TODO: use a faster tokenizer?

        # load pretrained, uncased model
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                     num_labels=2,
                                                                     hidden_dropout_prob=self.dropout)

        self.accuracy = metrics.Accuracy()

    def configure_optimizers(self):
        "This method handles optimization of params for PyTorch lightning"
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, x, attention_mask, labels, mask_matrices):
        "This method defines how the data is passed through the net."

        # if we are applying the R penalty, apply information flow restriction
        # by passing in mask_matrices into the the matrix_mask arg
        # if self.penalize:

        output = self.encoder(x,
                              labels=labels,
                              attention_mask=attention_mask,
                              matrix_mask=mask_matrices,
                              output_attentions=True)
        # else:
        #     output = self.encoder(x,
        #                           labels=labels,
        #                           attention_mask=attention_mask,
        #                           matrix_mask=None,
        #                           output_attentions=True)

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
        mask_matrices = batch['mask_matrices'].unsqueeze(1) # add necessary implicit 1st dim

        # Feed sentences through network
        outputs = self(x=sentences, attention_mask=attention_masks, mask_matrices=mask_matrices, labels=labels)

        # Compute loss w.r.t. predictions and labels
        loss = outputs.loss

        # Compute R component and add to loss
        R, attention_mass = self.compute_R(outputs, mask_vectors, self.lambeda, self.penalty_fn)
        R = R.mean()
        # flag to toggle manipulation of attention maps
        if self.penalize:
            loss += R

        self.log('train_penalty_R', R)
        self.log('train_attention_mass', attention_mass)

        preds = outputs.logits

        # Log train stats to Tensorboard
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(preds, labels))

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
        mask_matrices = batch['mask_matrices'].unsqueeze(1)

        # Feed sentences through network
        outputs = self(x=sentences, attention_mask=attention_masks, mask_matrices=mask_matrices, labels=labels)

        # Compute loss w.r.t. predictions and labels
        loss = outputs.loss

        # Compute R component and add to loss
        R, attention_mass = self.compute_R(outputs, mask_vectors, self.lambeda, self.penalty_fn)
        R = R.mean()
        # flag to toggle manipulation of attention maps
        if self.penalize:
            loss += R
        self.log('dev_penalty_R', R)
        self.log('dev_attention_mass', attention_mass)

        preds = outputs.logits

        # Log dev stats to Tensorboard
        self.log('dev_loss', loss)
        self.log('dev_acc_step', self.accuracy(preds, labels))

        return None

    def test_step(self, batch, batch_idx):
        "This method handles the test logic in PyTorch Lightning"

        # extract labels, sentences and vector masks m
        labels = batch['labels']
        sentences = batch['sentences']
        mask_vectors = batch['mask_vectors']
        attention_masks = batch['attention_masks']
        mask_matrices = batch['mask_matrices'].unsqueeze(1)

        # Feed sentences through network
        outputs = self(x=sentences, attention_mask=attention_masks, mask_matrices=mask_matrices, labels=labels)

        # Compute loss w.r.t. predictions and labels
        loss = outputs.loss

        # Compute R component and add to loss
        R, attention_mass = self.compute_R(outputs, mask_vectors, self.lambeda, self.penalty_fn)
        R = R.mean()
        # flag to toggle manipulation of attention maps
        if self.penalize:
            loss += R
        self.log('test_penalty_R', R)
        self.log('test_attention_mass', attention_mass)

        preds = outputs.logits

        # Log dev stats to Tensorboard
        self.log('test_loss', loss)
        self.log('test_acc_step', self.accuracy(preds, labels))

        return None

    def convert2vectors(self, matrices):
        """
        This function takes as its input a tuple with self-attention matrices,
        of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length],
        and returns a tensor with self-attention vectors, of shape
        [batch_size, num_layers, num_heads, sequence_length], by taking the max along the row-dimension.
        """

        # OLD CODE:

                # # convert tuple of tensors to tensor of tensors (= a tensor)
                # matrices = torch.stack(matrices)
                #
                # # for some reason, torch.stack switches the layer and batch dimension, which we do not want
                # matrices = matrices.permute(1, 0, 2, 3, 4)
                #
                # # take max of n x n attention maps along row-dimension
                # matrices, _ = torch.max(matrices, dim=3, out=None)
                #
                # # renormalize to 1?
                # m = nn.Softmax(dim=3)
                # matrices[matrices == 0.0] = -9999
                # matrices = m(matrices)

                # return matrices

        # NEW CODE

        # take last layer;
        # num_layers x batch_sz x num_heads x n x n
        # -> batch_sz x num_heads x n x n
        matrices = matrices[-1][:,:,0]

        # take first row of self-attention matrix
        # row represents extent to which CLS attends to other tokens
        vectors = matrices[:,:,0]

        return vectors

    def compute_R(self, outputs, mask_vectors, lambeda, penalty_fn):
        """
        This function computes the R component, which serves as the penalizing mechanism as described in the paper
        """

        # extract self-attention tuple of size 12 (one self-attention map per layer)
        # tuple contains tensors of shape [batch_size, num_heads, sequence_length, sequence_length]
        attention_matrices = outputs.attentions

        # convert attention matrix of layer 12 to vectors of shape [batch_sz x num_heads x seq_length]
        # we only consider the last (12th) layer, and only consider the first row
        # of the self-attention matrix (represents the extent to which CLS attends to others)
        attention_vectors = attention_matrices[-1][:,:,0]

        # print('attention vectors prior to softmax')
        # print(attention_vectors[0])

        # attention_vectors = torch.nn.Softmax(dim=-1)(attention_vectors)

        # print('attention vectors post softmax')
        # print(attention_vectors[0])
        #
        # print('\n')

        # add implicit dimension to mask_vectors such that it becomes [batch_size, 1, seq_length]
        mask_vectors = mask_vectors.unsqueeze(1)

        # print('vector mask')
        # print(mask_vectors[0])

        # compute impermissible attention tensor of shape [batch_size, num_heads, seq_length]
        impermissible_attention = attention_vectors * mask_vectors

        # print('impermissible_attention before sum')
        # print(impermissible_attention[0])

        # we sum over seq_length dim to get the impermissible attention per head
        impermissible_attention = torch.sum(impermissible_attention, dim=2)

        # print('impermissible_attention after sum over seq length')
        # print(impermissible_attention[0])

        # Compute the complement of impermissible attention, or permissible attention
        permissible_attention = 1 - impermissible_attention

        # print('permissible_attention')
        # print(permissible_attention[0])

        # TODO get rid of nans?
        # log permissible attention per head
        log_permissible_attention = torch.log(permissible_attention)

        # print('log permissible attention')
        # print(log_permissible_attention[0])

        if penalty_fn == 'mean':
            # Compute R value using 'mean' method
            R = - lambeda * torch.mean(log_permissible_attention, dim=1)
            attention_mass = torch.mean(impermissible_attention, dim=1) * 100

        elif penalty_fn == 'max':
            # Compute R value using 'max' method
            R = - lambeda * torch.min(log_permissible_attention, dim=1)[0]
            attention_mass = torch.max(impermissible_attention, dim=1)[0] * 100

        # # compute attention mass:
        # # "the sum of attention values over the set of impermissible tokens averaged over all the examples"
        # attention_mass = torch.mean(torch.mean(impermissible_attention, dim=1))*100

        # print(attention_mass)

        return R, torch.mean(attention_mass).item()


def main(args):

    print('\n -------------- Classification with BERT -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print("\nStandard transformer warning (which you don't need to worry about):\n")

    if config.debug:
        torch.autograd.set_detect_anomaly(True)

    # Logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(dropout=config.dropout,
                      lr=config.lr,
                      penalize=config.penalize,
                      lambeda=config.lambeda,
                      penalty_fn=config.penalty_fn)

    print("\nStandard GPU loading prompt (which you, again, don't need to worry about): \n ") if device_count() > 0 else print('')

    # Main Lightning code
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(gpus=config.gpus,
                      logger=tb_logger,
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=1000)

    dm = GenericDataModule(task=config.task,
                           anonymization=config.anon,
                           max_length=config.max_length,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers)

    # train model for 10 epochs
    trainer.fit(model, dm)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, type=bool,
                        help='toggle elaborate torch errors')

    # Torch / lightning specific args
    num_gpus = 1 if device_count() > 0 else None
    parser.add_argument('--gpus', default=num_gpus)

    accelerator = None if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = os.cpu_count() if device_count() > 0 else None
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    log_every = 10 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between loggings')

    # Learning specific args
    batch_size = 32 if device_count() > 0 else 16
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=180, type=int,
                        help='max no of tokens for tokenizer (default is enough for all tasks')

    # Experiment specific args
    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')
    parser.add_argument('--lambeda', default=1, type=float,
                        help='penalty coefficient')
    parser.add_argument('--penalty_fn', default='mean', type=str,
                        help='penalty fn [options: "mean" or "max" ')

    config = parser.parse_args()

    main(config)