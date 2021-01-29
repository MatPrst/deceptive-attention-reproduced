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
from pytorch_lightning.callbacks import ModelCheckpoint

# Local dependencies
from bert_util_pl import GenericDataModule

# Local transformer dependencies
from transformers_editted.src.transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers_editted.src.transformers.models.auto.tokenization_auto import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false" # A warning made me do it.

class BERTModel(LightningModule):
    """
    This class implements the attention-manipulation model with BERT in Pytorch Lightning.
    """
    def __init__(self, dropout, lr, penalize, lambeda, penalty_fn):
        super().__init__()
        """
        Args:
            penalize: flag to toggle attention manipulation and information flow restriction
            tokenizer: transformers object used to convert raw text to tensors
        """
        self.dropout = dropout # dropout applied to BERT
        self.lr = lr # learning rate
        self.penalize = penalize # flag to determine whether L = L or L = L + R
        self.lambeda = lambeda # lambda param
        self.penalty_fn = penalty_fn # str which specifies which penalty fn is used
        self.accuracy = metrics.Accuracy() # for logging to lightning
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #
        # load pre-trained, uncased, sequence-classification BERT model
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                     num_labels=2,
                                                                     hidden_dropout_prob=self.dropout)

    def configure_optimizers(self):
        "This method handles optimization of params for PyTorch lightning"
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, x, attention_mask, labels, mask_matrices):
        "This method defines how the data is passed through the net."
        output = self.encoder(x,
                              labels=labels,
                              attention_mask=attention_mask,
                              matrix_mask=mask_matrices,
                              output_attentions=True)
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
        self.log('train_acc', self.accuracy(preds, labels))

        return loss

    def validation_step(self, batch, batch_idx):
        """
        This method implements the validation step in PyTorch Lightning
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
        self.log('val_penalty_R', R)
        self.log('val_attention_mass', attention_mass)

        preds = outputs.logits

        # Log dev stats to Tensorboard
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(preds, labels))

        return None

    def test_step(self, batch, batch_idx):
        """
        This method implements the test step in PyTorch Lightning
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
        self.log('test_penalty_R', R)
        self.log('test_attention_mass', attention_mass)

        preds = outputs.logits

        # Log dev stats to Tensorboard
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(preds, labels))

        return None

    def compute_R(self, outputs, mask_vectors, lambeda, penalty_fn):
        """
        This method computes the R component, which serves as the penalizing mechanism as described in the paper
        """
        # outputs.attentions contains a tuple of size 12 (one self-attention map per layer)
        # tuple contains tensor of shape [batch_size, num_heads, sequence_length, sequence_length]
        attention_matrices = outputs.attentions

        # we only consider the last (12th) layer, and only consider the first row of the self-attn matrices
        # (this row represents the extent to which the CLS token attends to others)
        attention_vectors = attention_matrices[-1][:,:,0,:]

        # add implicit dimension to mask_vectors such that it becomes [batch_size, 1, seq_length]
        mask_vectors = mask_vectors.unsqueeze(1)

        # compute impermissible attention tensor of shape [batch_size, num_heads, seq_length]
        impermissible_attention = attention_vectors * mask_vectors
        # we sum over seq_length dim to get the impermissible attention per head
        impermissible_attention = torch.sum(impermissible_attention, dim=2)

        # For some miraculous reason, the attention_probs from BERT occasionally
        # exceed 1.00, which introduce nans with the torch.log() hereafter.
        # Therefore, we re-set values that exceed > 1 to a value just below 1.
        impermissible_attention[impermissible_attention[:,:] > 1] = 0.999

        # Compute the complement of impermissible attention, or permissible attention
        permissible_attention = 1 - impermissible_attention

        # log permissible attention per head
        log_permissible_attention = torch.log(permissible_attention)

        if penalty_fn == 'mean':
            # Compute R value using 'mean' method
            R = - lambeda * torch.mean(log_permissible_attention, dim=1)
            attention_mass = torch.mean(impermissible_attention, dim=1) * 100

        elif penalty_fn == 'max':
            # Compute R value using 'max' method
            R = - lambeda * torch.min(log_permissible_attention, dim=1)[0]
            attention_mass = torch.max(impermissible_attention, dim=1)[0] * 100

        return R, torch.mean(attention_mass).item()

def main(args):

    print('\n -------------- Classification with BERT -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    # Set global seed
    seed_everything(config.seed)

    if config.debug: # This mode turns on more detailed torch error descriptions
        torch.autograd.set_detect_anomaly(True)

    print("\nStandard transformer warning (which you don't need to worry about):\n")

    # Logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(dropout=config.dropout,
                      lr=config.lr,
                      penalize=config.penalize,
                      lambeda=config.lambeda,
                      penalty_fn=config.penalty_fn)

    print("\nStandard GPU loading prompt (which you, again, don't need to worry about): \n ") if device_count() > 0 else print('')

    # Main Lightning code
    tb_logger = pl_loggers.TensorBoardLogger('experiment_results/logs/seed_{}/task_{}/'.format(config.seed, config.task))
    tb_logger.log_hyperparams(config)

    """ ----------------------------------------------------------------------------------------------------------- """

    # for the anonymization task, we want to test using the ckpt with the best dev accuracy
    # therefore we define a dedicated chkpt callback:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='experiment_results/checkpoints/seed_{}/task_{}'.format(config.seed, config.task),
        filename='model-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    dm = GenericDataModule(task=config.task,
                           anonymization=config.anon,
                           max_length=config.max_length,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers)

    trainer = Trainer(gpus=config.gpus,
                      logger=tb_logger,
                      callbacks=[checkpoint_callback],
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=config.max_epochs,
                      limit_train_batches=config.toy_run, # if toy_run=1, we only train for a single batch
                      limit_test_batches=config.toy_run, # across all the splits, which is useful when debugging
                      limit_val_batches=config.toy_run) # (default arg is None)


    # train model for 10 epochs
    trainer.fit(model, dm)

    # load best checkpoint
    checkpoint_callback.best_model_path

    # run test set
    result = trainer.test()
    print(result)

    # idea:
    # do run with anonymization=True
    # do run to get baseline
    # do run with lambda 0.1
    # do run with lambda 1.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Experiment specific args
    parser.add_argument('--seed', default=42, type=int,
                        help='specifies global seed')
    parser.add_argument('--task', default='occupation', type=str,
                        help='str to specify task. Args: [occupation, pronoun, sstwiki]')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized impermissible tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')
    parser.add_argument('--penalty_fn', default='mean', type=str,
                        help='penalty fn [options: "mean" or "max" ')
    parser.add_argument('--lambeda', default=1, type=float,
                        help='penalty coefficient')

    # Learning specific args
    batch_size = 32 if device_count() > 0 else 16
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='no. of epochs to train for')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=180, type=int,
                        help='max no of tokens for tokenizer (default is enough for all tasks')

    # Torch / lightning specific args
    num_gpus = 1 if device_count() > 0 else None
    parser.add_argument('--gpus', default=num_gpus)

    accelerator = None if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = os.cpu_count() if device_count() > 0 else 1
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    log_every = 10 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between loggings')

    # Auxiliary args
    parser.add_argument('--debug', default=False, type=bool,
                        help='toggle elaborate torch errors')
    parser.add_argument('--toy_run', default=None, type=int,
                        help='set no of batches per datasplit per epoch (helpful for debugging)')
    config = parser.parse_args()

    # if toy_run is enabled, set a batch size of 1 for quicker epochs
    config.batch_size = 8 if config.toy_run is not None else config.batch_size

    main(config)