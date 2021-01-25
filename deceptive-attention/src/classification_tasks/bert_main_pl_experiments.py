# Global modules
import argparse
import sys
import time
import os
import warnings
import logging

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
from transformers_editted.src.transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

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


def get_lowest(path_dict, baseline):
    """ This beautiful function takes a dict of model paths and
    returns the path with the lowest dev attention mass """
    am_values = []  # save the dev AM values in a list
    acc_values = []  # save the dev acc values in a list
    for path in path_dict.keys():  # Iterate over the k paths

        # Behold the most robustest of parsing technologies known to mankind
        left_idx = path.find('val_attention_mass=') + len('val_attention_mass=')
        right_idx = path.find('.ckpt')
        am_value = path[left_idx:right_idx]
        am_values.append(float(am_value))

        left_idx = path.find('val_acc=') + len('val_acc=')
        right_idx = path.find('-val_attention_mass=')
        acc_value = path[left_idx:right_idx]
        acc_values.append(float(acc_value))

    eligible = [] # keep track of checkpoints whose test accuracies are /geq 0.02
    for i, path in enumerate(path_dict.keys()):
        difference = abs(acc_values[i] - baseline)
        if difference <= 0.02 or acc_values[i] > baseline:
            eligible.append(i)

    # from all models that are within the 0.02% range, we pick the one with the greatest
    # reduction in AM, i.e. the model with the smallest attention mass
    if len(eligible) > 0:
        smallest = 999.0
        smallest_idx = None
        for i, path in enumerate(path_dict.keys()):
            if i in eligible:
                if am_values[i] < smallest:
                    smallest = am_values[i]
                    smallest_idx = i
        # return the path of the model with the smallest AM
        for i, path in enumerate(path_dict.keys()):
            if i == smallest_idx:
                return path

    # if all of the checkpoints are below the baseline we stick to the best dev acc chkpt
    else:
        print('None of the models are within 2% acc range')
        return None

def del_checkpoints(path_dict, best_model_path):
    """
    This function deletes redundant checkpoint files
    after the train-val-test loop is finished
    """
    for path in path_dict.keys():
        if path != best_model_path:
            os.remove(path)

def main(args):

    print('\n -------------- Classification with BERT ------------------------------------ \n')

    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))

    # Set global Lightning seed
    seed_everything(config.seed)

    # This mode turns on more detailed torch error descriptions (disabled by default)
    if config.debug:
        torch.autograd.set_detect_anomaly(True)

    # Turn off GPU available prompts for less cluttered console output (disabled by default)
    if config.warnings == False:

        warnings.filterwarnings('ignore')
        # configure logging at the root level of lightning to get rid of GPU/TPU availability prompts
        logging.getLogger('lightning').setLevel(0)

    # -------------------- Code for replicating all 7 experiments for a given task ------------------------ #
    # time the total duration of the experiments
    start = time.time()

    # for a given task and seed, there is a single 'anon' experiment, and there are 6 'adversarial' experiments.
    # we simply run the 7 experiments one after another.
    for mode in ['anon', 'adversarial']:

        if mode == 'anon':

            ############################### Code for replicating anonymization experiment ############################
            print('\n -------------- Beginning Anonymization experiment for task: {} ------------\n'.format(config.task))

            # Define model
            model = BERTModel(dropout=config.dropout,
                              lr=config.lr,
                              penalize=False,
                              lambeda=0,
                              penalty_fn='mean')

            # Define logger and path
            logger = pl_loggers.TensorBoardLogger('experiment_results/logs/seed_{}/task_{}/anon/'.format(config.seed, config.task))
            logger.log_hyperparams(config)


            # for the anonymization task, we want to test using the ckpt with the best dev accuracy
            # therefore we define a dedicated chkpt callback that monitors the val_acc metric
            checkpoint_callback = ModelCheckpoint(
                        monitor='val_acc',
                        dirpath='experiment_results/checkpoints/seed_{}/task_{}/anon/'.format(config.seed, config.task),
                        filename='model-{epoch:02d}-{val_acc:.2f}',
                        save_top_k=1,
                        mode='max')

            dm = GenericDataModule(task=config.task,
                                   anonymization=True,
                                   max_length=config.max_length,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)

            trainer = Trainer(gpus=config.gpus,
                              logger=logger,
                              callbacks=[checkpoint_callback],
                              log_every_n_steps=config.log_every,
                              accelerator=config.accelerator,
                              max_epochs=config.max_epochs,
                              limit_train_batches=config.toy_run,  # if toy_run=1, we only train for a single batch
                              limit_test_batches=config.toy_run,  # across all the splits, which is useful when debugging
                              limit_val_batches=config.toy_run, # (default arg is None)
                              progress_bar_refresh_rate=config.progress_bar,
                              weights_summary=None) # don't print a summary

            # train model
            trainer.fit(model, dm)
            # load checkpoint with best dev accuracy
            checkpoint_callback.best_model_path
            # evaluate on test set
            print('Test results on {} with seed {} with anonymization: '.format(config.task, config.seed))
            result = trainer.test()

        if mode == 'adversarial':

            # ----------------------------  Code for replicating adversarial experiments ---------------------------- #
            print('\n -------------- Beginning adversarial experiments for task: {} -------------- \n'.format(config.task))

            # for the 'adversarial' models, there are 2 x 3 = 6 possible experiments that need to be ran.
            penalty_fns = ['mean', 'max']

            lambdas = [0, 0.1, 1.0]

            # run experiments for both penalty fns
            for penalty_fn in penalty_fns:

                # given a penalty fn, run experiments for all values of lambda
                for lambeda in lambdas:

                    print('Training for penalty_fn = {} and lambda = {}...'.format(penalty_fn, lambeda))

                    # Define model
                    model = BERTModel(dropout=config.dropout,
                                       lr=config.lr,
                                       penalize=True,
                                       lambeda=lambeda,
                                       penalty_fn=penalty_fn)

                    # Specify logger and path
                    logger = pl_loggers.TensorBoardLogger('experiment_results/logs/seed_{}/task_{}/penalty_{}_lambda_{}/'.format(
                                                            config.seed, config.task, penalty_fn, lambeda))
                    logger.log_hyperparams(config)

                    # For lambda 0 (the baseline), we checkpoint based on dev accuracy
                    checkpoint_callback = ModelCheckpoint(
                                          monitor='val_acc',
                                          dirpath='experiment_results/checkpoints/seed_{}/task_{}/penalty_{}_lambda_{}/'.format(
                                              config.seed, config.task, penalty_fn, lambeda),
                                          filename='model-{epoch:02d}-{val_acc:.2f}-{val_attention_mass:.2f}',
                                          save_top_k=10,
                                          mode='max', )

                    # Initialise DataModule
                    dm = GenericDataModule(task=config.task,
                                           anonymization=False,
                                           max_length=config.max_length,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers)

                    # Initialise Trainer
                    trainer = Trainer(gpus=config.gpus,
                                      logger=logger,
                                      callbacks=[checkpoint_callback],
                                      log_every_n_steps=config.log_every,
                                      accelerator=config.accelerator,
                                      max_epochs=config.max_epochs,
                                      limit_train_batches=config.toy_run,
                                      limit_test_batches=config.toy_run,
                                      limit_val_batches=config.toy_run,
                                      progress_bar_refresh_rate=config.progress_bar,
                                      weights_summary=None) # don't print a summary

                    # Train model
                    trainer.fit(model, dm)

                    # ---------------------------- Model Selection Logic ---------------------------------------

                    # For the lambda = 0 baseline, we evaluate on test set using ckpt with highest dev accuracy
                    # For the lambda!= 0 models, we consider all models whose dev acc is within 2% range of baseline acc
                    # From those models, we pick the model with the lowest dev attention mass
                    path_dict = checkpoint_callback.best_k_models # extract a dictionary of all k checkpoint paths
                    if lambeda != 0:    # comparison only holds for lambda 0.1, 1.0

                        # access baseline test accuracy
                        key = penalty_fn + '_test_acc'
                        baseline_acc = results_dict[key]

                        # def get_lowest(path_dict):
                        #     """ This function takes a dict of model paths and
                        #     returns the path with the lowest dev attention mass"""
                        #     values = [] # save the dev AM values in a list
                        #     for path in path_dict.keys(): # Iterate over the k paths
                        #         value = path[-10:-5] # Extract the Dev AM score from the path str
                        #         values.append(float(value)) # Add dev AM score to list
                        #     min_idx = values.index(min(values)) # Get idx of smallest dev AM value
                        #     for i, path in enumerate(path_dict.keys()):
                        #         if i == min_idx: # return path with lowest dev AM
                        #             return path

                        # obtain ckpt path with lowest dev AM
                        best_model_path = get_lowest(path_dict, baseline_acc) # obtain path with lowest dev AM
                        if best_model_path is not None: # we only overwrite the path if there is a checkpoint within 2% acc
                            checkpoint_callback.best_model_path = best_model_path

                    print(checkpoint_callback.best_model_path)

                    # (re)set best model path for testing
                    checkpoint_callback.best_model_path

                    # Evaluate on test set
                    print('Test results on task={} for model with penalty_fn={}, lambda={}: '.format(
                        config.task, penalty_fn, lambeda))
                    result = trainer.test()

                    # if lambda = 0, we want to store the baseline test acc for later use
                    result_dict = result[0]
                    if lambeda == 0:
                        results_dict = {}
                        key = penalty_fn + '_test_acc'
                        results_dict[key] = result[0]['test_acc']

                    # logic to delete files that are no longer needed
                    best_model_path = checkpoint_callback.best_model_path # determine which checkpoint to keep
                    del_checkpoints(path_dict, best_model_path) # remove other redundant checkpoints

    end = time.time()
    print("\n -------------- Finished running experiments --------------")
    elapsed = end - start
    print('Required time to run all experiments: {} seconds '.format(elapsed))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Experiment specific args
    parser.add_argument('--seed', default=42, type=int,
                        help='specifies global seed')
    parser.add_argument('--task', default='occupation', type=str,
                        help='str to specify task. Args: [occupation, pronoun, sstwiki]')

    # Optimization specific args
    batch_size = 32 if device_count() > 0 else 16 # some CLI args are assigned default values <-> a GPU is available
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='no. of epochs to train for')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=180, type=int,
                        help='max no of tokens for tokenizer (default is enough for all tasks)')

    # Torch / lightning specific args
    num_gpus = 1 if device_count() > 0 else None
    parser.add_argument('--gpus', default=num_gpus, type=int)

    accelerator = None if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = 12 if device_count() > 0 else 1
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    log_every = 10 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between loggings')

    # Auxiliary args
    parser.add_argument('--debug', default=False, type=bool,
                        help='toggle elaborate torch errors')

    toy_run = 1 if device_count() == 0 else 1.0
    parser.add_argument('--toy_run', default=toy_run, type=float,
                        help='set no of batches per datasplit per epoch (helpful for debugging)')

    progress_bar = 0 if device_count() > 0 else 1
    parser.add_argument('--progress_bar', default=progress_bar, type=int,
                        help='lightning progress bar flag. disabled on GPU to keep SLURM output neat')
    parser.add_argument('--warnings', default=False, type=bool,
                        help='disable warnings for less cluttered console output')

    config = parser.parse_args()

    # if toy_run is enabled, set a batch size of 8 for quicker epochs
    config.batch_size = 1 if type(config.toy_run) == int else config.batch_size

    main(config)
