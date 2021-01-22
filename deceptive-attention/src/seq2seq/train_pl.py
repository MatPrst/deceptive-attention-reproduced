import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
from batch_utils import SentenceDataModule, TRG_LANG, SRC_LANG
from log_utils import setup_logger
from model import BiGRU
from utils import *

LOG_PATH = "logs/"
BIGRU_LOGS = LOG_PATH + 'BiGRU'
DATA_PATH = "data/"
DATA_VOCAB_PATH = "data/vocab/"
DATA_MODELS_PATH = "data/models/"

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = utils.PAD_token
SOS_IDX = utils.SOS_token
EOS_IDX = utils.EOS_token


class TranslationCallback(pl.Callback):

    def __init__(self, test_loader, out_path, logger, every_n_epochs=10, save_to_disk=False):
        """
        Callback for translating sentences to TensorBoard and/or save them to disk every N epochs across training.
        Inputs:
            batch_size          - Number of sentences to be translated.
            every_n_epochs      - Only translate those sentences every N epochs (otherwise tensorboard gets quite large)
            save_to_disk        - If True, the samples should be saved to disk as well.
        """
        super().__init__()
        self.test_loader = test_loader
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk
        self.out_path = out_path
        self.logger = logger

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.logger.info('Generating initial translations ..........')
        self.generate(trainer, pl_module, 0)

    def teardown(self, trainer, pl_module, stage):
        self.logger.info('Generating final translations ..........')
        self.generate(trainer, pl_module, trainer.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.logger.info(f'Generating translations at epoch {trainer.current_epoch + 1} ..........')
            self.generate(trainer, pl_module, trainer.current_epoch + 1)

    def generate(self, trainer, pl_module, epoch):
        """
        Function that generates translations and saves them from the BiGRU.
        The generated samples should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer     - The PyTorch Lightning "Trainer" object.
            pl_module   - The BiGRU model that is currently being trained.
            epoch       - The epoch number to use for TensorBoard logging
                          and saving of the files.
        """

        # print('Translation: BiGRU Model on device: ', pl_module.device)
        # print('Translation: Seq2Seq Model on device: ', pl_module.model.device)

        translations, targets = pl_module.translate(self.test_loader)
        bleu_score = bleu_score_corpus(targets, translations, self.logger) * 100

        self.logger.info(f'BLEU score: {bleu_score}\n')
        pl_module.log("bleu_score", bleu_score)
        # trainer.logger.experiment.add_scalar("bleu_score", bleu_score, global_step=epoch)

        fw = open(f"{self.out_path}.test.out", 'w')
        for line in translations:
            fw.write(line.strip() + "\n")
        fw.close()


def train_gru(parameters):
    """
    Function for training and testing a bidirectional GRU model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(BIGRU_LOGS, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(DATA_MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_VOCAB_PATH, exist_ok=True)

    task = parameters.task
    batch_size = parameters.batch_size
    num_train = parameters.num_train
    debug = parameters.debug
    epochs = parameters.epochs
    seed = parameters.seed
    attention = parameters.attention
    coeff = parameters.loss_coeff

    logger = setup_logger(LOG_PATH, 'task=%s_coeff=%s_seed=%s' % (task, coeff, seed))

    logger.info(f'Configuration:\n epochs: {epochs}\n coeff: {coeff}\n seed: {seed}\n batch_size: ' +
                f'{batch_size}\n attention: {attention}\n debug: {debug}\n num_train: {num_train}\n '
                f'task: {task}\n')

    logger.info('Initializing data module ..........')

    data_module = SentenceDataModule(task=task,
                                     batch_size=batch_size,
                                     num_train=num_train,
                                     debug=debug)
    data_module.setup()

    # Create a PyTorch Lightning trainer with the generation callback

    translation_callback = None
    if task == 'en-de':
        logger.info('Creating translation callback ..........\n')
        translation_callback = TranslationCallback(data_module.test_dataloader(batch_size=1),
                                                   out_path=get_out_path(parameters),
                                                   logger=logger,
                                                   save_to_disk=True,
                                                   every_n_epochs=2)

    cuda_available = torch.cuda.is_available()
    print(cuda_available)

    trainer = Trainer(default_root_dir=BIGRU_LOGS,
                      # logger=pl_loggers.TensorBoardLogger(LOG_PATH),
                      checkpoint_callback=ModelCheckpoint(save_weights_only=True),
                      gpus=1 if cuda_available else 0,
                      max_epochs=epochs,
                      callbacks=[translation_callback],
                      progress_bar_refresh_rate=1
                      # ,accelerator=pl.accelerators.gpu_accelerator.GPUAccelerator
                      )

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Create model
    pl.seed_everything(seed)

    model = BiGRU(input_dim=SRC_LANG.get_vocab_size(),
                  output_dim=TRG_LANG.get_vocab_size(),
                  encoder_hid_dim=ENC_HID_DIM,
                  decoder_hid_dim=DEC_HID_DIM,
                  encoder_emb_dim=ENC_EMB_DIM,
                  decoder_emb_dim=DEC_EMB_DIM,
                  encoder_dropout=ENC_DROPOUT,
                  decoder_dropout=DEC_DROPOUT,
                  attention_type=attention,
                  pad_idx=PAD_IDX,
                  sos_idx=SOS_IDX,
                  eos_idx=EOS_IDX,
                  coeff=coeff,
                  decode_with_no_attention=parameters.no_attn_inference)

    logger.info('\nInitializing model ..........')
    logger.info(f'The model has {model.count_parameters():,} trainable parameters.\n')

    # Training

    # if translation_callback is not None:
    #     logger.info('Generating initial translations ..........')
    #     translation_callback.generate(trainer, model, epoch=0)

    logger.info('Fitting model ..........')
    trainer.fit(model, data_module)

    # Testing
    model = BiGRU.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=data_module.test_dataloader(), verbose=True)

    logger.info(f'Test Result: {test_result}')
    # logger.info(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc:0.2f} \
    #             |  Test Attn Mass: {test_attn_mass:0.2f} |  Test PPL: {math.exp(test_loss):7.3f}')
    #
    # logger.info(f"Final Test Accuracy ..........\t{test_acc:0.2f}")
    # logger.info(f"Final Test Attention Mass ....\t{test_attn_mass:0.2f}")
    # logger.info(f"Convergence time in seconds ..\t{convergence_time:0.2f}")
    # logger.info(f"Sample efficiency in epochs ..\t{epochs_taken_to_converge}")

    # save the vocabulary
    out_path = f"{DATA_VOCAB_PATH}{task}{get_suffix(attention)}_seed={str(seed)}_coeff={str(coeff)}_num-train={str(num_train)}"
    SRC_LANG.save_vocab(f"{out_path}.src.vocab")
    TRG_LANG.save_vocab(f"{out_path}.trg.vocab")

    return model


def get_out_path(parameters):
    suffix = get_suffix(parameters.attention)
    return f"{DATA_VOCAB_PATH}{parameters.task}{suffix}_seed={str(parameters.seed)}_coeff=" \
           f"{str(parameters.loss_coeff)}_num-train={str(parameters.num_train)}"


def get_suffix(attention):
    suffix = ''
    if attention == 'uniform':
        suffix = "_uniform"
    elif attention == 'no_attention':
        suffix = "_no-attn"
    return suffix


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)

    parser.add_argument('--attention', dest='attention', type=str, default='dot-product')

    # Optimizer hyperparameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)

    # Other hyperparameters
    parser.add_argument('--task', dest='task', default='en-de',
                        choices=('copy', 'reverse-copy', 'binary-flip', 'en-hi', 'en-de'),
                        help='select the task you want to run on')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--epochs', dest='epochs', type=int, default=5)

    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    parser.add_argument('--num-train', dest='num_train', type=int, default=1000000)

    parser.add_argument('--decode-with-no-attn', dest='no_attn_inference', action='store_true')

    parser.add_argument('--tensorboard_log', dest='tensorboard_log', action='store_true')

    params = parser.parse_args()

    train_gru(params)
