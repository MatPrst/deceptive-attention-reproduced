import argparse
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

LOG_PATH = "logs/"
DATA_PATH = "data/"
DATA_VOCAB_PATH = "data/vocab/"
DATA_MODELS_PATH = "data/models/"


def train_gru(parameters):
    """
    Function for training and testing a bidirectional GRU model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(DATA_MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_VOCAB_PATH, exist_ok=True)

    train_loader = mnist(batch_size=parameters.batch_size,
                         num_workers=args.num_workers)

    # Create a PyTorch Lightning trainer with the generation callback

    # translation_callback = TranslationCallback(save_to_disk=True)
    # inter_callback = InterpolationCallback(save_to_disk=True)
    # callbacks = [translation_callback]
    callbacks = []

    trainer = pl.Trainer(default_root_dir=LOG_PATH,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=parameters.epochs,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=1)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Create model
    pl.seed_everything(parameters.seed)

    model = BiGRU(hidden_dims_gen=parameters.hidden_dims_gen,
                hidden_dims_disc=parameters.hidden_dims_disc,
                dp_rate_gen=parameters.dp_rate_gen,
                dp_rate_disc=parameters.dp_rate_disc,
                lr=parameters.lr)

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader)

    # inter_callback.sample_and_save(trainer, model, epoch=0)

    return model


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)

    parser.add_argument('--attention', dest='attention', type=str, default='dot-product')

    parser.add_argument('--hidden_dims_disc', default=[512, 256],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' +
                             'discriminator. To specify multiple, use " " to ' +
                             'separate them. Example: \"512 256\"')

    parser.add_argument('--dp_rate_gen', default=0.1, type=float,
                        help='Dropout rate in the discriminator')

    parser.add_argument('--dp_rate_disc', default=0.3, type=float,
                        help='Dropout rate in the discriminator')

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

    params = vars(parser.parse_args())

    train_gru(params)
