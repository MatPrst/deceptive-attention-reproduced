import argparse
import os
import random
import time

from torch.utils.tensorboard import SummaryWriter

from data_utils import *
from log_utils import setup_logger
from train_utils import *

LONG_TYPE = torch.LongTensor
FLOAT_TYPE = torch.FloatTensor
if torch.cuda.is_available():
    LONG_TYPE = torch.cuda.LongTensor
    FLOAT_TYPE = torch.cuda.FloatTensor

# PARSING STUFF FROM COMMANDLINE

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--emb-size', dest='emb_size', type=int, default=128,
                    help='number of dimensions for the embedding layer')

parser.add_argument('--hid-size', dest='hid_size', type=int, default=64,
                    help='size of the hidden dimension')

parser.add_argument('--model', dest='model', default='emb-att',
                    choices=('emb-att', 'emb-lstm-att', 'no-att-only-lstm'),
                    help='select the model you want to run')

parser.add_argument('--task', dest='task', default='pronoun',
                    choices=(
                        'pronoun', 'sst', 'sst-wiki', 'sst-wiki-unshuff', 'reco', 'reco-rank', 'de-pronoun', 'de-refs',
                        'de-sst-wiki', 'occupation-classification', 'de-occupation-classification',
                        'occupation-classification_all'),
                    help='select the task you want to run on')

parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=5,
                    help='number of epochs')

parser.add_argument('--num-visualize', dest='num_vis', type=int, default=5,
                    help='number of examples to visualize')

parser.add_argument('--loss-entropy', dest='loss_entropy', type=float, default=0.,
                    help='strength for entropy loss on attention weights')

parser.add_argument('--loss-hammer', dest='loss_hammer', type=float, default=0.,
                    help='strength for hammer loss on attention weights')

parser.add_argument('--loss-kld', dest='loss_kld', type=float, default=0.,
                    help='strength for KL Divergence Loss on attention weights')

parser.add_argument('--top', dest='top', type=int, default=3,
                    help='how many of the most attended words to ignore (default is 3)')

parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='set random seed, defualt = 1')

parser.add_argument('--tensorboard_log', dest='tensorboard_log', default=False, action='store_true')

# flags specifying whether to use the block and attn file or not
parser.add_argument('--use-attn-file', dest='use_attn_file', action='store_true')

parser.add_argument('--use-block-file', dest='use_block_file', action='store_true')

parser.add_argument('--block-words', dest='block_words', nargs='+', default=None,
                    help='list of words you wish to block (default is None)')

parser.add_argument('--dump-attn', dest='dump_attn', action='store_true')

parser.add_argument('--use-loss', dest='use_loss', action='store_true')

parser.add_argument('--anon', dest='anon', action='store_true')

parser.add_argument('--debug', dest='debug', action='store_true')

parser.add_argument('--understand', dest='understand', action='store_true')

parser.add_argument('--flow', dest='flow', action='store_true')

parser.add_argument('--clip-vocab', dest='clip_vocab', action='store_true')

parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=20000,
                    help='in case you clip vocab, specify the vocab size')

params = vars(parser.parse_args())

# useful constants
SEED = params['seed']
TENSORBOARD = params['tensorboard_log']
LOG_PATH = "logs/"

# user specified constants
LOSS_CONFIG = LossConfig(params['loss_hammer'], params['loss_entropy'], params['loss_kld'])
NUM_VIS = params['num_vis']
NUM_EPOCHS = params['num_epochs']
EMB_SIZE = params['emb_size']
HID_SIZE = params['hid_size']
EPSILON = 1e-12
TO_ANON = params['anon']
TO_DUMP_ATTN = params['dump_attn']
BLOCK_TOP = params['top']
BLOCK_WORDS = params['block_words']
USE_ATTN_FILE = params['use_attn_file']
USE_BLOCK_FILE = params['use_block_file']

MODEL_TYPE = params['model']
TASK_NAME = params['task']
USE_LOSS = params['use_loss']
DEBUG = params['debug']
UNDERSTAND = params['understand']
FLOW = params['flow']
CLIP_VOCAB = params['clip_vocab']
VOCAB_SIZE = params['vocab_size']

# create required folders if not present
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_MODELS_PATH, exist_ok=True)

# SETUP LOGGING
LOGGER = setup_logger(LOG_PATH, f"task={TASK_NAME}__model={MODEL_TYPE}_hammer={LOSS_CONFIG.c_hammer}_seed={SEED}")
LOGGER.info(f"Task: {TASK_NAME}")
LOGGER.info(f"Model: {MODEL_TYPE}")
LOGGER.info(f"Coef (hammer): {LOSS_CONFIG.c_hammer:0.2f}")
LOGGER.info(f"Coef (random-entropy): {LOSS_CONFIG.c_entropy:0.2f}")
LOGGER.info(f"Seed: {SEED}\n")

set_seed(SEED)

# READING THE DATA

TRAIN, DEV, TEST, VOCABULARY = read_data(TASK_NAME, MODEL_TYPE, LOGGER, CLIP_VOCAB, BLOCK_WORDS, TO_ANON, VOCAB_SIZE,
                                         USE_BLOCK_FILE, USE_ATTN_FILE)

if DEBUG:
    TRAIN = TRAIN[:100]
    DEV = DEV[:100]
    TEST = TEST[:100]

LOGGER.info(f"The source vocabulary size / input_dim is {VOCABULARY.n_words}")
LOGGER.info(f"The target vocabulary size / output_dim is {VOCABULARY.n_tags}")

current_model = get_model(MODEL_TYPE, VOCABULARY, EMB_SIZE, HID_SIZE)
calc_ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(current_model.parameters())

LOGGER.info(f"\nEvaluating without any training ...")
LOGGER.info(f"ITER: {0}")
_, _ = evaluate(current_model, TEST, VOCABULARY, LOSS_CONFIG, UNDERSTAND, FLOW, LOGGER,
                stage='test', attn_stats=True, num_vis=0)

WRITER = None
if TENSORBOARD:
    WRITER = SummaryWriter(os.path.join(LOG_PATH, "tensorboard"))

LOGGER.info("Starting to train. \n")

best_dev_accuracy = 0.
best_dev_loss = np.inf
best_test_accuracy = 0.
best_epoch = 0

for ITER in range(1, NUM_EPOCHS + 1):
    LOGGER.info(f"ITER: {ITER}")

    random.shuffle(TRAIN)
    train_loss = 0.0
    train_ce_loss = 0.0
    train_entropy_loss = 0.0
    train_hammer_loss = 0.0
    train_kld_loss = 0.0

    start = time.time()
    for num, (idx, words_orig, block_ids, attn_orig, tag) in enumerate(TRAIN):

        words = torch.tensor([words_orig]).type(LONG_TYPE)
        tag = torch.tensor([tag]).type(LONG_TYPE)
        if attn_orig is not None:
            attn_orig = torch.tensor(attn_orig).type(FLOAT_TYPE)

        # forward pass
        out, attns = current_model(words)
        attention = attns[0]

        ce_loss = calc_ce_loss(out, tag)
        entropy_loss = calc_entropy_loss(attention, LOSS_CONFIG.c_entropy)
        hammer_loss = calc_hammer_loss(words_orig, attention,
                                       block_ids, LOSS_CONFIG.c_hammer)

        kld_loss = calc_kld_loss(attention, attn_orig, LOSS_CONFIG.c_kld)

        loss = ce_loss + entropy_loss + hammer_loss + kld_loss
        train_loss += loss.item()

        train_ce_loss += ce_loss.item()
        train_entropy_loss += entropy_loss.item()
        train_hammer_loss += hammer_loss.item()
        train_kld_loss += kld_loss.item()

        # TODO: add tensorboard summary writer
        # logger.info("ID: %4d\t CE: %0.4f\t ENTROPY: %0.4f\t HAMMER: %0.4f\t KLD: %.4f\t TOTAL: %0.4f" %(
        #     num,
        #     ce_loss.item(),
        #     entropy_loss.item(),
        #     hammer_loss.item(),
        #     kld_loss.item(),
        #     loss.item()
        # ), end='\r')

        # update the params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_duration = time.time() - start

    len_train_set = len(TRAIN)
    avg_train_loss = train_loss / len_train_set
    avg_train_ce_loss = train_ce_loss / len_train_set
    avg_train_entropy_loss = train_entropy_loss / len_train_set
    avg_train_hammer_loss = train_hammer_loss / len_train_set
    avg_train_kld_loss = train_kld_loss / len_train_set

    LOGGER.info("train loss=%.4f, ce_loss=%.4f, entropy_loss=%.4f, hammer_loss=%.4f, kld_loss==%.4f, time=%.2fs\n"
                % (avg_train_loss, avg_train_ce_loss, avg_train_entropy_loss, avg_train_hammer_loss, avg_train_kld_loss,
                   epoch_duration))

    train_acc, train_loss = evaluate(current_model, TRAIN, VOCABULARY, LOSS_CONFIG, logger=LOGGER, stage='train')
    dev_acc, dev_loss = evaluate(current_model, DEV, VOCABULARY, LOSS_CONFIG, logger=LOGGER, stage='dev',
                                 attn_stats=True)
    test_acc, test_loss = evaluate(current_model, TEST, VOCABULARY, LOSS_CONFIG, logger=LOGGER, stage='test',
                                   attn_stats=True, num_vis=NUM_VIS)

    if WRITER is not None:
        # Training metrics
        WRITER.add_scalar("Loss/train", avg_train_loss, ITER)
        WRITER.add_scalar("CE_loss/train", avg_train_ce_loss, ITER)
        WRITER.add_scalar("Entropy_loss/train", avg_train_entropy_loss, ITER)
        WRITER.add_scalar("Hammer_loss/train", avg_train_hammer_loss, ITER)
        WRITER.add_scalar("KLD_loss/train", avg_train_kld_loss, ITER)
        WRITER.add_scalar("Duration", epoch_duration, ITER)

        # Evaluation metrics
        WRITER.add_scalar("Accuracy/train", train_acc, ITER)
        WRITER.add_scalar("Accuracy/dev", dev_acc, ITER)
        WRITER.add_scalar("Accuracy/test", test_acc, ITER)

    if ((not USE_LOSS) and dev_acc > best_dev_accuracy) or (USE_LOSS and dev_loss < best_dev_loss):

        if USE_LOSS:
            best_dev_loss = dev_loss
        else:
            best_dev_accuracy = dev_acc
        best_test_accuracy = test_acc
        best_epoch = ITER

        if TO_DUMP_ATTN:
            # log.pr_bmagenta("dumping attention maps")
            LOGGER.info("dumping attention maps")
            dump_attention_maps(current_model, TRAIN, DATA_PREFIX + "train.txt.attn." + MODEL_TYPE)
            dump_attention_maps(current_model, DEV, DATA_PREFIX + "dev.txt.attn." + MODEL_TYPE)
            dump_attention_maps(current_model, TEST, DATA_PREFIX + "test.txt.attn." + MODEL_TYPE)

    LOGGER.info(f"best test accuracy = {best_test_accuracy:0.4f} attained after epoch = {best_epoch}\n")

    # save the trained model
    LOGGER.info("Saving trained model.\n")
    torch.save(current_model.state_dict(), get_model_path(LOSS_CONFIG, best_epoch, MODEL_TYPE, SEED, TASK_NAME))
