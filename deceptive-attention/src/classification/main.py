import argparse
import os
import random
import time

import torch.nn as nn
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from data_utils import *
from log_utils import setup_logger
from train_utils import *
from train_utils import LONG_TYPE, FLOAT_TYPE

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
DATA_MODELS_PATH = "data/models/"

# user specified constants
C_ENTROPY = params['loss_entropy']
C_HAMMER = params['loss_hammer']
C_KLD = params['loss_kld']
NUM_VIS = params['num_vis']
NUM_EPOCHS = params['num_epochs']
EMB_SIZE = params['emb_size']
HID_SIZE = params['hid_size']
EPSILON = 1e-12
TO_ANON = params['anon']
print(TO_ANON)
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
LOGGER = setup_logger(LOG_PATH, f"task={TASK_NAME}__model={MODEL_TYPE}_hammer={C_HAMMER}_seed={SEED}")
LOGGER.info(f"Task: {TASK_NAME}")
LOGGER.info(f"Model: {MODEL_TYPE}")
LOGGER.info(f"Coef (hammer): {C_HAMMER:0.2f}")
LOGGER.info(f"Coef (random-entropy): {C_ENTROPY:0.2f}")
LOGGER.info(f"Seed: {SEED}")

set_seed(SEED)

W2I = defaultdict(lambda: len(W2I))
W2C = defaultdict(lambda: 0.0)  # word to count
T2I = defaultdict(lambda: len(T2I))
UNK = W2I["<unk>"]


def evaluate(model, dataset, i2w, i2t, logger=LOGGER, stage='test', attn_stats=False, num_vis=0):
    logger.info(f"Evaluating on {stage} set.\n")

    # Perform testing
    test_correct = 0.0
    test_base_prop = 0.0
    test_attn_prop = 0.0
    test_base_emb_norm = 0.0
    test_attn_emb_norm = 0.0
    test_base_h_norm = 0.0
    test_attn_h_norm = 0.0

    example_data = []

    total_loss = 0.0
    if num_vis > 0 and UNDERSTAND:
        wts, bias = model.get_linear_wts()
        logger.info("Weights below")
        logger.info(wts.detach().cpu().numpy())
        logger.info("bias below")
        logger.info(bias.detach().cpu().numpy())

    for idx, words, block_ids, attn_orig, tag in dataset:
        words_t = torch.tensor([words]).type(LONG_TYPE)
        tag_t = torch.tensor([tag]).type(LONG_TYPE)
        if attn_orig is not None:
            attn_orig = torch.tensor(attn_orig).type(FLOAT_TYPE)

        block_ids_t = torch.tensor([block_ids]).type(FLOAT_TYPE)

        if stage == 'test' and FLOW:
            pred, attn = model(words_t, block_ids_t)
        else:
            pred, attn = model(words_t)
        attention = attn[0]

        if not FLOW or (stage != 'test'):
            assert 0.99 < torch.sum(attention).item() < 1.01

        ce_loss = calc_ce_loss(pred, tag_t)
        entropy_loss = calc_entropy_loss(attention, C_ENTROPY)
        hammer_loss = calc_hammer_loss(words, attention, block_ids, C_HAMMER)
        kld_loss = calc_kld_loss(attention, attn_orig, C_KLD)

        assert hammer_loss.item() >= 0.0
        assert ce_loss.item() >= 0.0

        loss = ce_loss + entropy_loss + hammer_loss
        total_loss += loss.item()

        word_embeddings = model.get_embeddings(words_t)
        word_embeddings = word_embeddings[0].detach().cpu().numpy()
        assert len(words) == len(word_embeddings)

        final_states = model.get_final_states(words_t)
        final_states = final_states[0].detach().cpu().numpy()
        assert len(words) == len(final_states)

        predict = pred[0].argmax().item()
        if predict == tag:
            test_correct += 1

        if idx < num_vis:

            attn_scores = attn[0].detach().cpu().numpy()

            example_data.append([[i2w[w] for w in words], attn_scores, i2t[predict], i2t[tag]])

            if UNDERSTAND:
                headers = ['words', 'attn'] + ['e' + str(i + 1) for i in range(EMB_SIZE)]
                tabulated_list = []
                for j in range(len(words)):
                    temp_list = [i2w[words[j]], attn_scores[j]]
                    for emb in word_embeddings[j]:
                        temp_list.append(emb)
                    tabulated_list.append(temp_list)
                logger.info(tabulate(tabulated_list, headers=headers))

        base_prop, attn_prop = quantify_attention(words, attention.detach().cpu().numpy(), block_ids)
        base_emb_norm, attn_emb_norm = quantify_norms(words, word_embeddings, block_ids)
        base_h_norm, attn_h_norm = quantify_norms(words, final_states, block_ids)

        test_base_prop += base_prop
        test_attn_prop += attn_prop

        test_base_emb_norm += base_emb_norm
        test_attn_emb_norm += attn_emb_norm

        test_base_h_norm += base_h_norm
        test_attn_h_norm += attn_h_norm

    '''
    outfile_name = "examples/" + TASK_NAME + "_" + MODEL_TYPE + "_hammer=" + str(C_HAMMER) \
         +"_kld=" + str(C_KLD) + "_seed=" + str(SEED) + "_iter=" +  str(iter) + ".pickle"

    pickle.dump(example_data, open(outfile_name, 'wb'))
    '''

    if attn_stats:
        logger.info("in %s set base_ratio = %.8f, attention_ratio = %.14f" % (
            stage,
            test_base_prop / len(dataset),
            test_attn_prop / len(dataset)))

        logger.info("in %s set base_emb_norm = %.4f, attn_emb_norm = %.4f" % (
            stage,
            test_base_emb_norm / len(dataset),
            test_attn_emb_norm / len(dataset)))

        logger.info("in %s set base_h_norm = %.4f, attn_h_norm = %.4f\n" % (
            stage,
            test_base_h_norm / len(dataset),
            test_attn_h_norm / len(dataset)))

    accuracy = test_correct / len(dataset)
    loss = total_loss / len(dataset)

    logger.info(f"Stage {stage}: acc = {accuracy * 100.:.2f}")
    logger.info(f"Stage {stage}: loss = {loss:.8f}\n")

    return accuracy, loss


# READING THE DATA

PREFIX = "data/" + TASK_NAME + "/"
TRAIN, DEV, TEST, N_WORDS, W2I, T2I = read_data(USE_BLOCK_FILE, USE_ATTN_FILE, PREFIX, W2I, W2C, T2I, UNK, VOCAB_SIZE,
                                                TO_ANON, TASK_NAME, VOCAB_SIZE, BLOCK_WORDS, CLIP_VOCAB, MODEL_TYPE)

# assigning updated vocabs to global ones
# W2I = w2i
# T2I = t2i

if DEBUG:
    TRAIN = TRAIN[:100]
    DEV = DEV[:100]
    TEST = TEST[:100]

# CREATE REVERSE DICTS
I2W = {v: k for k, v in W2I.items()}
I2W[UNK] = "<unk>"
I2T = {v: k for k, v in T2I.items()}

N_TAGS = len(T2I)

LOGGER.info(f"The vocabulary size is {N_WORDS}")

current_model = get_model(MODEL_TYPE, N_WORDS, EMB_SIZE, HID_SIZE, N_TAGS)
calc_ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(current_model.parameters())

LOGGER.info(f"\nEvaluating without any training ...")
LOGGER.info(f"ITER: {0}")
_, _ = evaluate(current_model, TEST, I2W, I2T, stage='test', attn_stats=True, num_vis=0)

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
        entropy_loss = calc_entropy_loss(attention, C_ENTROPY)
        hammer_loss = calc_hammer_loss(words_orig, attention,
                                       block_ids, C_HAMMER)

        kld_loss = calc_kld_loss(attention, attn_orig, C_KLD)

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

    LOGGER.info(
        "iter %r: train loss=%.4f, ce_loss=%.4f, entropy_loss=%.4f, hammer_loss=%.4f, kld_loss==%.4f, time=%.2fs"
        % (ITER, avg_train_loss, avg_train_ce_loss, avg_train_entropy_loss, avg_train_hammer_loss, avg_train_kld_loss,
           epoch_duration))

    train_acc, train_loss = evaluate(current_model, TRAIN, I2W, I2T, stage='train')
    dev_acc, dev_loss = evaluate(current_model, DEV, I2W, I2T, stage='dev', attn_stats=True)
    test_acc, test_loss = evaluate(current_model, TEST, I2W, I2T, stage='test', attn_stats=True, num_vis=NUM_VIS)

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
            dump_attention_maps(current_model, TRAIN, PREFIX + "train.txt.attn." + MODEL_TYPE)
            dump_attention_maps(current_model, DEV, PREFIX + "dev.txt.attn." + MODEL_TYPE)
            dump_attention_maps(current_model, TEST, PREFIX + "test.txt.attn." + MODEL_TYPE)

    LOGGER.info(f"iter {ITER}: best test accuracy = {best_test_accuracy:0.4f} attained after epoch = {best_epoch}")

    # save the trained model
    model_path = f"{DATA_MODELS_PATH}model+{MODEL_TYPE}_task={TASK_NAME}_epoch={best_epoch}_seed={str(SEED)}_hammer=" \
                 f"{C_HAMMER:0.2f}_rand-entropy={C_ENTROPY:0.2f}.pt"
    torch.save(current_model.state_dict(), model_path)
