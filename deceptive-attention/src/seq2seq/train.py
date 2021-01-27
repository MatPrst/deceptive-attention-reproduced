import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
import utils
from batch_utils import *
from gen_utils import *
from log_utils import *
from models import Attention, Seq2Seq, Encoder, Decoder, DecoderNoAttn, DecoderUniform
from utils import *

# --------------- non-determinism issues with RNN methods ----------------- #
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
# CUDA 10.1
# CUDA_LAUNCH_BLOCKING = 1

# CUDA 10.2
# CUBLAS_WORKSPACE_CONFIG =:16:8

# --------------- parse the flags etc ----------------- #
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', dest='task', default='en-de',
                    choices=('copy', 'reverse-copy', 'binary-flip', 'en-hi', 'en-de'),
                    help='select the task you want to run on')

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)
parser.add_argument('--epochs', dest='epochs', type=int, default=5)
parser.add_argument('--seed', dest='seed', type=int, default=1234)

parser.add_argument('--attention', dest='attention', type=str, default='dot-product')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
parser.add_argument('--num-train', dest='num_train', type=int, default=1000000)
parser.add_argument('--decode-with-no-attn', dest='no_attn_inference', action='store_true')

parser.add_argument('--tensorboard_log', dest='tensorboard_log', action='store_true')

params = vars(parser.parse_args())
TASK = params['task']
DEBUG = params['debug']
COEFF = params['loss_coeff']
EPOCHS = params['epochs']
TENSORBOARD_LOG = params['tensorboard_log']

LOG_PATH = "logs/"
DATA_PATH = "data/"
DATA_VOCAB_PATH = "data/vocab/"
DATA_TRANSLATIONS_PATH = "data/translations/"
DATA_MODELS_PATH = "data/models/"

long_type = torch.LongTensor
float_type = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor

DEVICE = torch.device('cuda' if use_cuda else 'cpu')

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = utils.PAD_token
SOS_IDX = utils.SOS_token
EOS_IDX = utils.EOS_token

# UNIFORM = params['uniform']
# NO_ATTN = params['no_attn']

# can have values 'dot-product', 'uniform', or 'no-attention'
ATTENTION = params['attention']

NUM_TRAIN = params['num_train']
DECODE_WITH_NO_ATTN = params['no_attn_inference']

# INPUT_VOCAB = 10000
# OUTPUT_VOCAB = 10000

SRC_LANG = Language('src')
TRG_LANG = Language('trg')

SPLITS = ['train', 'dev', 'test']

SEED = params['seed']
BATCH_SIZE = params['batch_size']


# The following function is not being used right now, and is deprecated.
def generate_mask(attn_shape, task, list_src_lens=None):
    trg_len, batch_size, src_len = attn_shape

    mask = torch.zeros(attn_shape).type(float_type)
    min_seq_len = min(trg_len, src_len)

    if task == 'copy':
        diag_items = torch.arange(min_seq_len)
        mask[diag_items, :, diag_items] = 1.0
    elif task == 'rev':
        assert list_src_lens is not None
        for b in range(batch_size):
            i = torch.arange(min_seq_len)
            j = torch.tensor([max(0, list_src_lens[b] - i - 1) for i in range(min_seq_len)])
            mask[i, b, j] = 1.0
    elif task == 'binary-flip':
        last = min_seq_len if min_seq_len % 2 == 1 else min_seq_len - 1
        i = torch.tensor([i for i in range(1, last)])
        j = torch.tensor([i - 1 if i % 2 == 0 else i + 1 for i in range(1, last)])
        mask[i, :, j] = 1.0
    elif task == 'en-hi':
        # english hindi, nothing as of now... will have a bilingual dict later.
        pass
    else:
        raise ValueError("task can be one of copy, reverse-copy, binary-flip")

        # make sure there are no impermissible tokens for first target
    mask[0, :, :] = 0.0  # the first target is free...
    mask[:, :, 0] = 0.0  # attention to sos is permissible
    mask[:, :, -1] = 0.0  # attention to eos is permissible

    return mask


def train_model(model, data, optimizer, criterion, coeff):
    model.train()

    epoch_loss = 0
    total_trg = 0.0
    # total_src = 0.0
    total_correct = 0.0
    total_attn_mass_imp = 0.0

    for src, src_len, trg, trg_len, alignment in tqdm(data):
        # create tensors here...
        src = torch.tensor(src).type(long_type).permute(1, 0)
        trg = torch.tensor(trg).type(long_type).permute(1, 0)
        alignment = torch.tensor(alignment).type(float_type).permute(1, 0, 2)
        # alignment is not trg_len x batch_size x src_len

        optimizer.zero_grad()

        # print (f"source shape {src.shape}") 
        # print (f"source lens {src_len}")
        output, attention = model(src, src_len, trg)
        # attention is 

        mask = alignment  # generate_mask(attention.shape, src_len)
        # mask shape trg_len x batch_size x src_len

        attn_mass_imp = torch.einsum('ijk,ijk->', attention, mask)
        total_attn_mass_imp += attn_mass_imp

        # print (output.shape)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].contiguous().view(-1, output.shape[-1])
        # print (output.shape)
        predictions = torch.argmax(output, dim=1)  # long tensor
        trg = trg[1:].contiguous().view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        trg_non_pad_indices = (trg != utils.PAD_token)
        non_pad_tokens_trg = torch.sum(trg_non_pad_indices).item()
        # non_pad_tokens_src = torch.sum((src != utils.PAD_token)).item()

        total_trg += non_pad_tokens_trg  # non pad tokens trg
        # total_src += non_pad_tokens_src # non pad tokens src
        total_correct += torch.sum((trg == predictions) * trg_non_pad_indices).item()

        loss = criterion(output, trg) - coeff * torch.log(1 - attn_mass_imp / non_pad_tokens_trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data), 100. * total_correct / total_trg, 100. * total_attn_mass_imp / total_trg


def evaluate(model, data, criterion):
    model.eval()

    epoch_loss = 0
    total_correct = 0.0
    total_trg = 0.0
    # total_src = 0.0
    total_attn_mass_imp = 0.0

    with torch.no_grad():
        for src, src_len, trg, trg_len, alignment in tqdm(data):
            # create tensors here...
            src = torch.tensor(src).type(long_type).permute(1, 0)
            trg = torch.tensor(trg).type(long_type).permute(1, 0)
            alignment = torch.tensor(alignment).type(float_type).permute(1, 0, 2)
            # alignment is not trg_len x batch_size x src_len

            # output, attention = model(src, src_len, None, 0) # turn off teacher forcing
            output, attention = model(src, src_len, trg, 0)  # turn off teacher forcing
            # NOTE: it is not a bug to not count extra produce from the model, beyond target len

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            mask = alignment  # generate_mask(attention.shape, src_len)
            # print ("Mask shape ", mask.shape)
            # mask shape trg_len x batch_size x src_len

            attn_mass_imp = torch.einsum('ijk,ijk->', attention, mask)
            total_attn_mass_imp += attn_mass_imp

            output = output[1:].contiguous().view(-1, output.shape[-1])
            trg = trg[1:].contiguous().view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            predictions = torch.argmax(output, dim=1)  # long tensor

            trg_non_pad_indices = (trg != utils.PAD_token)
            non_pad_tokens_trg = torch.sum(trg_non_pad_indices).item()
            # non_pad_tokens_src = torch.sum((src != utils.PAD_token)).item()

            total_trg += non_pad_tokens_trg  # non pad tokens trg
            # total_src += non_pad_tokens_src # non pad tokens src

            total_correct += torch.sum((trg == predictions) * trg_non_pad_indices).item()

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(data), 100. * total_correct / total_trg, 100. * total_attn_mass_imp / total_trg


def generate(model, data):
    # NOTE this assumes batch size 1
    model.eval()

    epoch_loss = 0
    total_correct = 0.0
    total_trg = 0.0
    # total_src = 0.0
    total_attn_mass_imp = 0.0
    generated_lines = []

    with torch.no_grad():
        for src, src_len, _, _, _ in tqdm(data):
            # create tensors here...
            src = torch.tensor(src).type(long_type).permute(1, 0)
            # trg = torch.tensor(trg).type(long_type).permute(1, 0)

            output, attention = model(src, src_len, None, 0)  # turn off teacher forcing

            output = output[1:].squeeze(dim=1)
            # output = [(trg sent len - 1), output dim]

            predictions = torch.argmax(output, dim=1)  # long tensor
            # shape [trg len - 1]
            generated_tokens = [TRG_LANG.get_word(w) for w in predictions.cpu().numpy()]

            generated_lines.append(" ".join(generated_tokens))

    return generated_lines


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    models.set_seed(seed)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(attention, encoder_emb_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, logger):
    input_dim = SRC_LANG.get_vocab_size()
    output_dim = TRG_LANG.get_vocab_size()
    logger.info(f"Input vocabulary size {input_dim} and output vocabulary size {output_dim}.")

    suffix = ""

    attn = Attention(encoder_hid_dim, decoder_hid_dim)
    enc = Encoder(input_dim, encoder_emb_dim, encoder_hid_dim, decoder_hid_dim, ENC_DROPOUT)

    if attention == 'uniform':
        dec = DecoderUniform(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)
        suffix = "_uniform"
    elif attention == 'no-attention' or DECODE_WITH_NO_ATTN:
        dec = DecoderNoAttn(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)
        if attention == 'no-attention':
            suffix = "_no-attn"
    else:
        dec = Decoder(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, DEVICE).to(DEVICE)

    # init weights
    model.apply(init_weights)

    # count the params
    logger.info(f'The model has {count_parameters(model):,} trainable parameters.\n')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    return optimizer, criterion, model, suffix


def train(task=TASK,
          epochs=EPOCHS,
          coeff=COEFF,
          seed=SEED,
          batch_size=BATCH_SIZE,
          attention=ATTENTION,
          debug=DEBUG,
          num_train=NUM_TRAIN,
          encoder_emb_dim=ENC_EMB_DIM,
          decoder_emb_dim=DEC_EMB_DIM,
          encoder_hid_dim=ENC_HID_DIM,
          decoder_hid_dim=DEC_HID_DIM,
          tensorboard_log=TENSORBOARD_LOG):
    set_seed(SEED)

    writer = None
    if tensorboard_log:
        writer = SummaryWriter(LOG_PATH + 'tensorboard/')

    logger = setup_logger(LOG_PATH, 'task=%s_coeff=%s_seed=%s' % (task, coeff, seed))

    logger.info("Starting training..........")
    logger.info(f'Configuration:\n epochs: {epochs}\n coeff: {coeff}\n seed: {seed}\n batch_size: ' +
                f'{batch_size}\n attention: {attention}\n debug: {debug}\n num_train: {num_train}\n device: {DEVICE}\n '
                f'task: {task}\n')

    # load vocabulary if already present
    src_vocab_path = DATA_PATH + task + '_coeff=' + str(coeff) + ".src.vocab"
    trg_vocab_path = DATA_PATH + task + '_coeff=' + str(coeff) + ".trg.vocab"

    if os.path.exists(src_vocab_path):
        SRC_LANG.load_vocab(src_vocab_path)

    if os.path.exists(trg_vocab_path):
        TRG_LANG.load_vocab(trg_vocab_path)

    sentences = initialize_sentences(task, debug, num_train, SPLITS)

    train_batches, dev_batches, test_batches = get_batches_from_sentences(sentences, batch_size, SRC_LANG, TRG_LANG)

    # setup the model
    optimizer, criterion, model, suffix = initialize_model(attention, encoder_emb_dim, decoder_emb_dim, encoder_hid_dim,
                                                           decoder_hid_dim, logger)

    best_valid_loss = float('inf')
    convergence_time = 0.0
    epochs_taken_to_converge = 0

    no_improvement_last_time = False

    for epoch in range(epochs):

        start_time = time.time()

        train_loss, train_acc, train_attn_mass = train_model(model, train_batches, optimizer, criterion, coeff)
        val_loss, val_acc, val_attn_mass = evaluate(model, dev_batches, criterion)

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("AttentionMass/train", train_attn_mass, epoch)

            # writer.add_hparams({"lr": "learning_rate", "bsize": batch_size, "task": task, "coeff": coeff, "seed": seed},
            #                    {})
            # {'accuracy': train_acc, 'loss': train_loss})

            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("Accuracy/valid", val_acc, epoch)
            writer.add_scalar("AttentionMass/valid", val_attn_mass, epoch)

        end_time = time.time()

        epoch_minutes, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(),
                       DATA_MODELS_PATH + 'model_' + task + suffix + '_seed=' + str(seed) + '_coeff='
                       + str(coeff) + '_num-train=' + str(num_train) + '.pt')
            epochs_taken_to_converge = epoch + 1
            convergence_time += (end_time - start_time)
            no_improvement_last_time = False
        else:
            # no improvement this time
            if no_improvement_last_time:
                break
            no_improvement_last_time = True

        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_minutes}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:0.2f} \
            | Train Attn Mass: {train_attn_mass:0.2f} | Train PPL: {math.exp(train_loss):7.3f}')
        logger.info(f'\t Val. Loss: {val_loss:.3f} |   Val Acc: {val_acc:0.2f} \
            |  Val. Attn Mass: {val_attn_mass:0.2f} |  Val. PPL: {math.exp(val_loss):7.3f}')

    # load the best model and print stats:
    model.load_state_dict(torch.load(DATA_MODELS_PATH + 'model_' + task + suffix + '_seed=' + str(seed) + '_coeff='
                                     + str(coeff) + '_num-train=' + str(num_train) + '.pt'))

    test_loss, test_acc, test_attn_mass = evaluate(model, test_batches, criterion)
    logger.info(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc:0.2f} \
            |  Test Attn Mass: {test_attn_mass:0.2f} |  Test PPL: {math.exp(test_loss):7.3f}')

    logger.info(f"Final Test Accuracy ..........\t{test_acc:0.2f}")
    logger.info(f"Final Test Attention Mass ....\t{test_attn_mass:0.2f}")
    logger.info(f"Convergence time in seconds ..\t{convergence_time:0.2f}")
    logger.info(f"Sample efficiency in epochs ..\t{epochs_taken_to_converge}")

    data_out_path = f"{task}{suffix}_seed={str(seed)}_coeff={str(coeff)}_epoch={str(epochs_taken_to_converge)}"
    vocab_out_path = f"{DATA_VOCAB_PATH}{data_out_path}"
    SRC_LANG.save_vocab(f"{vocab_out_path}.src.vocab")
    TRG_LANG.save_vocab(f"{vocab_out_path}.trg.vocab")

    if task in ['en-hi', 'en-de']:
        # generate the output to compute bleu scores as well...
        logger.info("Generating the output translations from the model.")

        translations_out_path = f"{DATA_TRANSLATIONS_PATH}{data_out_path}"
        translations, src_sentences, bleu_score = generate_translations(model, sentences, logger)

        logger.info(f"BLEU score ..........\t{bleu_score:0.2f}")

        # store files for computing BLEU score for compare-mt afterwards manually via the terminal
        logger.info("[done] .... now dumping the translations.")
        fw = open(f"{translations_out_path}.test.out", 'w', encoding='utf-8')
        for line in translations:
            fw.write(line.strip() + "\n")
        fw.close()

        logger.info(" .... now dumping the respective src sentences.")

        # flatten list
        src_sentences = [j for i in src_sentences for j in i]

        fw = open(f"{translations_out_path}.src.out", 'w', encoding='utf-8')
        for line in src_sentences:
            fw.write(line.strip() + "\n")
        fw.close()

    if writer is not None:
        writer.close()


def generate_translations(model, sentences, logger):
    test_sentences = sentences[2]
    single_test_batch = list(get_batches(test_sentences, 1, SRC_LANG, TRG_LANG))

    # logger.info(f'batch single {str(single_test_batch)}')
    # logger.info(f'batch single length {str(len(single_test_batch))}')

    output_lines = generate(model, single_test_batch)
    target_sentences = get_target_sentences_as_list(single_test_batch, TRG_LANG)
    bleu_nltk = bleu_score_nltk(target_sentences, output_lines)

    # [[...], [...]] --> [..., ...]
    targets = []
    for word_list in target_sentences:
        sentence = [' '.join(word) for word in word_list]
        targets.append(sentence)

    return output_lines, targets, bleu_nltk * 100  # report it in percentage


def main():
    # Create directories if not already existent

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(DATA_MODELS_PATH):
        os.makedirs(DATA_MODELS_PATH)

    if not os.path.exists(DATA_VOCAB_PATH):
        os.makedirs(DATA_VOCAB_PATH)

    os.makedirs(DATA_TRANSLATIONS_PATH, exist_ok=True)

    train()


if __name__ == "__main__": main()
