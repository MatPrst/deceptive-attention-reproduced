import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import models
import utils
from gen_utils import *
from models import Attention, Seq2Seq, Encoder, Decoder, DecoderNoAttn, DecoderUniform
from utils import Language
import os

# import log

# --------------- parse the flags etc ----------------- #
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', dest='task', default='copy',
                    choices=('copy', 'reverse-copy', 'binary-flip', 'en-hi', 'en-de'),
                    help='select the task you want to run on')

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)
parser.add_argument('--epochs', dest='epochs', type=int, default=5)
parser.add_argument('--seed', dest='seed', type=int, default=1234)

# parser.add_argument('--uniform', dest='uniform', action='store_true')
# parser.add_argument('--no-attn', dest='no_attn', action='store_true')
parser.add_argument('--attention', dest='attention', action='store_true', default='head-by-head')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
parser.add_argument('--num-train', dest='num_train', type=int, default=1000000)
parser.add_argument('--decode-with-no-attn', dest='no_attn_inference', action='store_true', default=True)

params = vars(parser.parse_args())
TASK = params['task']
DEBUG = params['debug']
COEFF = params['loss_coeff']
NUM_EPOCHS = params['epochs']

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

# can have values 'head-by-head', 'uniform', or 'no_attention'
ATTENTION = params['attention']

NUM_TRAIN = params['num_train']
DECODE_WITH_NO_ATTN = params['no_attn_inference']

# INPUT_VOCAB = 10000
# OUTPUT_VOCAB = 10000

long_type = torch.LongTensor
float_type = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # algiment is not trg_len x batch_size x src_len

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

            # output, attention = model(src, src_len, None, 0) #turn off teacher forcing
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
    torch.backends.cudnn.deterministic = True
    models.set_seed(seed)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_sentences(task, debug, num_train):
    sentences = []

    for sp in SPLITS:
        src_filename = "./data/" + sp + "." + task + ".src"
        trg_filename = "./data/" + sp + "." + task + ".trg"

        src_sentences = open(src_filename).readlines()
        trg_sentences = open(trg_filename).readlines()

        alignment_filename = "./data/" + sp + "." + task + ".align"

        alignment_sentences = open(alignment_filename).readlines()

        if debug:  # small scale
            src_sentences = src_sentences[:int(1e5)]
            trg_sentences = trg_sentences[:int(1e5)]
            alignment_sentences = alignment_sentences[: int(1e5)]

        if sp == 'train':
            src_sentences = src_sentences[:num_train]
            trg_sentences = trg_sentences[:num_train]
            alignment_sentences = alignment_sentences[:num_train]

        sentences.append([src_sentences, trg_sentences, alignment_sentences])

    # train_sentences = sentences[0]

    '''
    train_src_sents = train_sents[0]
    train_trg_sents = train_sents[1]
    train_alignments = train_sents[2]
    top_src_words = compute_frequencies(train_src_sents, INPUT_VOCAB)
    top_trg_words = compute_frequencies(train_trg_sents, OUTPUT_VOCAB)
    
    train_src_sents = unkify_lines(train_src_sents, top_src_words)
    train_trg_sents = unkify_lines(train_trg_sents, top_trg_words)
    train_sents = train_src_sents, train_trg_sents
    '''

    # dev_sentences = sentences[1]
    # test_sentences = sentences[2]

    return sentences


def get_batches_from_sentences(sentences, batch_size):
    train_sentences = sentences[0]
    dev_sentences = sentences[1]
    test_sentences = sentences[2]

    train_batches = list(get_batches(train_sentences[0], train_sentences[1], train_sentences[2], batch_size))
    SRC_LANG.stop_accepting_new_words()
    TRG_LANG.stop_accepting_new_words()
    dev_batches = list(get_batches(dev_sentences[0], dev_sentences[1], dev_sentences[2], batch_size))
    test_batches = list(get_batches(test_sentences[0], test_sentences[1], test_sentences[2], batch_size))

    return train_batches, dev_batches, test_batches


def get_batches(src_sentences, trg_sentences, alignments, batch_size):
    # parallel should be at least equal len
    assert (len(src_sentences) == len(trg_sentences))

    for b_idx in range(0, len(src_sentences), batch_size):

        # get the slice
        src_sample = src_sentences[b_idx: b_idx + batch_size]
        trg_sample = trg_sentences[b_idx: b_idx + batch_size]
        align_sample = alignments[b_idx: b_idx + batch_size]

        # represent them
        src_sample = [SRC_LANG.get_sent_rep(s) for s in src_sample]
        trg_sample = [TRG_LANG.get_sent_rep(s) for s in trg_sample]

        # sort by decreasing source len
        sorted_ids = sorted(enumerate(src_sample), reverse=True, key=lambda x: len(x[1]))
        src_sample = [src_sample[i] for i, v in sorted_ids]
        trg_sample = [trg_sample[i] for i, v in sorted_ids]
        align_sample = [align_sample[i] for i, v in sorted_ids]

        src_len = [len(s) for s in src_sample]
        trg_len = [len(t) for t in trg_sample]

        # largeset seq len 
        max_src_len = max(src_len)
        max_trg_len = max(trg_len)

        # pad the extra indices 
        src_sample = SRC_LANG.pad_sequences(src_sample, max_src_len)
        trg_sample = TRG_LANG.pad_sequences(trg_sample, max_trg_len)

        # generated masks
        aligned_outputs = []

        for alignment in align_sample:
            # print (alignment)
            current_alignment = np.zeros([max_trg_len, max_src_len])

            for pair in alignment.strip().split():
                src_i, trg_j = pair.split("-")
                src_i = min(int(src_i) + 1, max_src_len - 1)
                trg_j = min(int(trg_j) + 1, max_trg_len - 1)
                current_alignment[trg_j][src_i] = 1

            aligned_outputs.append(current_alignment)

        # numpy them
        src_sample = np.array(src_sample, dtype=np.int64)
        trg_sample = np.array(trg_sample, dtype=np.int64)
        aligned_outputs = np.array(aligned_outputs)
        # align output is batch_size x max target_len x max_src_len

        assert (src_sample.shape[1] == max_src_len)

        yield src_sample, src_len, trg_sample, trg_len, aligned_outputs


def initialize_model(attention, encoder_emb_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim):
    # --------------------------------------------------------#
    # ------------------- define the model -------------------#
    # --------------------------------------------------------#

    input_dim = SRC_LANG.get_vocab_size()
    output_dim = TRG_LANG.get_vocab_size()
    print(f"Input vocabulary size {input_dim} and output vocabulary size {output_dim}.")

    suffix = ""

    attn = Attention(encoder_hid_dim, decoder_emb_dim)
    enc = Encoder(input_dim, encoder_emb_dim, encoder_hid_dim, decoder_hid_dim, ENC_DROPOUT)

    if attention == 'uniform':
        dec = DecoderUniform(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)
        suffix = "_uniform"
    elif attention == 'no_attention' or DECODE_WITH_NO_ATTN:
        dec = DecoderNoAttn(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)
        if attention == 'no_attention':
            suffix = "_no-attn"
    else:
        dec = Decoder(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, DEVICE).to(DEVICE)

    # init weights
    model.apply(init_weights)

    # count the params
    print(f'The model has {count_parameters(model):,} trainable parameters.')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    return optimizer, criterion, model, suffix

    # --------- end of model definition --------- #


def train(task=TASK,
          num_epochs=NUM_EPOCHS,
          coeff=COEFF,
          seed=SEED,
          batch_size=BATCH_SIZE,
          attention=ATTENTION,
          debug=DEBUG,
          num_train=NUM_TRAIN,
          encoder_emb_dim=ENC_EMB_DIM,
          decoder_emb_dim=DEC_EMB_DIM,
          encoder_hid_dim=ENC_HID_DIM,
          decoder_hid_dim=DEC_HID_DIM):
    print(f"Starting training..........")
    print(f"Configuration:\n num_epochs: {num_epochs}\n coeff: {coeff}\n seed: {seed}\n batch_size: "
          f"{batch_size}\n attention: {attention}\n debug: {debug}\n num_train: {num_train}\n")

    # load vocabulary if already present
    src_vocab_path = "data/" + task + '_coeff=' + str(coeff) + ".src.vocab"
    trg_vocab_path = "data/" + task + '_coeff=' + str(coeff) + ".trg.vocab"

    if os.path.exists(src_vocab_path):
        SRC_LANG.load_vocab(src_vocab_path)

    if os.path.exists(trg_vocab_path):
        TRG_LANG.load_vocab(trg_vocab_path)

    # setup the model
    optimizer, criterion, model, suffix = initialize_model(attention, encoder_emb_dim, decoder_emb_dim, encoder_hid_dim,
                                                           decoder_hid_dim)

    sentences = initialize_sentences(task, debug, num_train)

    train_batches, dev_batches, test_batches = get_batches_from_sentences(sentences, batch_size)

    best_valid_loss = float('inf')
    convergence_time = 0.0
    epochs_taken_to_converge = 0

    no_improvement_last_time = False

    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss, train_acc, train_attn_mass = train_model(model, train_batches, optimizer, criterion, coeff)
        valid_loss, val_acc, val_attn_mass = evaluate(model, dev_batches, criterion)

        end_time = time.time()

        epoch_minutes, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'data/models/model_' + task + suffix + '_seed=' + str(seed) + '_coeff='
                       + str(coeff) + '_num-train=' + str(num_train) + '.pt')
            epochs_taken_to_converge = epoch + 1
            convergence_time += (end_time - start_time)
            no_improvement_last_time = False
        else:
            # no improvement this time
            if no_improvement_last_time:
                break
            no_improvement_last_time = True

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_minutes}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:0.2f} \
            | Train Attn Mass: {train_attn_mass:0.2f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |   Val Acc: {val_acc:0.2f} \
            |  Val. Attn Mass: {val_attn_mass:0.2f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # load the best model and print stats:
    model.load_state_dict(torch.load('data/models/model_' + task + suffix + '_seed=' + str(seed) + '_coeff='
                                     + str(coeff) + '_num-train=' + str(num_train) + '.pt'))

    test_loss, test_acc, test_attn_mass = evaluate(model, test_batches, criterion)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc:0.2f} \
            |  Test Attn Mass: {test_attn_mass:0.2f} |  Test PPL: {math.exp(test_loss):7.3f}')

    print(f"Final Test Accuracy ..........\t{test_acc:0.2f}")
    print(f"Final Test Attention Mass ....\t{test_attn_mass:0.2f}")
    print(f"Convergence time in seconds ..\t{convergence_time:0.2f}")
    print(f"Sample efficiency in epochs ..\t{epochs_taken_to_converge}")

    SRC_LANG.save_vocab("data/vocab/" + task + suffix + '_seed=' + str(seed)
                        + '_coeff=' + str(coeff) + '_num-train=' + str(num_train) + ".src.vocab")
    TRG_LANG.save_vocab("data/vocab/" + task + suffix + '_seed=' + str(seed)
                        + '_coeff=' + str(coeff) + '_num-train=' + str(num_train) + ".trg.vocab")

    if task in ['en-hi', 'en-de']:
        # generate the output to compute bleu scores as well...
        print("generating the output translations from the model")

        test_sentences = sentences[2]
        test_batches_single = list(get_batches(test_sentences[0], test_sentences[1], test_sentences[2], 1))
        output_lines = generate(model, test_batches_single)

        print("[done] .... now dumping the translations")

        outfile = "data/" + task + suffix + "_seed" + str(seed) + '_coeff=' + str(coeff) + '_num-train=' \
                  + str(num_train) + ".test.out"
        fw = open(outfile, 'w')
        for line in output_lines:
            fw.write(line.strip() + "\n")
        fw.close()


def main():

    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/models/'):
        os.makedirs('data/models/')

    if not os.path.exists('data/vocab/'):
        os.makedirs('data/vocab/')

    train()


if __name__ == "__main__": main()
