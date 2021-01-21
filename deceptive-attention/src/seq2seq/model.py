import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import numpy as np

import utils
from batch_utils import TRG_LANG

long_type = torch.LongTensor
float_type = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor

DEVICE = torch.device('cuda' if use_cuda else 'cpu')


class BiGRU(pl.LightningModule):

    def __init__(self, input_dim, output_dim, encoder_hid_dim, decoder_hid_dim, encoder_emb_dim, decoder_emb_dim,
                 encoder_dropout, decoder_dropout, attention_type, pad_idx, sos_idx, eos_idx, coeff,
                 decode_with_no_attention=False):
        """
        PyTorch Lightning module that summarizes all components to train a GAN.

        Inputs:
            hidden_dims_gen  - List of hidden dimensionalities to use in the
                              layers of the generator
            hidden_dims_disc - List of hidden dimensionalities to use in the
                               layers of the discriminator
            dp_rate_gen      - Dropout probability to use in the generator
            dp_rate_disc     - Dropout probability to use in the discriminator
            z_dim            - Dimensionality of latent space
            lr               - Learning rate to use for the optimizer
        """
        super().__init__()

        self.save_hyperparameters()

        attention = Attention(encoder_hid_dim, decoder_hid_dim)
        encoder = Encoder(input_dim, encoder_emb_dim, encoder_hid_dim, decoder_hid_dim, encoder_dropout)

        if attention_type == 'uniform':
            dec = DecoderUniform(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, decoder_dropout,
                                 attention)
            suffix = "_uniform"
        elif attention_type == 'no_attention' or decode_with_no_attention:
            dec = DecoderNoAttn(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, decoder_dropout,
                                attention)
            if attention_type == 'no_attention':
                suffix = "_no-attn"
        else:
            dec = Decoder(output_dim, decoder_emb_dim, encoder_hid_dim, decoder_hid_dim, decoder_dropout, attention)

        self.model = Seq2Seq(encoder, dec, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)

        # TODO: init weights
        # model.apply(init_weights)

        # count the params
        # logger.info(f'The model has {count_parameters(model):,} trainable parameters.\n')

        self.loss_module = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.total_attn_mass_imp = 0.0
        self.total_trg = 0.0
        self.total_correct = 0.0

        self.coefficient = coeff

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters())
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        """
        Inputs:
            batch         - Input batch, output of the training loader.
            batch_idx     - Index of the batch in the dataset (not needed here).
        """

        src, src_len, trg, trg_len, alignment = batch

        # create tensors here...
        src = src.clone().detach().type(long_type).permute(1, 0)
        trg = trg.clone().detach().type(long_type).permute(1, 0)
        alignment = alignment.clone().detach().type(float_type).permute(1, 0, 2)
        # alignment is not trg_len x batch_size x src_len

        # print (f"source shape {src.shape}")
        # print (f"source lens {src_len}")
        output, attention = self.model(src, src_len, trg)
        # attention is

        mask = alignment  # generate_mask(attention.shape, src_len)
        # mask shape trg_len x batch_size x src_len

        attn_mass_imp = torch.einsum('ijk,ijk->', attention, mask)
        self.total_attn_mass_imp += attn_mass_imp

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

        self.total_trg += non_pad_tokens_trg  # non pad tokens trg
        # total_src += non_pad_tokens_src # non pad tokens src
        self.total_correct += torch.sum((trg == predictions) * trg_non_pad_indices).item()

        loss = self.loss_module(output, trg) - self.coefficient * torch.log(1 - attn_mass_imp / non_pad_tokens_trg)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def translate(self, test_loader):
        """
        Function for interpolating between a batch of pairs of randomly sampled
        images. The interpolation is performed on the latent input space of the
        generator.

        Inputs:
            batch_size - Number of sentences to translate.
        Outputs:
            x - Generated images of shape [B,interpolation_steps+2,C,H,W]
        """

        # ATTENTION this assumes batch size 1, code only works with batch_size 1

        # model.eval()

        epoch_loss = 0
        total_correct = 0.0
        total_trg = 0.0
        # total_src = 0.0
        total_attn_mass_imp = 0.0
        generated_lines = []
        targets = []

        for src, src_len, trg, trg_len, _ in tqdm(test_loader):

            if len(generated_lines) > 2:
                break

            # create tensors here...
            src = src.clone().detach().type(long_type).permute(1, 0)
            # trg = torch.tensor(trg).type(long_type).permute(1, 0)

            output, attention = self.model(src, src_len, None, 0)  # turn off teacher forcing

            output = output[1:].squeeze(dim=1)
            # output = [(trg sent len - 1), output dim]

            predictions = torch.argmax(output, dim=1)  # long tensor
            # shape [trg len - 1]
            generated_tokens = [TRG_LANG.get_word(w) for w in predictions.cpu().numpy()]

            generated_lines.append(" ".join(generated_tokens))

            # still with padding
            target = trg[0].cpu().numpy()

            index_eof = np.where(trg[0].cpu().numpy() == 2)
            assert len(index_eof) == 1      # there should only be one <eof>
            target = target[1:int(index_eof[0])]

            target_tokens = [TRG_LANG.get_word(w) for w in target]
            targets.append(target_tokens)

        return generated_lines, targets


# NOTE: parts of the code are inspired from Tutorial 4 in
# https://github.com/bentrevett/pytorch-seq2seq/


class Encoder(torch.nn.Module):
    """ encoder for seq2seq model """

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """ returns out from RNN
            along with the last value i.e. ht (for initializing the decoder)
        Parameters:
            src:  max_seq_len x batch_size
            src_len: list of lens of individuals sequences
        """
        embedding = self.embedding(src)

        embedded = self.dropout(embedding)

        # embedded = [src sent len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src sent len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class DecoderUniform(nn.Module):
    """ a decoder in seq2seq tasks which has uniform attention """

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        """ returns
        Parameters:
            input: shape is batch size, comprising current input to the decoder
            hidden: batch size x dec hid dim
            encoder_outputs: src sent len x batch_size x enc hid dim * 2
            mask: batch size x src sent len to remove attention for PADS
        """

        # NOTE this is run one time step at a time...

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        # have a forced uniform attention weights
        a = torch.ones_like(mask).type(float_type)
        # a is batch_size x src_sent_len
        a = a.masked_fill(mask == 0, -1e10)
        a = F.softmax(a, dim=1)
        a = a.unsqueeze(1)
        # a is batch_size x 1 x src_sent_len

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class DecoderNoAttn(nn.Module):
    """ a decoder in seq2seq tasks which has uniform attention """

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        """ returns
        Parameters:
            input: shape is batch size, comprising current input to the decoder
            hidden: batch size x dec hid dim
            encoder_outputs: src sent len x batch_size x enc hid dim * 2
            mask: batch size x src sent len to remove attention for PADS
        """

        # NOTE this is run one time step at a time...

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        # have a forced attention such that only the last encoder state is attended to...
        a = -1e10 * torch.ones_like(mask).type(float_type)
        # last_indices = torch.argmax(-mask, dim=1)

        # batch_size, _ = mask.shape

        # a[torch.arrange(batch_size), last_indices] = 1.0
        a[:, 0] = 1.0
        #
        # after normalization the 1 will remain 1, and everything else will become zero..

        a = F.softmax(a, dim=1)
        a = a.unsqueeze(1)
        # a is batch_size x 1 x src_sent_len

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class Decoder(nn.Module):
    """ a simple decoder in seq2seq tasks """

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        """ returns
        Parameters:
            input: shape is batch size, comprising current input to the decoder
            hidden: batch size x dec hid dim
            encoder_outputs: src sent len x batch_size x enc hid dim * 2
            mask: batch size x src sent len to remove attention for PADS
        """

        # NOTE this is run one time step at a time...

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src sent len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src sent len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):

        # src = [src sent len, batch size]
        # src_len = [batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        max_len = 100

        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((max_len, src.shape[1])).long().fill_(self.sos_idx).to(src.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src sent len]

        for t in range(1, max_len):

            # insert input token embedding, previous hidden state, all encoder hidden states
            # and mask
            # receive output tensor (predictions), new hidden state and attention tensor
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # place attentions in a tensor holding attention value for each input token
            attentions[t] = attention

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input is batch_size

            # if doing inference and all next token/prediction is an eos token then stop
            if inference and torch.equal(input, self.eos_idx * torch.ones_like(input)):
                return outputs[:t], attentions[:t]

        return outputs, attentions
