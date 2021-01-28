import numpy as np
import torch
from torch import nn

LONG_TYPE = torch.LongTensor
FLOAT_TYPE = torch.FloatTensor
if torch.cuda.is_available():
    LONG_TYPE = torch.cuda.LongTensor
    FLOAT_TYPE = torch.cuda.FloatTensor


class EmbAttModel(nn.Module):
    def __init__(self, vocabulary, emb_dim):
        super(EmbAttModel, self).__init__()

        # layers
        self.embedding_layer = nn.Embedding(vocabulary.n_words, emb_dim)
        self.weight_layer = nn.Linear(emb_dim, 1)
        self.linear = nn.Linear(emb_dim, vocabulary.n_tags)

        self.embedding_dim = emb_dim
        self.w2i = vocabulary.w2i

        # workaround for LIME to work with out code (we need the block_ids for evaluation, which LIME does not pass)
        self.data_instance = None

    def data_instance_for_prediction(self, data_instance):
        self.data_instance = data_instance

    def predict_probabilities(self, lime_instance):

        # data_instance: perturbed data, 2d array. first element is assumed to be the original data point.
        # we now make predictions for all these data points
        all_predictions = []

        # transform sentence into indices
        original_sentence = lime_instance[0]
        indices = [self.w2i[w] for w in original_sentence.split()]

        block_ids_t, w_indices = None, None
        if self.data_instance is not None:

            _, w_indices, block_ids, _, _ = self.data_instance

            # original data instance indices should match
            # assert w_indices == self.data_instance[1]

            if w_indices == indices:
                w_indices = torch.tensor([w_indices]).type(LONG_TYPE)
                block_ids_t = torch.tensor([block_ids]).type(FLOAT_TYPE)

        if w_indices is None:
            w_indices = torch.tensor([indices]).type(LONG_TYPE)

        pred, _ = self.forward(w_indices, block_ids_t)
        prediction_probabilities = pred[0].softmax(dim=0)
        all_predictions.append(prediction_probabilities.detach().numpy())

        # predict all other samples without block_ids (because we don't know them)
        for instance in lime_instance[1:]:
            indices = [self.w2i[w] for w in instance.split()]
            indices = torch.tensor([indices]).type(LONG_TYPE)

            pred, _ = self.forward(indices)

            all_predictions.append(pred[0].softmax(dim=0).detach().numpy())

        # check that we have predictions for all neighborhood examples
        assert len(lime_instance) == len(all_predictions)

        return np.array(all_predictions)

    def forward(self, inp, block_ids=None):  # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)  # shape(emb_output): B x W x emd_dim
        weights = self.weight_layer(emb_output)  # shape(weights)   : B x W x 1
        weights = weights.squeeze(-1)  # shape(weights)   : B x W
        attentions = nn.Softmax(dim=-1)(weights)  # shape(attention) : B x W

        # NOTE: ensure block_ids are right tensors
        if block_ids is not None:
            attentions = (1 - block_ids) * attentions

        context = torch.einsum('bw,bwe->be', [attentions, emb_output])  # shape(context)   : B x W
        out = self.linear(context)  # shape(out)       : B X out_dim
        return out, attentions

    def get_embeddings(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output

    def get_linear_wts(self):
        return self.linear.weight, self.linear.bias

    def get_final_states(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output


class BiLSTMAttModel(nn.Module):
    def __init__(self, vocabulary, emb_dim, hid_dim):
        super(BiLSTMAttModel, self).__init__()

        # layers
        self.embedding_layer = nn.Embedding(vocabulary.n_words, emb_dim)
        self.lstm_layer = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.weight_layer = nn.Linear(2 * hid_dim, 1)
        self.linear = nn.Linear(2 * hid_dim, vocabulary.n_tags)

        self.embedding_dim = emb_dim
        self.w2i = vocabulary.w2i

        # workaround for LIME to work with out code (we need the block_ids for evaluation, which LIME does not pass)
        self.data_instance = None

    def data_instance_for_prediction(self, data_instance):
        self.data_instance = data_instance

    def predict_probabilities(self, lime_instance):

        # data_instance: perturbed data, 2d array. first element is assumed to be the original data point.
        # we now make predictions for all these data points
        all_predictions = []

        # transform sentence into indices
        original_sentence = lime_instance[0]
        indices = [self.w2i[w] for w in original_sentence.split()]

        block_ids_t, w_indices = None, None
        if self.data_instance is not None:

            _, w_indices, block_ids, _, _ = self.data_instance

            # original data instance indices should match
            # assert w_indices == self.data_instance[1]

            if w_indices == indices:
                w_indices = torch.tensor([w_indices]).type(LONG_TYPE)
                block_ids_t = torch.tensor([block_ids]).type(FLOAT_TYPE)

        if w_indices is None:
            w_indices = torch.tensor([indices]).type(LONG_TYPE)

        pred, _ = self.forward(w_indices, block_ids_t)
        prediction_probabilities = pred[0].softmax(dim=0)
        all_predictions.append(prediction_probabilities.detach().numpy())

        # predict all other samples without block_ids (because we don't know them)
        for instance in lime_instance[1:]:
            indices = [self.w2i[w] for w in instance.split()]
            indices = torch.tensor([indices]).type(LONG_TYPE)

            pred, _ = self.forward(indices)

            all_predictions.append(pred[0].softmax(dim=0).detach().numpy())

        # check that we have predictions for all neighborhood examples
        assert len(lime_instance) == len(all_predictions)

        return np.array(all_predictions)

    def forward(self, inp, block_ids=None):  # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)  # shape(emb_output): B x W x emd_dim
        lstm_hs, _ = self.lstm_layer(emb_output)  # shape(lstm_hs)   : B x W x 2*hid_dim
        weights = self.weight_layer(lstm_hs)  # shape(weights)   : B x W x 1
        weights = weights.squeeze(-1)  # shape(weights)   : B x W
        attentions = nn.Softmax(dim=-1)(weights)  # shape(attention) : B x W

        # NOTE: ensure block_ids are right tensors
        if block_ids is not None:
            attentions = (1 - block_ids) * attentions

        context = torch.einsum('bw,bwe->be', [attentions, lstm_hs])  # shape(context)   : B x W
        out = self.linear(context)  # shape(out)       : B X out_dim
        return out, attentions

    def get_embeddings(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output

    def get_linear_wts(self):
        return self.linear.weight, self.linear.bias

    def get_final_states(self, inp):
        emb_output = self.embedding_layer(inp)
        lstm_hs, _ = self.lstm_layer(emb_output)
        return lstm_hs


class BiLSTMModel(nn.Module):
    def __init__(self, vocabulary, emb_dim, hid_dim):
        super(BiLSTMModel, self).__init__()

        # layers
        self.embedding_layer = nn.Embedding(vocabulary.n_words, emb_dim)
        self.lstm_layer = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hid_dim, vocabulary.n_tags)

        self.embedding_dim = emb_dim
        self.w2i = vocabulary.w2i

    def forward(self, inp):  # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)  # shape(emb_output): B x W x emd_dim
        lstm_hns, _ = self.lstm_layer(emb_output)  # shape(lstm_hs)   : B x W x 2*hid_dim
        b, w, h = lstm_hns.size()
        output = lstm_hns.view(b, w, 2, h // 2)

        # NOTE: last states is concat of fwd lstm, and bwd lstm
        #      0 refers to fwd direction, and 1 for bwd direction
        #      https://pytorch.org/docs/stable/nn.html?highlight=lstm#lstm
        last_states = torch.cat((output[:, -1, 0, :], output[:, 0, 1, :]), -1)

        out = self.linear(last_states)

        # NOTE: uniform attention is returned for consistency
        #      w/ other modules which return attention weights
        attentions = 1.0 / w * torch.ones((b, w))

        return out, attentions
