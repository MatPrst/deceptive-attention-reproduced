import numpy as np
import torch
from numpy import linalg as la

from models import EmbAttModel, BiLSTMAttModel, BiLSTMModel

LONG_TYPE = torch.LongTensor
FLOAT_TYPE = torch.FloatTensor
if torch.cuda.is_available():
    LONG_TYPE = torch.cuda.LongTensor
    FLOAT_TYPE = torch.cuda.FloatTensor


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_type, nwords, emb_size, hid_size, ntags):
    if model_type == 'emb-att':
        model = EmbAttModel(nwords, emb_size, ntags)
    elif model_type == 'emb-lstm-att':
        model = BiLSTMAttModel(nwords, emb_size, hid_size, ntags)
    elif model_type == 'no-att-only-lstm':
        model = BiLSTMModel(nwords, emb_size, hid_size, ntags)
    else:
        raise ValueError("model type not compatible")

    if torch.cuda.is_available():
        model.cuda()

    return model


def quantify_attention(ix, p, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_prop = len(sent_keyword_idxs) / len(ix)
    att_prop = sum([p[i] for i in sent_keyword_idxs])
    return base_prop, att_prop


def quantify_norms(ix, word_embeddings, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_ratio = len(sent_keyword_idxs) / len(ix)
    attn_ratio = sum([la.norm(word_embeddings[i]) for i in sent_keyword_idxs])
    # normalize the attn_ratio
    attn_ratio /= sum([la.norm(emb) for emb in word_embeddings])
    return base_ratio, attn_ratio


def calc_hammer_loss(ix, attention, block_ids, coef=0.0):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    if len(sent_keyword_idxs) == 0:
        return torch.zeros([1]).type(FLOAT_TYPE)
    loss = -1 * coef * torch.log(1 - torch.sum(attention[sent_keyword_idxs]))
    return loss


def calc_kld_loss(p, q, coef=0.0):
    if p is None or q is None:
        return torch.tensor([0.0]).type(FLOAT_TYPE)
    return -1 * coef * torch.dot(p, torch.log(p / q))


def entropy(p):
    return torch.distributions.Categorical(probs=p).entropy()


def calc_entropy_loss(p, beta):
    return -1 * beta * entropy(p)


def dump_attention_maps(model, dataset, filename):
    fw = open(filename, 'w')

    dataset = sorted(dataset, key=lambda x: x[0])
    for _, words, _, _, _ in dataset:
        words_t = torch.tensor([words]).type(LONG_TYPE)
        _, attn = model(words_t)
        attention = attn[0].detach().cpu().numpy()

        for att in attention:
            fw.write(str(att) + " ")
        fw.write("\n")
    fw.close()
    return
