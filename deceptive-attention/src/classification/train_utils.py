import numpy as np
import torch
from numpy import linalg as la
from tabulate import tabulate
from torch import nn

from models import EmbAttModel, BiLSTMAttModel, BiLSTMModel

LONG_TYPE = torch.LongTensor
FLOAT_TYPE = torch.FloatTensor
if torch.cuda.is_available():
    LONG_TYPE = torch.cuda.LongTensor
    FLOAT_TYPE = torch.cuda.FloatTensor

DATA_MODELS_PATH = "data/models/"


class LossConfig(object):
    def __init__(self, c_hammer, c_entropy, c_kld):
        super().__init__()
        self.c_hammer = c_hammer
        self.c_entropy = c_entropy
        self.c_kld = c_kld


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_trained_model(model_type, task_name, epoch, seed, loss_config, vocabulary, emb_size=128, hid_size=64):
    model = get_model(model_type, vocabulary, emb_size, hid_size)
    map_location = torch.cuda if torch.cuda.is_available() else torch.device('cpu')
    model_object = torch.load(get_model_path(loss_config, epoch, model_type, seed, task_name), map_location=map_location)
    model.load_state_dict(model_object)
    return model


def get_model(model_type, vocabulary, emb_size, hid_size):
    if model_type == 'emb-att':
        model = EmbAttModel(vocabulary, emb_size)
    elif model_type == 'emb-lstm-att':
        model = BiLSTMAttModel(vocabulary, emb_size, hid_size)
    elif model_type == 'no-att-only-lstm':
        model = BiLSTMModel(vocabulary, emb_size, hid_size)
    else:
        raise ValueError("model type not compatible")

    if torch.cuda.is_available():
        model.cuda()

    return model


def get_model_path(loss_config, epoch, model_type, seed, task_name):
    return f"{DATA_MODELS_PATH}model={model_type}_task={task_name}_epoch={epoch}_seed={str(seed)}_hammer=" \
           f"{loss_config.c_hammer:0.2f}_rand-entropy={loss_config.c_entropy:0.2f}.pt"


def evaluate(model, dataset, vocabulary, loss_config, understand=False, flow=False, logger=None, stage='test',
             attn_stats=False, num_vis=0):
    logger.info(f"\nEvaluating on {stage} set.\n")

    # Perform testing
    test_correct = 0.0
    test_base_prop = 0.0
    test_attn_prop = 0.0
    test_base_emb_norm = 0.0
    test_attn_emb_norm = 0.0
    test_base_h_norm = 0.0
    test_attn_h_norm = 0.0

    calc_ce_loss = nn.CrossEntropyLoss()

    example_data = []

    total_loss = 0.0
    if num_vis > 0 and understand:
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

        if stage == 'test' and flow:
            pred, attn = model(words_t, block_ids_t)
        else:
            pred, attn = model(words_t)
        attention = attn[0]

        if not flow or (stage != 'test'):
            assert 0.99 < torch.sum(attention).item() < 1.01

        ce_loss = calc_ce_loss(pred, tag_t)
        entropy_loss = calc_entropy_loss(attention, loss_config.c_entropy)
        hammer_loss = calc_hammer_loss(words, attention, block_ids, loss_config.c_hammer)
        kld_loss = calc_kld_loss(attention, attn_orig, loss_config.c_kld)

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

            example_data.append(
                [[vocabulary.i2w[w] for w in words], attn_scores, vocabulary.i2t[predict], vocabulary.i2t[tag]])

            if understand:
                headers = ['words', 'attn'] + ['e' + str(i + 1) for i in range(model.embedding_dim)]
                tabulated_list = []
                for j in range(len(words)):
                    temp_list = [vocabulary.i2w[words[j]], attn_scores[j]]
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
