import argparse
import os
import random
import time

from torch.utils.tensorboard import SummaryWriter

from data_utils import *
from log_utils import setup_logger
from train_utils import *

def main():
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

    parser.add_argument('--loss-hammer', dest='loss_hammer', type=float, default=0.,
                        help='strength for hammer loss on attention weights')

    parser.add_argument('--seed', dest='seed', type=int, default=1,
                        help='set random seed, defualt = 1')

    parser.add_argument('--tensorboard_log', dest='tensorboard_log', default=False, action='store_true')

    # flags specifying whether to use the block and attn file or not

    parser.add_argument('--use-block-file', dest='use_block_file', action='store_true')

    parser.add_argument('--block-words', dest='block_words', nargs='+', default=None,
                        help='list of words you wish to block (default is None)')

    parser.add_argument('--use-loss', dest='use_loss', action='store_true')

    parser.add_argument('--anon', dest='anon', action='store_true')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--understand', dest='understand', action='store_true')

    parser.add_argument('--clip-vocab', dest='clip_vocab', action='store_true')

    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=20000,
                        help='in case you clip vocab, specify the vocab size')

    params = vars(parser.parse_args())

    # useful constants
    SEED = params['seed']
    TENSORBOARD = params['tensorboard_log']
    LOG_PATH = "logs/"

    # user specified constants
    HAMMER_LOSS = params['loss_hammer']
    NUM_EPOCHS = params['num_epochs']
    EMB_SIZE = params['emb_size']
    HID_SIZE = params['hid_size']
    TO_ANON = params['anon']
    BLOCK_WORDS = params['block_words']
    USE_BLOCK_FILE = params['use_block_file']

    MODEL_TYPE = params['model']
    TASK_NAME = params['task']
    USE_LOSS = params['use_loss']
    DEBUG = params['debug']
    UNDERSTAND = params['understand']
    CLIP_VOCAB = params['clip_vocab']
    VOCAB_SIZE = params['vocab_size']

    run_experiment(
        TASK_NAME,
        MODEL_TYPE,
        NUM_EPOCHS,
        BLOCK_WORDS,
        USE_BLOCK_FILE,
        TO_ANON,
        SEED,
        HAMMER_LOSS,
        debug=DEBUG
        )

def run_experiment(
    task_name, 
    model_type, 
    num_epochs,  
    block_words, 
    use_block_file, 
    anonymize, 
    seed,
    hammer_loss,
    emb_size=128, 
    hid_size=64,
    log_path="logs/", 
    vocab_size=20000, 
    clip_vocab=False, 
    tensorboard=False, 
    use_loss=False, 
    understand=False, 
    debug=True):

    LONG_TYPE = torch.LongTensor
    FLOAT_TYPE = torch.FloatTensor
    if torch.cuda.is_available():
        LONG_TYPE = torch.cuda.LongTensor
        FLOAT_TYPE = torch.cuda.FloatTensor
    
    loss_config = LossConfig(hammer_loss, 0, 0)

    # create required folders if not present
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(DATA_MODELS_PATH, exist_ok=True)

    # SETUP LOGGING
    LOGGER = setup_logger(log_path, f"task={task_name}__model={model_type}_hammer={loss_config.c_hammer}_seed={seed}")
    LOGGER.info(f"Task: {task_name}")
    LOGGER.info(f"Model: {model_type}")
    LOGGER.info(f"Coef (hammer): {loss_config.c_hammer:0.2f}")
    LOGGER.info(f"Seed: {seed}\n")

    set_seed(seed)

    # READING THE DATA

    TRAIN, DEV, TEST, VOCABULARY = read_data(task_name, model_type, LOGGER, clip_vocab, block_words, anonymize, vocab_size,
                                            use_block_file)

    if debug:
        TRAIN = TRAIN[:100]
        DEV = DEV[:100]
        TEST = TEST[:100]

    LOGGER.info(f"The source vocabulary size / input_dim is {VOCABULARY.n_words}")
    LOGGER.info(f"The target vocabulary size / output_dim is {VOCABULARY.n_tags}")

    current_model = get_model(model_type, VOCABULARY, emb_size, hid_size)
    calc_ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(current_model.parameters())

    LOGGER.info(f"\nEvaluating without any training ...")
    LOGGER.info(f"ITER: {0}")
    _, _, _ = evaluate(current_model, TEST, VOCABULARY, loss_config, understand, False, LOGGER,
                    stage='test', attn_stats=True, num_vis=0)

    WRITER = None
    if tensorboard:
        WRITER = SummaryWriter(os.path.join(log_path, "tensorboard"))

    LOGGER.info("Starting to train. \n")

    best_dev_accuracy = 0.
    best_dev_loss = np.inf
    best_test_accuracy = 0.
    best_att_mass = 0.
    best_epoch = 0

    for ITER in range(1, num_epochs + 1):
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
            entropy_loss = calc_entropy_loss(attention, loss_config.c_entropy)
            hammer_loss = calc_hammer_loss(words_orig, attention,
                                        block_ids, loss_config.c_hammer)

            kld_loss = calc_kld_loss(attention, attn_orig, loss_config.c_kld)

            loss = ce_loss + entropy_loss + hammer_loss + kld_loss
            train_loss += loss.item()

            train_ce_loss += ce_loss.item()
            train_entropy_loss += entropy_loss.item()
            train_hammer_loss += hammer_loss.item()
            train_kld_loss += kld_loss.item()

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

        train_acc, train_loss, train_att_mass = evaluate(current_model, TRAIN, VOCABULARY, loss_config, logger=LOGGER, stage='train')
        dev_acc, dev_loss, dev_att_mass = evaluate(current_model, DEV, VOCABULARY, loss_config, logger=LOGGER, stage='dev',
                                    attn_stats=True)
        test_acc, test_loss, test_att_mass = evaluate(current_model, TEST, VOCABULARY, loss_config, logger=LOGGER, stage='test',
                                    attn_stats=True, num_vis=0)

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

        if ((not use_loss) and dev_acc > best_dev_accuracy) or (use_loss and dev_loss < best_dev_loss):

            if use_loss:
                best_dev_loss = dev_loss
            else:
                best_dev_accuracy = dev_acc
            best_test_accuracy = test_acc
            best_att_mass = test_att_mass
            best_epoch = ITER

        LOGGER.info(f"best test accuracy = {best_test_accuracy:0.4f}, attention mass = {best_att_mass:0.4f} attained after epoch = {best_epoch}\n")

        # save the trained model
        LOGGER.info("Saving trained model.\n")
        torch.save(current_model.state_dict(), get_model_path(loss_config, best_epoch, model_type, seed, task_name))
    
    return best_test_accuracy, best_att_mass

def run_sstwiki_experiment(model_type, num_epochs, anonymize, seed, hammer_loss):
    assert model_type in ["emb-att", "emb-lstm-att"], "model type should be: emb-att or emb-lstm-att"

    if type(seed) is int:
        acc, att_mass = run_experiment("sst-wiki", model_type, num_epochs, None, True, anonymize, seed, hammer_loss)
        metrics = {"acc": acc, "att_mass": att_mass}
    elif type(seed) is list:
        metrics = {"acc": [], "att_mass": []}
        for s in seed:
            acc, att_mass = run_experiment("sst-wiki", model_type, num_epochs, None, True, anonymize, s, hammer_loss)
            metrics["acc"].append(acc)
            metrics["att_mass"].append(att_mass)
    
    return metrics

def run_occupation_experiment(model_type, num_epochs, anonymize, seed, hammer_loss):
    assert model_type in ["emb-att", "emb-lstm-att"], "model type should be: emb-att or emb-lstm-att"

    block_words = "he she her his him himself herself hers mr mrs ms mr. mrs. ms."
    
    if type(seed) is int: 
        acc, att_mass = run_experiment("occupation-classification", model_type, num_epochs, block_words, False, anonymize, seed, hammer_loss, clip_vocab=True)
        metrics = {"acc": acc, "att_mass": att_mass}
    
    elif type(seed) is list:
        metrics = {"acc": [], "att_mass": []}
        for s in seed:
            acc, att_mass = run_experiment("occupation-classification", model_type, num_epochs, block_words, False, anonymize, s, hammer_loss, clip_vocab=True)
            metrics["acc"].append(acc)
            metrics["att_mass"].append(att_mass)
    
    return metrics

def run_pronoun_experiment(model_type, num_epochs, anonymize, seed, hammer_loss):
    assert model_type in ["emb-att", "emb-lstm-att"], "model type should be: emb-att or emb-lstm-att"

    block_words = "he she her his him himself herself"
    
    if type(seed) is int: 
        acc, att_mass = run_experiment("pronoun", model_type, num_epochs, block_words, False, anonymize, seed, hammer_loss)
        metrics = {"acc": acc, "att_mass": att_mass}
    
    elif type(seed) is list:
        metrics = {"acc": [], "att_mass": []}
        for s in seed:
            acc, att_mass = run_experiment("pronoun", model_type, num_epochs, block_words, False, anonymize, s, hammer_loss)
            metrics["acc"].append(acc)
            metrics["att_mass"].append(att_mass)
    
    return metrics


if __name__ == "__main__":
    main()
