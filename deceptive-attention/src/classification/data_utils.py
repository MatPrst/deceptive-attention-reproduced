from collections import defaultdict

import util

DATA_PREFIX = "data/"


def initialize_vocab():
    w2i = defaultdict(lambda: len(w2i))
    w2c = defaultdict(lambda: 0.0)  # word to count
    t2i = defaultdict(lambda: len(t2i))
    unk = w2i["<unk>"]

    return w2i, w2c, t2i, unk


def read_data(task_name, model_type, logger, clip_vocab=False, block_words=None, to_anon=False, vocab_size=20000,
              use_block_file=False, use_attention_file=False):
    w2i, w2c, t2i, unk = initialize_vocab()

    prefix = f"{DATA_PREFIX}{task_name}/"
    train_path = prefix + "train.txt"
    dev_path = prefix + "dev.txt"
    test_path = prefix + "test.txt"

    train_block_file, dev_block_file, test_block_file = None, None, None
    train_attention_file, dev_attention_file, test_attention_file = None, None, None
    block_w = None

    if use_block_file:
        # log.pr_blue("Using block file")
        print(logger)
        logger.info("Using block file")

        train_block_file = f"{prefix}train.txt.block"
        dev_block_file = f"{prefix}dev.txt.block"
        test_block_file = f"{prefix}test.txt.block"
    elif use_attention_file:
        # log.pr_blue("Using attn file")
        logger.info("Using attn file")

        block_w = block_words
        train_attention_file = f"{prefix}train.txt.attn.{model_type}"
        dev_attention_file = f"{prefix}dev.txt.attn.{model_type}"
        test_attention_file = f"{prefix}test.txt.attn.{model_type}"
    else:
        if block_words is None:
            # log.pr_blue("Vanilla case: no attention manipulation")
            logger.info("Vanilla case: no attention manipulation")
        else:
            # log.pr_blue("Using block words")
            logger.info("Using block words")

        block_w = block_words

    train = list(
        read_dataset(train_path, w2i, w2c, t2i, vocab_size, unk, to_anon, task_name, block_file=train_block_file,
                     block_words=block_w, attn_file=train_attention_file, clip_vocab=clip_vocab))

    n_words = len(w2i) if not clip_vocab else vocab_size
    w2i = defaultdict(lambda: unk, w2i)
    t2i = defaultdict(lambda: unk, t2i)

    dev = list(read_dataset(dev_path, w2i, w2c, t2i, vocab_size, unk, to_anon, task_name, block_file=dev_block_file,
                            block_words=block_w, attn_file=dev_attention_file))
    test = list(read_dataset(test_path, w2i, w2c, t2i, vocab_size, unk, to_anon, task_name, block_file=test_block_file,
                             block_words=block_w, attn_file=test_attention_file))

    # reverse dictionaries
    i2w = {v: k for k, v in w2i.items()}
    i2w[unk] = "<unk>"
    i2t = {v: k for k, v in t2i.items()}

    n_tags = len(t2i)

    return train, dev, test, n_words, i2w, i2t, n_tags


def read_dataset(data_file, w2i, w2c, t2i, vocab_size, unk, to_anon, task_name, block_words=None,
                 block_file=None, attn_file=None, clip_vocab=False):
    data_lines = open(data_file, encoding="utf-8").readlines()

    if clip_vocab:
        for line in data_lines:
            tag, words = line.strip().lower().split("\t")

            for word in words.split():
                w2c[word] += 1.0

        # take only top vocab_size words
        word_freq_list = sorted(w2c.items(), key=lambda x: x[1], reverse=True)[:vocab_size - len(w2i)]

        for idx, (word, freq) in enumerate(word_freq_list):
            temp = w2i[word]  # assign the next available idx

        w2i = defaultdict(lambda: unk, w2i)

    block_lines = None
    if block_file is not None:
        block_lines = open(block_file).readlines()
        if len(data_lines) != len(block_lines):
            raise ValueError("num lines in data file does not match w/ block file")

    attn_lines = None
    if attn_file is not None:
        attn_lines = open(attn_file).readlines()
        if len(data_lines) != len(attn_lines):
            raise ValueError("num lines in data file does not match w/ attn file")

    for idx, data_line in enumerate(data_lines):
        tag, words = data_line.strip().lower().split("\t")

        if to_anon:
            words = util.anonymize(words)

        # populate block ids
        words = words.strip().split()
        block_ids = [0 for _ in words]
        attn_wts = None
        if block_words is not None:
            block_ids = [1 if i in block_words else 0 for i in words]
        elif block_lines is not None:
            block_ids = [int(i) for i in block_lines[idx].strip().split()]

            # in the case of sst-wiki anonymizing means removing
            if to_anon and task_name == 'sst-wiki':
                sst_or_wiki = map(int, block_lines[idx].strip().split())

                words = [word for word, block in zip(words, sst_or_wiki) if block == 0]  # select only wiki words
                block_ids = [i for i in block_ids if i == 0]

        if attn_lines is not None:
            attn_wts = [float(i) for i in attn_lines[idx].strip().split()]

        # check for the right len
        if len(block_ids) != len(words):
            raise ValueError("num of block words not equal to words")
        # done populating
        yield idx, [w2i[x] for x in words], block_ids, attn_wts, t2i[tag]
