import numpy as np


def initialize_sentences(task, debug, num_train, splits):
    sentences = []

    for sp in splits:
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

    return sentences


def get_batches_from_sentences(sentences, batch_size, source_lang, target_lang):
    train_sentences = sentences[0]
    dev_sentences = sentences[1]
    test_sentences = sentences[2]

    train_batches = list(get_batches(train_sentences, batch_size, source_lang, target_lang))

    # don't accept new words from validation and test set
    source_lang.stop_accepting_new_words()
    target_lang.stop_accepting_new_words()

    dev_batches = list(get_batches(dev_sentences, batch_size, source_lang, target_lang))
    test_batches = list(get_batches(test_sentences, batch_size, source_lang, target_lang))

    return train_batches, dev_batches, test_batches


def get_batches(sentences, batch_size, source_lang, target_lang):
    src_sentences, trg_sentences, alignments = sentences

    # parallel should be at least equal len
    assert (len(src_sentences) == len(trg_sentences))

    for b_idx in range(0, len(src_sentences), batch_size):

        # get the slice
        src_sample = src_sentences[b_idx: b_idx + batch_size]
        trg_sample = trg_sentences[b_idx: b_idx + batch_size]
        align_sample = alignments[b_idx: b_idx + batch_size]

        # represent them
        src_sample = [source_lang.get_sent_rep(s) for s in src_sample]
        trg_sample = [target_lang.get_sent_rep(s) for s in trg_sample]

        # sort by decreasing source len
        sorted_ids = sorted(enumerate(src_sample), reverse=True, key=lambda x: len(x[1]))
        src_sample = [src_sample[i] for i, v in sorted_ids]
        trg_sample = [trg_sample[i] for i, v in sorted_ids]
        align_sample = [align_sample[i] for i, v in sorted_ids]

        src_len = [len(s) for s in src_sample]
        trg_len = [len(t) for t in trg_sample]

        # large set seq len
        max_src_len = max(src_len)
        max_trg_len = max(trg_len)

        # pad the extra indices
        src_sample = source_lang.pad_sequences(src_sample, max_src_len)
        trg_sample = target_lang.pad_sequences(trg_sample, max_trg_len)

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
