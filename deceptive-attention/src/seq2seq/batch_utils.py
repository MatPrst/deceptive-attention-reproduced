import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from utils import Language

SRC_LANG = Language('src')
TRG_LANG = Language('trg')


class SentenceDataset(Dataset):
    """
    This class loads the desired data split for the Occupation Classification dataset
    """

    def __init__(self, task, num_train, batch_size, dataset, debug=False):
        """
        Args:
        """

        self.batch_size = batch_size

        self.src_file = './data/' + dataset + "." + task + '.src'
        self.trg_file = './data/' + dataset + "." + task + '.trg'
        src_sentences = open(self.src_file).readlines()
        trg_sentences = open(self.trg_file).readlines()

        self.alignment_file = "./data/" + dataset + "." + task + ".align"
        alignment_sentences = open(self.alignment_file).readlines()

        if debug:  # small scale
            src_sentences = src_sentences[:int(1e5)]
            trg_sentences = trg_sentences[:int(1e5)]
            alignment_sentences = alignment_sentences[: int(1e5)]

        if dataset == 'train':
            src_sentences = src_sentences[:num_train]
            trg_sentences = trg_sentences[:num_train]
            alignment_sentences = alignment_sentences[:num_train]

        # parallel should be at least equal len
        assert (len(src_sentences) == len(trg_sentences))

        self.src_samples = []
        self.trg_samples = []
        self.aligned_outputs = []

        for idx in range(0, len(src_sentences)):
            # get the slice
            src_sample = src_sentences[idx: idx + self.batch_size]
            trg_sample = src_sentences[idx: idx + self.batch_size]
            align_sample = alignment_sentences[idx: idx + self.batch_size]

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

            # large set seq len
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

            self.src_samples.append(src_sample)
            self.trg_samples.append(trg_sample)
            self.aligned_outputs.append(aligned_outputs)

        print('ready')

    def __len__(self):
        return len(self.src_samples)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        src_sample = self.src_samples[idx]
        trg_sample = self.trg_samples[idx]
        aligned_outputs = self.aligned_outputs[idx]

        sample = [src_sample, len(src_sample), trg_sample, len(trg_sample), aligned_outputs]

        return sample


class SentenceDataModule(pl.LightningDataModule):
    """
    This Lightning module takes a "task" argument and produces DataLoaders for that task
    using predefined task-Dataset instances
    """

    def __init__(self, task, batch_size, num_train, debug=False):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.num_train = num_train
        self.debug = debug

    # noinspection PyAttributeOutsideInit
    def setup(self, stage=None):
        self.train = SentenceDataset(self.task, self.num_train, self.batch_size, dataset='train', debug=self.debug)

        # don't accept new words from validation and test set
        SRC_LANG.stop_accepting_new_words()
        TRG_LANG.stop_accepting_new_words()

        self.val = SentenceDataset(self.task, self.num_train, self.batch_size, dataset='dev', debug=self.debug)
        self.test = SentenceDataset(self.task, self.num_train, self.batch_size, dataset='test', debug=self.debug)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        # pin_memory=True
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)

    def prepare_data(self, *args, **kwargs):
        pass

# def initialize_sentences(task, debug, num_train, splits):
#     sentences = []
#
#     for sp in splits:
#         src_filename = "./data/" + sp + "." + task + ".src"
#         trg_filename = "./data/" + sp + "." + task + ".trg"
#
#         src_sentences = open(src_filename).readlines()
#         trg_sentences = open(trg_filename).readlines()
#
#         alignment_filename = "./data/" + sp + "." + task + ".align"
#
#         alignment_sentences = open(alignment_filename).readlines()
#
#         if debug:  # small scale
#             src_sentences = src_sentences[:int(1e5)]
#             trg_sentences = trg_sentences[:int(1e5)]
#             alignment_sentences = alignment_sentences[: int(1e5)]
#
#         if sp == 'train':
#             src_sentences = src_sentences[:num_train]
#             trg_sentences = trg_sentences[:num_train]
#             alignment_sentences = alignment_sentences[:num_train]
#
#         sentences.append([src_sentences, trg_sentences, alignment_sentences])
#
#     # train_sentences = sentences[0]
#
#     '''
#     train_src_sents = train_sents[0]
#     train_trg_sents = train_sents[1]
#     train_alignments = train_sents[2]
#     top_src_words = compute_frequencies(train_src_sents, INPUT_VOCAB)
#     top_trg_words = compute_frequencies(train_trg_sents, OUTPUT_VOCAB)
#
#     train_src_sents = unkify_lines(train_src_sents, top_src_words)
#     train_trg_sents = unkify_lines(train_trg_sents, top_trg_words)
#     train_sents = train_src_sents, train_trg_sents
#     '''
#
#     # dev_sentences = sentences[1]
#     # test_sentences = sentences[2]
#     return sentences


# def get_batches_from_sentences(sentences, batch_size, source_lang, target_lang):
#     train_sentences = sentences[0]
#     dev_sentences = sentences[1]
#     test_sentences = sentences[2]
#
#     train_batches = list(get_batches(train_sentences, batch_size, source_lang, target_lang))
#
#     # don't accept new words from validation and test set
#     source_lang.stop_accepting_new_words()
#     target_lang.stop_accepting_new_words()
#
#     dev_batches = list(get_batches(dev_sentences, batch_size, source_lang, target_lang))
#     test_batches = list(get_batches(test_sentences, batch_size, source_lang, target_lang))
#
#     return train_batches, dev_batches, test_batches

# def get_batches(sentences, batch_size, source_lang, target_lang):
#     src_sentences, trg_sentences, alignments = sentences
#
#     # parallel should be at least equal len
#     assert (len(src_sentences) == len(trg_sentences))
#
#     for b_idx in range(0, len(src_sentences), batch_size):
#
#         # get the slice
#         src_sample = src_sentences[b_idx: b_idx + batch_size]
#         trg_sample = trg_sentences[b_idx: b_idx + batch_size]
#         align_sample = alignments[b_idx: b_idx + batch_size]
#
#         # represent them
#         src_sample = [source_lang.get_sent_rep(s) for s in src_sample]
#         trg_sample = [target_lang.get_sent_rep(s) for s in trg_sample]
#
#         # sort by decreasing source len
#         sorted_ids = sorted(enumerate(src_sample), reverse=True, key=lambda x: len(x[1]))
#         src_sample = [src_sample[i] for i, v in sorted_ids]
#         trg_sample = [trg_sample[i] for i, v in sorted_ids]
#         align_sample = [align_sample[i] for i, v in sorted_ids]
#
#         src_len = [len(s) for s in src_sample]
#         trg_len = [len(t) for t in trg_sample]
#
#         # large set seq len
#         max_src_len = max(src_len)
#         max_trg_len = max(trg_len)
#
#         # pad the extra indices
#         src_sample = source_lang.pad_sequences(src_sample, max_src_len)
#         trg_sample = target_lang.pad_sequences(trg_sample, max_trg_len)
#
#         # generated masks
#         aligned_outputs = []
#
#         for alignment in align_sample:
#             # print (alignment)
#             current_alignment = np.zeros([max_trg_len, max_src_len])
#
#             for pair in alignment.strip().split():
#                 src_i, trg_j = pair.split("-")
#                 src_i = min(int(src_i) + 1, max_src_len - 1)
#                 trg_j = min(int(trg_j) + 1, max_trg_len - 1)
#                 current_alignment[trg_j][src_i] = 1
#
#             aligned_outputs.append(current_alignment)
#
#         # numpy them
#         src_sample = np.array(src_sample, dtype=np.int64)
#         trg_sample = np.array(trg_sample, dtype=np.int64)
#         aligned_outputs = np.array(aligned_outputs)
#         # align output is batch_size x max target_len x max_src_len
#
#         assert (src_sample.shape[1] == max_src_len)
#
#         yield src_sample, src_len, trg_sample, trg_len, aligned_outputs
