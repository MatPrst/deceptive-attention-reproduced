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

    def __init__(self, task, num_train, batch_size, data_path, dataset, debug=False):
        """
        Args:
        """

        self.batch_size = batch_size

        self.src_file = data_path + dataset + "." + task + '.src'
        self.trg_file = data_path + dataset + "." + task + '.trg'
        src_sentences = open(self.src_file).readlines()
        trg_sentences = open(self.trg_file).readlines()

        self.alignment_file = data_path + dataset + "." + task + ".align"
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

        self.samples = []
        self.src_samples = []
        self.trg_samples = []
        self.aligned_outputs = []

        # represent all sentences
        for idx in range(0, len(src_sentences)):
            # get the slice
            src_sample = SRC_LANG.get_sent_rep(src_sentences[idx])
            trg_sample = TRG_LANG.get_sent_rep(trg_sentences[idx])
            align_sample = alignment_sentences[idx]

            self.src_samples.append(src_sample)
            self.trg_samples.append(trg_sample)
            self.aligned_outputs.append(align_sample)

        # represent them
        # src_sample = [SRC_LANG.get_sent_rep(s) for s in src_sample]
        # trg_sample = [TRG_LANG.get_sent_rep(s) for s in trg_sample]

        # sort by decreasing source len
        sorted_ids = sorted(enumerate(self.src_samples), reverse=True, key=lambda x: len(x[1]))
        src_sample = [self.src_samples[i] for i, v in sorted_ids]
        trg_sample = [self.trg_samples[i] for i, v in sorted_ids]
        align_sample = [self.aligned_outputs[i] for i, v in sorted_ids]

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
        self.src_samples = np.array(src_sample, dtype=np.int64)
        self.trg_samples = np.array(trg_sample, dtype=np.int64)
        self.aligned_outputs = np.array(aligned_outputs)
        # align output is batch_size x max target_len x max_src_len

        assert (self.src_samples.shape[1] == max_src_len)
        assert (self.trg_samples.shape[1] == max_trg_len)

        # craft samples out of prepared data
        for idx in range(0, len(self.src_samples)):
            src_sample = self.src_samples[idx]
            trg_sample = self.trg_samples[idx]
            self.samples.append([src_sample, len(src_sample), trg_sample, len(trg_sample), self.aligned_outputs[idx]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.samples[idx]


class SentenceDataModule(pl.LightningDataModule):
    """
    This Lightning module takes a "task" argument and produces DataLoaders for that task
    using predefined task-Dataset instances.
    """

    def __init__(self, task, batch_size, num_train, data_path, debug=False):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.num_train = num_train
        self.debug = debug
        self.data_path = data_path

    # noinspection PyAttributeOutsideInit
    def setup(self, stage=None):
        self.train = SentenceDataset(self.task, self.num_train, self.batch_size, self.data_path, 'train', debug=self.debug)

        # don't accept new words from validation and test set
        SRC_LANG.stop_accepting_new_words()
        TRG_LANG.stop_accepting_new_words()

        self.val = SentenceDataset(self.task, self.num_train, self.batch_size, self.data_path, 'dev', debug=self.debug)
        self.test = SentenceDataset(self.task, self.num_train, self.batch_size, self.data_path, 'test', debug=self.debug)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # pin_memory=True
        return DataLoader(self.test, batch_size=batch_size, num_workers=4)

    def prepare_data(self, *args, **kwargs):
        # download or similar ...
        pass
