import torch
from torch.utils.data import Dataset, DataLoader
from pandas import read_csv
import pytorch_lightning as pl
from transformers import AutoTokenizer

import numpy as np
import sys

# #TODO: implement Dataset object for SST task


def compute_m(sentence_ids, impermissible_ids):
    """
    This function computes the masking vector m, as used to compute the penalty R
    for a given sentence and a set of impermissible tokens.

    args:
    sentence_ids: array of token ids of shape [1, max_seq_length]
    impermissible_ids: array of token ids of shape [1, || impermissible tokens || ]
    returns:
    one-hot array of shape [1, max_seq_length]
    """

    mask = np.asarray([1 if idx in impermissible_ids else 0 for idx in sentence_ids])

    return mask

def compute_M(sentence_ids, impermissible_ids):
    """
    This function computes the masking matrix M,
    as used to restrict the information flow between different sets of tokens,
    for a given sentence and a set of impermissible tokens.

    args:
    sentence_ids: array of token ids of shape [1, max_seq_length]
    impermissible_ids: array of token ids of shape [1, || impermissible tokens || ]
    returns:
    one-hot array of shape [1, max_seq_length]
    """

    # ------ Defining attention mask M ------
    sequence_length = sentence_ids.shape[0]
    mask_matrix = np.zeros((sequence_length, sequence_length))
    # define the "sets"
    permissible_set = set(sentence_ids) - set(impermissible_ids)
    impermissible_set = set(impermissible_ids)
    # set entry = 1 where token i and j belong to same sets
    for i in range(sequence_length):
        for j in range(sequence_length):
            if sentence_ids[i] in permissible_set and sentence_ids[j] in permissible_set:
                mask_matrix[i][j] = 1
            if sentence_ids[i] in impermissible_set and sentence_ids[j] in impermissible_set:
                mask_matrix[i][j] = 1

    # ------ Dealing with CLS token ------
    # None of the other tokens should attend to the CLS token
    mask_matrix[:, 0] = 0
    # CLS token should attend to all tokens (including itself)
    mask_matrix[0,:] = 1

    # ------ Dealing with padding tokens ------
    try:
        # None of the padding tokens should attend to each other
        padding_idx = np.where(sentence_ids == 0)[0][0] # idx at which first padding token occurs
        # set all columns belonging to padding token ids to 0
        mask_matrix[:, padding_idx:] = 0
        # set all rows belonging to padding token ids to 0
        mask_matrix[padding_idx:, :] = 0
    except IndexError:
        pass

    return mask_matrix

class PronounDataset(Dataset):
    """
    This class loads the desired data split for the Pronoun Classification dataset
    """
    #TODO: downsample minority classes?

    def __init__(self, dataset, max_length=180, anonymization=False):
        """
        Args:

        """
        self.path = './data/pronoun/' + dataset + '.txt'
        self.text = read_csv(self.path, sep="\t", header=None)

        self.anonymization = anonymization
        self.impermissible = "he she her his him Ms Mr"

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.impermissible_ids = self.tokenizer(text=self.impermissible, return_tensors='np')['input_ids'][0][1:-1]
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label, sentence = self.text.iloc[idx]

        # convert capitalized <str> labels to binary
        if label == 'F':
            label = 0
        else:
            label = 1

        # if anonymization, we remove the impermissible words from each sentence
        splitted_sentence = sentence.split()
        if self.anonymization:
            sentence = ' '.join([i for i in splitted_sentence if i not in self.impermissible]) # shouldn't it be self.impermissible.split()?

        # tokenize sentence and convert to sequence of ids
        tokenized = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='np',
            truncation=True
        )

        sentence = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]

        # compute vector mask m for each sentence
        mask_vector = compute_m(sentence, self.impermissible_ids)

        # compute matrix mask M for each sentence
        mask_matrix = compute_M(sentence, self.impermissible_ids)

        # convert np arrays to tensors
        sentence = torch.from_numpy(sentence)
        attention_mask = torch.from_numpy(attention_mask)
        mask_vector = torch.from_numpy(mask_vector)
        mask_matrix = torch.from_numpy(mask_matrix).float()

        # return sample dict with sentence, associated label, and one-hot mask_vectors m
        sample = {'sentences': sentence, 'attention_masks': attention_mask,
                  'labels': label, 'mask_vectors': mask_vector, 'mask_matrices': mask_matrix}
        return sample

class OccupationDataset(Dataset):
    """
    This class loads the desired data split for the Occupation Classification dataset
    """
    #TODO: downsample minority classes?

    def __init__(self, dataset, max_length=180, anonymization=False):
        """
        Args:

        """
        self.path = './data/occupation-classification/' + dataset + '.txt'
        self.text = read_csv(self.path, sep="\t", header=None)

        self.anonymization = anonymization
        self.impermissible = "he she her his him Ms Mr"

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.impermissible_ids = self.tokenizer(text=self.impermissible, return_tensors='np')['input_ids'][0][1:-1]
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label, sentence = self.text.iloc[idx]

        # if anonymization, we remove the impermissible words from each sentence
        splitted_sentence = sentence.split()
        if self.anonymization:
            sentence = ' '.join([i for i in splitted_sentence if i not in self.impermissible]) # shouldn't it be self.impermissible.split()?

        # tokenize sentence and convert to sequence of ids
        tokenized = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='np',
            truncation=True
        )


        sentence = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]

        # compute vector mask m for each sentence
        mask_vector = compute_m(sentence, self.impermissible_ids)

        # compute matrix mask M for each sentence
        mask_matrix = compute_M(sentence, self.impermissible_ids)

        # convert np arrays to tensors
        sentence = torch.from_numpy(sentence)
        attention_mask = torch.from_numpy(attention_mask)
        mask_vector = torch.from_numpy(mask_vector)
        mask_matrix = torch.from_numpy(mask_matrix).float()

        # return sample dict with sentence, associated label, and one-hot mask_vectors m
        sample = {'sentences': sentence, 'attention_masks': attention_mask,
                  'labels': label, 'mask_vectors': mask_vector, 'mask_matrices': mask_matrix}
        return sample

class GenericDataModule(pl.LightningDataModule):
    """
    This Lightning module takes a "task" argument and produces DataLoaders for that task
    using predefined task-specific Dataset instances
    """
    def __init__(self, task, anonymization, max_length, batch_size, num_workers):
        super().__init__()
        self.task = task
        self.anonymization = anonymization
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.task == 'occupation':
            self.train = OccupationDataset(dataset='train', max_length=self.max_length, anonymization=self.anonymization)
            self.val = OccupationDataset(dataset='dev', max_length=self.max_length,  anonymization=self.anonymization)
            self.test = OccupationDataset(dataset='test', max_length=self.max_length,  anonymization=self.anonymization)
        if self.task == 'pronoun':
            self.train = PronounDataset(dataset='train', max_length=self.max_length, anonymization=self.anonymization)
            self.val = PronounDataset(dataset='dev', max_length=self.max_length,  anonymization=self.anonymization)
            self.test = PronounDataset(dataset='test', max_length=self.max_length,  anonymization=self.anonymization)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

