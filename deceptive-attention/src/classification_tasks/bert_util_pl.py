import torch
from torch.utils.data import Dataset, DataLoader
from pandas import read_csv
import pytorch_lightning as pl
from transformers import AutoTokenizer

import numpy as np

import sys

# #TODO: implement Dataset object for Gender Identification task
# class GenderDataset(Dataset):
#     raise NotImplementedError
#
# #TODO: implement Dataset object for SST task
# class SSTDataset(Dataset):
#     raise NotImplementedError

def generate_mask(self, tokenized_sents_ids, tokenized_impermissible_ids):
    """
    This function generates the self-attention mask M required to ensure
    a restriction in the information flow between permissible and impermissible tokens.
    """
    raise NotImplementedError

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

        # convert np arrays to tensors
        sentence = torch.from_numpy(sentence)
        mask_vector = torch.from_numpy(mask_vector)
        attention_mask = torch.from_numpy(attention_mask)

        # return sample dict with sentence, associated label, and one-hot mask_vectors m
        sample = {'sentences': sentence, 'attention_masks': attention_mask,
                  'labels': label, 'mask_vectors': mask_vector}
        return sample

class GenericDataModule(pl.LightningDataModule):
    """
    This Lightning module takes a "task" argument and produces DataLoaders for that task
    using predefined task-Dataset instances
    """
    def __init__(self, task, anonymization, max_length, batch_size):
        super().__init__()
        self.task = task
        self.anonymization = anonymization
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.task == 'occupation':
            self.train = OccupationDataset(dataset='train', max_length=self.max_length, anonymization=self.anonymization)
            self.val = OccupationDataset(dataset='dev', max_length=self.max_length,  anonymization=self.anonymization)
            self.test = OccupationDataset(dataset='test', max_length=self.max_length,  anonymization=self.anonymization)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)

