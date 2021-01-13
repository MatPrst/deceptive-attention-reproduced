import torch
from torch.utils.data import Dataset, DataLoader
from pandas import read_csv
import pytorch_lightning as pl

# #TODO: implement Dataset object for Gender Identification task
# class GenderDataset(Dataset):
#     raise NotImplementedError
#
# #TODO: implement Dataset object for SST task
# class SSTDataset(Dataset):
#     raise NotImplementedError

class OccupationDataset(Dataset):

    """Occupation Classification dataset."""
    #TODO: downsample minority classes?
    #TODO: extend set of impermissible words with ms., mr. ?

    def __init__(self, dataset, anonymization=False, transform=None):
        """
        Args:

        """
        self.path = './data/occupation-classification/' + dataset + '.txt'
        self.text = read_csv(self.path, sep="\t", header=None)
        self.transform = transform
        self.anonymization = anonymization
        self.impermissible = "he she her his him"

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label, sentence = self.text.iloc[idx]

        # if anonymization, we remove the impermissible words from each sentence
        if self.anonymization:
            splitted_sentence = sentence.split()
            sentence = ' '.join([i for i in splitted_sentence if i not in self.impermissible])

        # return samples with sentence, label and impermissible words
        sample = {'sentence': sentence, 'label': label, 'impermissible': self.impermissible}
        return sample

class GenericDataModule(pl.LightningDataModule):
    """
    This Lightning module takes a "task" argument and produces DataLoaders for that task
    using predefined task-Dataset instances
    """
    def __init__(self, task, anonymization, batch_size):
        super().__init__()
        self.task = task
        self.anonymization = anonymization
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.task == 'occupation':
            self.train = OccupationDataset(dataset='train', anonymization=self.anonymization)
            self.val = OccupationDataset(dataset='dev', anonymization=self.anonymization)
            self.test = OccupationDataset(dataset='test', anonymization=self.anonymization)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)

