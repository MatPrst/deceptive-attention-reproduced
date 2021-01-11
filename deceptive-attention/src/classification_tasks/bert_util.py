import torch
from torch.utils.data import Dataset
import pandas as pd

#TODO: implement Dataset object for Gender Identificaiton task

#TODO: implement Dataset object for SST task

#TODO: implement tokenizer function

class OccupationDataset(Dataset):
    """Occupation Classification dataset."""

    #TODO: downsample minority classes?
    #TODO: extend set of impermissible words with ms., mr. ?

    def __init__(self, dataset, anonymization=False, transform=None):
        """
        Args:

        """
        self.path = './data/occupation-classification/' + dataset + '.txt'
        self.text = pd.read_csv(self.path, sep="\t", header=None)
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