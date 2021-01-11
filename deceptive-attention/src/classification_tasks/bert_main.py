# global dependencies
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

from transformers import AutoModel, AutoTokenizer, BertTokenizer
import transformers

import math
import time

# local dependencies
from bert_util import OccupationDataset
from bert_model import BERTModel
from bert_attention import BertSelfAttention_Altered


def train(model, task_loaders, impermissible_words):

    for epoch in range(1):

        train_loader = task_loaders['train']
        for i, batch in enumerate(train_loader):

            labels = batch['label']
            sents = batch['sentence']

            print(sents)

            break

def main(args):

    print('Begin training.. \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print('\n')

    #TODO: Create logic to select the right task, create dataset object, create dataloader
    split = ['train', 'dev', 'test']
    task_loaders = {}

    if config.task == 'occupation':
        for subset in split:
            dataset = OccupationDataset(dataset=subset, anonymization=config.anon)
            if subset == 'test':
                task_loaders[subset] = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            else:
                task_loaders[subset] = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        impermissible_words = dataset.impermissible

    #TODO: Create logic to define model with specified R calculation, specified self attn mask
    model = BERTModel()

    #TODO: Create train() function
    train(model=model,
          task_loaders=task_loaders,
          impermissible_words=impermissible_words)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')

    config = parser.parse_args()

    main(config)