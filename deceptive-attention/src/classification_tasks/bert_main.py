# global dependencies
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

from transformers import AutoModel, AutoTokenizer, BertTokenizer
import transformers

import math
import time
import sys

# local dependencies
from bert_util import OccupationDataset
from bert_model import BERTModel
from bert_attention import BertSelfAttention_Altered

# TODO: Implement this function
def generate_mask(tokenized_sents_ids, tokenized_impermissible_ids):

    batch_size, seq_length = tokenized_sents_ids.shape

    # for i in range(batch_size):
    #
    #     # print(tokenized_sents_ids[i])
    #
    #     print(tokenized_impermissible_ids[i])
    #
    #     for ids in tokenized_impermissible_ids[i]:
    #         print(AutoTokenizer.from_pretrained('bert-base-uncased').decode(ids))
    #
    #     # id 101 = [CLS]
    #     # id 102 = [SEP]
    #
    #     sys.exit()

    return 2


def train(model, task_loaders, tokenizer, optimizer):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        train_loader = task_loaders['train']
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()
            model.train()

            labels = batch['label']
            sents = batch['sentence']
            impermissible = batch['impermissible']

            #Tokenize batch of sentences, return sentence_ids
            tokenized_sents = tokenizer(sents, padding=True, truncation=False, return_tensors="pt")

            #Tokenize batch of impermissibles, return impermissible_ids
            tokenized_impermissible = tokenizer(impermissible, padding=False, truncation=True, return_tensors="pt")

            # TODO: using sentence_ids and impermissible words, generate self-attention matrix per sentence
            if config.penalize:

                #Generate self attention mask based on permissible and impermissible token ids
                self_attention_masks = generate_mask(tokenized_sents["input_ids"],
                                                     tokenized_impermissible["input_ids"])

                # Feed data through model, along with self-attn masks
                preds, attentions = model(tokenized_sents["input_ids"])

                # Compute loss w.r.t. predictions and labels
                loss = criterion(preds, labels)

                # Add penalty R to loss
                # TODO compute R component

                # Compute gradients w.r.t loss and perform update
                loss.backward()
                optimizer.step()

                print(loss)

def main(args):

    print('\n -------------- Classification with BERT -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))

    # Logic to select the right task, create dataset object, create dataloader
    split = ['train', 'dev', 'test']
    task_loaders = {}

    print('\nLoading data..')
    if config.task == 'occupation':
        for subset in split:
            dataset = OccupationDataset(dataset=subset, anonymization=config.anon)
            if subset == 'test':
                task_loaders[subset] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            else:
                task_loaders[subset] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
                #TODO: turn shuffle back to True

    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(penalize=config.penalize)
    optimizer = torch.optim.Adam(model.parameters())

    # Crude logic to freeze all the BERT parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    #TODO: Create train() function
    print('Beginning training..')
    train(model=model,
          task_loaders=task_loaders,
          tokenizer=tokenizer,
          optimizer=optimizer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='no. of sentences sampled per pass')

    config = parser.parse_args()

    main(config)