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

def generate_mask(tokenized_sents_ids, tokenized_impermissible_ids):

    batch_size, seq_length = tokenized_sents_ids.shape

    for i in range(batch_size):

        # print(tokenized_sents_ids[i])

        print(tokenized_impermissible_ids[i])

        for ids in tokenized_impermissible_ids[i]:
            print(AutoTokenizer.from_pretrained('bert-base-uncased').decode(ids))

        # id 101 = [CLS]
        # id 102 = [SEP]

        sys.exit()

    return 2



def train(model, task_loaders, tokenizer):

    for epoch in range(1):

        train_loader = task_loaders['train']
        for i, batch in enumerate(train_loader):

            model.train()

            labels = batch['label']
            sents = batch['sentence']
            impermissible = batch['impermissible']

            #Tokenize batch of sentences, return sentence_ids
            tokenized_sents = tokenizer(sents, padding=True, truncation=False, return_tensors="pt")

            #Tokenize batch of impermissibles, return impermissible_ids
            tokenized_impermissible = tokenizer(impermissible, padding=False, truncation=True, return_tensors="pt")

            # for ids in tokenized_sents["input_ids"]:
            #     print(tokenizer.decode(ids))

            # print(tokenized_sents["input_ids"][0])
            # print(tokenized_sents["input_ids"][1])

            # TODO: using sentence_ids and impermissible words, generate self-attention matrix per sentence
            if config.penalize:

                #Generate self attention mask based on permissible and impermissible token ids
                self_attention_masks = generate_mask(tokenized_sents["input_ids"],
                                                     tokenized_impermissible["input_ids"])

                # output = model(sents["input_ids"].unsqueeze(dim=0))
                preds, attentions = model(tokenized_sents["input_ids"])

            sys.exit()

            # for ids in tokenized_sents["input_ids"]:
            #     print(tokenizer.decode(ids))

            break

def main(args):

    print('Classification with BERT...\n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print('\n')

    #TODO: Create logic to select the right task, create dataset object, create dataloader
    split = ['train', 'dev', 'test']
    task_loaders = {}

    print('Loading data..')
    if config.task == 'occupation':
        for subset in split:
            dataset = OccupationDataset(dataset=subset, anonymization=config.anon)
            if subset == 'test':
                task_loaders[subset] = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            else:
                task_loaders[subset] = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    #TODO: specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #TODO: Create logic to define model with specified R calculation, specified self attn mask
    model = BERTModel(penalize=config.penalize)

    #TODO: Create train() function
    print('Beginning training..')
    train(model=model,
          task_loaders=task_loaders,
          tokenizer=tokenizer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='occupation', type=str,
                        help='arg to specify task to train on')
    parser.add_argument('--anon', default=False, type=bool,
                        help='arg to toggle anonymized tokens')
    parser.add_argument('--penalize', default=True, type=bool,
                        help='flag to toggle penalisation of attn to impermissible words')

    config = parser.parse_args()

    main(config)