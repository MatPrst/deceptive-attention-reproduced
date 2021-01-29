# Deceive with Attention-Based Explanations

This repository and paper have been conducted during the Fairness, Accountability, Confidentiality and Transparency (FACT) course at University of Amsterdam. 
Starting off with reproducing the original paper `Learning to Deceive with Attention-Based Explanations` we extend upon it with <TODO>.
  

# Organisation of the Repository

Following the author's repository, we divided the codebase into [```classification```](https://github.com/MatPrst/FACT/tree/main/deceptive-attention/src/classification) and [```seq2seq```](https://github.com/MatPrst/FACT/tree/main/deceptive-attention/src/seq2seq) subfolders. In ```classification```, you can find both code used for *reproducing* the results, and code for *replicating* the results (specifically, see the [```BERT_replication```](https://github.com/MatPrst/FACT/tree/main/deceptive-attention/src/classification/BERT_replication) subfolder).

## Environment Configuration

In order to run the code either create an Anaconda environment:

```
conda env create -f env.yml
```

or create a new environment and install all required packages with:

```
pip install -r requirements.txt
```
Please note that for the *replication* code, [```a separate environment```](./FACT/blob/main/deceptive-attention/src/classification/BERT_replication/BERT_env.yml) should be installed as this part of the code employs more recent libraries.

## Visualizing Results

We provide several IPython notebooks detailing all results from the respective parts of our reproducibility report:

-
-
- [```BERT replication notebook```](https://github.com/MatPrst/FACT/blob/main/deceptive-attention/src/classification/BERT_replication/BERT%20replication%20notebook%20completed.ipynb)

## Authors

- Rahel Habacker
- Andrew Harrison
- Mathias Parisot
- Ard Snijders

## Acknowledgements

The original implementation and paper that we base on have been proposed by:

> Learning to Deceive with Attention-Based Explanations \
> Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, Zachary C. Lipton \
> 2020 (https://arxiv.org/abs/1909.07913)

Full credits for proposed methods used from this paper belong to these authors. Huge parts of the code we provide in this repository are based on their Github repository: \
https://github.com/danishpruthi/deceptive-attention.

In addition to that, we also use code from:
