# Sed2Seq Tasks

This is about how to reproduce Table 4 and 5 in our reproduction report using the unidirectional and bidirectional GRU.

Assuming [our environment `attention`](../../../env.yml) has been properly installed and activated. If not - check out our [root README file](../../../README.md).

We provide the [Jupyter Notebook `seq2seq.ipynb`](seq2seq.ipynb) which allows to reproduce all the experiments.
There are two main sections with allow for:

- Training + Evaluation: Training the model for task `Bigram Flip`, `Sequence Copy`, `Sequence Reverse` or `English to German translation`.
- Evaluation: Load our pretrained models and run the experiments with these for flexible configurations regarding seeds, coefficients and tasks.