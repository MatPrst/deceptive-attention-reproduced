# Sed2Seq Tasks

This is about how to reproduce Table 4 and 4 in our reproduction report using the unidirectional and bidirectional GRU.

Assuming [our environment `attention`](../../README.md) has been properly installed and activated.

We provide the [Jupyter Notebook `experiments.ipynb`](experiments.ipynb) which allows to reproduce all the experiments.
There are two main sections with allow for:

- Training + Evaluation: Training the model for task `Bigram Flip`, `Sequence Copy`, `Sequence Reverse` or `English to German translation`.
- Evaluation: 

## Training + Evaluation

## Evaluation

In order to run evaluation with our pretrained models:

1. Download the models from here TODO.
2. Add them to the [`pretrained-models` folder](author-based/data/pretrained-models/).
3. Run respective cells in the notebook.

The notebook will output tables with the respective means over all configured seeds for accuracy, attention mass and possibly BLEU score.
