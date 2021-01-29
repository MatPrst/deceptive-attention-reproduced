# BERT Replication

This subdirectory contains the necessary scripts to run our replication of the BERT-based classification model.

## Environment Configuration

In order to run the code either create an Anaconda environment:

```
conda env create -f BERT_env.yml
```

Once the environment is created and activated, you can install our modified version of the transformer library using:

```
pip install transformers_editted/
```

## Visualizing Results

We provide an IPython notebook detailing all results from our reproducibility report. 
Start your jupyter server and run all cells in `BERT replication notebook.ipynb`to reproduce the reported results, or have a look at `BERT replication notebook completed.ipynb` for a pre-ran notebook.


## Running Experiments

All 7 experiments for a specified task and seed can be produced by running:

```
python bert_main_pl_experiments.py
```

For a list of possible arguments, please see

## Acknowledgements

A part of the code-base relies on the transformers library - however, we had to make several changes to some of the functionalities in this library, and therefore a local copy of this library - ` transformers_editted ` - is included in this repository. 
Namely, we had to make a change in the file `/transformers_editted/src/transformers/models/bert/modeling_bert.py`, in the class BertSelfAttention.
