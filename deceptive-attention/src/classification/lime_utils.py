import numpy as np

from data_utils import read_data
from train_utils import get_trained_model, DATA_MODELS_PATH
from IPython.display import clear_output

BEST_EPOCHS = {'emb-att-occupation-classification-0.0': 2,
               'emb-att-occupation-classification-0.1': 1,
               'emb-att-occupation-classification-1.0': 1,
               'emb-lstm-att-occupation-classification-0.0': 4,
               'emb-lstm-att-occupation-classification-0.1': 2,
               'emb-lstm-att-occupation-classification-1.0': 2,
               'emb-att-pronoun-0.0': 1,
               'emb-att-pronoun-0.1': 1,
               'emb-att-pronoun-1.0': 1,
               'emb-lstm-att-pronoun-0.0': 1,
               'emb-lstm-att-pronoun-0.1': 1,
               'emb-lstm-att-pronoun-1.0': 1}

BLOCK_WORDS = ['he', 'she', 'her', 'his', 'him', 'himself', 'herself', 'hers', 'mr', 'mrs', 'ms', 'mr.', 'mrs.', 'ms.']
CLASS_NAMES = ['surgeon', 'non-surgeon']


def get_as_sentence(w_indices, vocabulary):
    return ' '.join([vocabulary.i2w[index] for index in w_indices])


def fool_lime_with_models(model_type, task, explainer, num_explanations=1, instance_indices=None, regularisations=None,
                          clear_out=False):
    if clear_out:
        clear_output(wait=True)

    clip_vocab = task == 'occupation-classification'

    # read data
    _, _, dataset, vocabulary = read_data(task, model_type, block_words=BLOCK_WORDS, clip_vocab=clip_vocab)

    # pick the data instances we want to explain
    if instance_indices is None:
        # if no instances to be explained provided, randomly generate some
        instance_indices = np.random.randint(len(dataset), size=num_explanations)
    elif type(instance_indices) == int:
        instance_indices = [instance_indices]

    if regularisations is None:
        regularisations = [0.00, 0.10, 1.00]

    for index in instance_indices:

        data_instance = dataset[index]
        sentence, target = get_as_sentence(data_instance[1], vocabulary), data_instance[4]
        print(f'Explaining sentence:\n\n{sentence}\n')
        print(f'Index in dataset: {index}')

        for regularization in regularisations:
            print('---------------------------------------------------------------------------')
            print(f'Attention Regularization: {regularization:0.2f}')
            print('---------------------------------------------------------------------------')

            # get trained model
            best_epoch = BEST_EPOCHS['-'.join([model_type, task, str(regularization)])]
            model_path = f'{DATA_MODELS_PATH}model={model_type}_task={task}_epoch={best_epoch}_seed=1_hammer=' \
                         f'{regularization:0.2f}_rand-entropy=0.00.pt'
            model = get_trained_model(model_path, vocabulary)

            # store this instance inside of the model, so that while predicting we can access the block_idx
            model.data_instance_for_prediction(data_instance)

            prediction = model.predict_probabilities([sentence])[0]

            print(f'Probability ({CLASS_NAMES[1]}) =', prediction[1])
            print(f'Probability ({CLASS_NAMES[0]}) =', prediction[0])
            print('True class: %s\n' % CLASS_NAMES[target])

            explanation = explainer.explain_instance(sentence, model.predict_probabilities, num_features=6)
            explanation.as_list()
            explanation.show_in_notebook(text=True)
