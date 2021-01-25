import numpy as np

from data_utils import read_data
from train_utils import get_trained_model


def get_as_sentence(w_indices, vocabulary):
    return ' '.join([vocabulary.i2w[index] for index in w_indices])


def run_lime(model_path, logger, explainer, class_names, block_words, num_explanations=1):
    # extract model_type and task from model path
    model_type = model_path[model_path.find('model='):model_path.find('_task=')].strip()
    task = model_path[model_path.find('task='):model_path.find('_epoch=')].strip()

    # read data
    _, _, test, vocabulary = read_data(task, model_type, logger, block_words=block_words, clip_vocab=True)

    # get trained model
    model = get_trained_model(model_path, vocabulary)

    # we explain 5 instances

    instance_indices = np.random.randint(len(test), size=num_explanations)
    explanations = []

    for index in instance_indices:
        data_instance = test[index]
        target = data_instance[4]

        # prediction = model.predict_probabilities(test_instance)
        # print('prediction: ', prediction)
        # print('target: ', target)

        sentence = get_as_sentence(data_instance[1], vocabulary)
        print('Explaining sentence:\n\t', sentence)

        # store this instance inside of the model, so that while predicting we can access the block_idx
        model.data_instance_for_prediction(data_instance)

        print('Data instance index: %d' % index)

        prediction = model.predict_probabilities([sentence])[0]
        print(f'Probability ({class_names[1]}) =', prediction[1])
        print(f'Probability ({class_names[0]}) =', prediction[0])
        print('True class: %s' % class_names[target])

        print('Explaining this prediction ...')

        explanations.append(explainer.explain_instance(sentence, model.predict_probabilities, num_features=6))

    return explanations
