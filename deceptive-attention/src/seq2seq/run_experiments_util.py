from train import *

# BEST_EPOCHS = {'en-de-dot-product-0.0-1': 7,
#                'en-de-dot-product-0.1-1': 8,
#                'en-de-dot-product-1.0-1': 7,
#                'en-de-dot-product-0.0-2': 6,
#                'en-de-dot-product-0.1-2': 9,
#                'en-de-dot-product-1.0-2': 7,
#                'en-de-dot-product-0.0-3': 6,
#                'en-de-dot-product-0.1-3': 6,
#                'en-de-dot-product-1.0-3': 8,
#                'en-de-dot-product-0.0-4': 7,
#                'en-de-dot-product-0.1-4': 6,
#                'en-de-dot-product-1.0-4': 8,
#                'en-de-dot-product-0.0-5': 7,
#                'en-de-dot-product-0.1-5': 7,
#                'en-de-dot-product-1.0-5': 7,
#                'en-de-uniform-0.0-1': 10,
#                'en-de-uniform-0.0-2': 7,
#                'en-de-uniform-0.0-3': 7,
#                'en-de-uniform-0.0-4': 8,
#                'en-de-uniform-0.0-5': 7,
#                'en-de-no-attn-0.0-1': 10,
#                'en-de-no-attn-0.0-2': 7,
#                'en-de-no-attn-0.0-3': 7,
#                'en-de-no-attn-0.0-4': 8,
#                'en-de-no-attn-0.0-5': 7,
#                'binary-flip-dot-product-0.0-1': 13,
#                'binary-flip-dot-product-0.1-1': 5,
#                'binary-flip-dot-product-1.0-1': 5,
#                'binary-flip-dot-product-0.0-2': 17,
#                'binary-flip-dot-product-0.1-2': 15,
#                'binary-flip-dot-product-1.0-2': 10,
#                'binary-flip-dot-product-0.0-3': 3,
#                'binary-flip-dot-product-0.1-3': 4,
#                'binary-flip-dot-product-1.0-3': 4,
#                }

BEST_EPOCHS = {'en-de-dot-product-0.0-1': 7,
               'en-de-dot-product-0.0-2': 6,
               'en-de-dot-product-0.0-3': 6,
               'en-de-dot-product-0.0-4': 7,
               'en-de-dot-product-0.0-5': 7  # ,
               # 'en-de-no-attn-0.0-1': 10,
               # 'en-de-no-attn-0.0-2': 7,
               # 'en-de-no-attn-0.0-3': 7,
               # 'en-de-no-attn-0.0-4': 8,
               # 'en-de-no-attn-0.0-5': 7
               }


def get_trained_model(task, seed, coefficient, attention, best_epoch, decode_no_att_inference=False, logger=None):
    _, criterion, model, _ = initialize_model(attention, ENC_EMB_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM,
                                              decode_no_att_inference, logger)
    map_location = torch.cuda if torch.cuda.is_available() else torch.device('cpu')
    model_path = get_model_path(task, attention, seed, coefficient, best_epoch)
    model_path.replace('models', 'models-pretrained')
    print(f'Loading model with path: {model_path}')
    model_object = torch.load(model_path, map_location=map_location)
    model.load_state_dict(model_object)
    return model, criterion


def evaluate_test(task, coefficient, seed, attention, batch_size=128):
    load_vocabulary(coefficient, task)

    sentences = initialize_sentences(task, debug=False, num_train=100000, splits=SPLITS)

    _, _, test_batches = get_batches_from_sentences(sentences, batch_size, SRC_LANG, TRG_LANG)

    # load the best model for the given settings
    epoch_key = '-'.join([task, attention, str(coefficient), str(seed)])
    if epoch_key not in BEST_EPOCHS:
        print(f'Could not find file. Proceeding to next model.')
        return None, None, None

    try:
        model, criterion = get_trained_model(task, seed, coefficient, attention, BEST_EPOCHS[epoch_key])
    except FileNotFoundError:
        print(f'Could not find file. Proceeding to next model.')
        return None, None, None

    # print('Evaluating ...')

    _, accuracy, attention_mass = evaluate(model, test_batches, criterion)

    # evaluate BLEU scores
    bleu_score = None
    if task == 'en-de':
        # print('Generating translations ...')

        # translation_path = get_translations_path(coefficient, best_epoch, seed, '_' + attention, task)
        # command = ['compare-mt', f'{translation_path}.src.out', f'{translation_path}.test.out']
        # output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
        # python_output = output.decode("utf-8")
        #
        # bleu_index = python_output.index('BLEU    ')
        # bleu_scor__e = python_output[bleu_index + 8:bleu_index + 8 + 5]
        # print(bleu_score)

        _, _, bleu_score = generate_translations(model, sentences)

    return accuracy, attention_mass, bleu_score


def run_experiment(task, epochs, seed, coefficient, attention, stage='train', num_sentences_train=1000000):
    if type(seed) is int:
        seed = [seed]

    metrics = {"acc": [], "att_mass": [], 'bleu_score': []}

    for s in seed:
        print(f'Configuration: coeff: {coefficient} seed: {s} attention: {attention} device: {DEVICE} task: {task}')
        acc, att_mass, bleu_score = None, None, None

        if stage == 'train':
            acc, att_mass, bleu_score = train(task, coefficient, s, attention, epochs, num_train=num_sentences_train)
        elif stage == 'test':
            acc, att_mass, bleu_score = evaluate_test(task, coefficient, s, attention)

        if torch.is_tensor(att_mass):
            att_mass = att_mass.item()

        if torch.is_tensor(acc):
            acc = acc.item()

        if acc is not None:
            metrics["acc"].append(acc)
        if att_mass is not None:
            metrics["att_mass"].append(att_mass)
        if bleu_score is not None:
            metrics["bleu_score"].append(bleu_score)

    return metrics


def run_en_de_experiments(clear_out=False, stage='train', seeds=None, coefficients=None, attentions=None, epochs=None,
                          num_sentences_train=1000000):
    if clear_out:
        clear_output(wait=True)

    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    task = 'en-de'
    # task_name = "English German"

    if coefficients is None:
        coefficients = [0.0, 1.0, 0.1]

    if attentions is None:
        attentions = ['dot-product', 'uniform', 'no-attention']

    if epochs is None:
        epochs = 1

    data = {
        "Attention": [],
        "$\\lambda$": [],
        "BLEU (NLTK)": [],
        "BLEU Std.": [],
        "Accuracy": [],
        "Acc. Std.": [],
        "Attention Mass": [],
        "A.M. Std.": [],
    }

    attention_names = {
        attentions[0]: "Dot-Product",
        attentions[1]: attentions[1].capitalize(),
        attentions[2]: "None"
    }

    for attention in attentions:
        for coefficient in get_coefficients(attention, coefficients):
            data["Attention"].append(attention_names[attention])
            data["$\\lambda$"].append(coefficient)

            metrics = run_experiment(task, epochs, seeds, coefficient, attention, stage=stage,
                                     num_sentences_train=num_sentences_train)

            data['Accuracy'].append(mean(metrics["acc"]))
            data['Acc. Std.'].append(std(metrics["acc"]))
            data['Attention Mass'].append(mean(metrics["att_mass"]))
            data['A.M. Std.'].append(std(metrics["att_mass"]))
            data['BLEU (NLTK)'].append(mean(metrics["bleu_score"]))
            data['BLEU Std.'].append(std(metrics["bleu_score"]))

            if clear_out:
                clear_output(wait=True)

    print("Finished all experiments. Displaying ...")

    data_frame = pd.DataFrame(data)
    data_frame.style.set_properties(**{'text-align': 'center'})
    return data_frame


def run_synthetic_experiments(clear_out=False, stage='train', seeds=None, tasks=None, coefficients=None,
                              attentions=None, epochs=None, num_sentences_train=1000000):
    if clear_out:
        clear_output(wait=True)

    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    if tasks is None:
        tasks = ['copy', 'reverse-copy', 'binary-flip']
        coefficients = [0.0, 1.0, 0.1]

    if attentions is None:
        attentions = ['dot-product', 'uniform', 'no-attention']

    if epochs is None:
        epochs = 30

    data = {
        "Attention": [],
        "$\\lambda$": [],
        "Bigram Flip Acc.": [],
        "Bigram Flip A.M.": [],
        "Sequence Copy Acc.": [],
        "Sequence Copy A.M.": [],
        "Sequence Reverse Acc.": [],
        "Sequence Reverse A.M.": []
    }

    attention_names = {
        attentions[0]: "Dot-Product",
        attentions[1]: attentions[1].capitalize(),
        attentions[2]: "None"
    }

    task_names = {
        tasks[0]: "Sequence Copy",
        tasks[1]: "Sequence Reverse",
        tasks[2]: "Bigram Flip"
    }

    for task in tasks:
        for attention in attentions:
            for coefficient in get_coefficients(attention, coefficients):
                data["Attention"].append(attention_names[attention])
                data["$\\lambda$"].append(coefficient)

                metrics = run_experiment(task, epochs, seeds, coefficient, attention, stage=stage,
                                         num_sentences_train=num_sentences_train)

                task_name = task_names[task]
                data[task_name + " Acc."].append(mean(metrics["acc"]))
                data[task_name + " A.M."].append(mean(metrics["att_mass"]))

                if clear_out:
                    clear_output(wait=True)

    print("Finished all experiments. Displaying ...")

    frame = pd.DataFrame(data)
    frame.style.set_properties(**{'text-align': 'center'})
    return frame


def get_coefficients(attention, coefficients):
    return coefficients if attention == 'dot-product' else [0.0]


def mean(data):
    if data is None or len(data) == 0:
        return -1
    return round(sum(data) / len(data), 2)


def std(data):
    n = len(data)
    if n == 0:
        return -1
    mn = mean(data)
    deviations = [(x - mn) ** 2 for x in data]
    return round(sum(deviations) / n, 2)
