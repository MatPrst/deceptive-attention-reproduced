
def get_lowest(path_dict, baseline):
    """ This beautiful function takes a dict of model paths and
    returns the path with the lowest dev attention mass """
    am_values = []  # save the dev AM values in a list
    acc_values = []  # save the dev acc values in a list
    for path in path_dict.keys():  # Iterate over the k paths

        # Behold the most robustest of parsing technologies known to mankind
        left_idx = path.find('val_attention_mass=') + len('val_attention_mass=')
        right_idx = path.find('.ckpt')
        am_value = path[left_idx:right_idx]
        am_values.append(float(am_value))

        left_idx = path.find('val_acc=') + len('val_acc=')
        right_idx = path.find('-val_attention_mass=')
        acc_value = path[left_idx:right_idx]
        acc_values.append(float(acc_value))

    eligible = [] # keep track of checkpoints whose test accuracies are /geq 0.02
    for i, path in enumerate(path_dict.keys()):
        difference = abs(acc_values[i] - baseline)
        if difference <= 0.02 or acc_values[i] > baseline:
            eligible.append(i)

    # from all models that are within the 0.02% range, we pick the one with the greatest
    # reduction in AM, i.e. the model with the smallest attention mass
    if len(eligible) > 0:
        smallest = 999.0
        smallest_idx = None
        for i, path in enumerate(path_dict.keys()):
            if i in eligible:
                if am_values[i] < smallest:
                    smallest = am_values[i]
                    smallest_idx = i
        # return the path of the model with the smallest AM
        for i, path in enumerate(path_dict.keys()):
            if i == smallest_idx:
                return path

    # if none of the models are within the 0.02% range we output None
    else:
        print('None of the models are within 2% acc range')
        return None




path_dict = {'/penalty_mean_lambda_0.1/model-epoch=00-val_acc=1.00-val_attention_mass=40.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.90-val_attention_mass=35.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.89-val_attention_mass=36.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.20-val_attention_mass=37.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.10-val_attention_mass=38.99.ckpt': 1}

baseline = 0.90


# obtain ckpt path with lowest dev AM
best_model_path = get_lowest(path_dict, baseline)
print(best_model_path)