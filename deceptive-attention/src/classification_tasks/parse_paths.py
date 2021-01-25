
def get_lowest(path_dict, baseline):
    """ This function takes a dict of model paths and
    returns the path with the lowest dev attention mass"""
    am_values = [] # save the dev AM values in a list
    acc_values = []  # save the dev acc values in a list
    for path in path_dict.keys(): # Iterate over the k paths

        am_value = path[-10:-5] # Extract the Dev AM score from the path str
        am_values.append(float(am_value)) # Add dev AM score to list

        acc_value = path[-34:-30]
        acc_values.append(float(acc_value))

    eligible = []
    for i, path in enumerate(path_dict.keys()):
        difference = abs(acc_values[i] - baseline)
        if difference < 0.02:
            eligible.append(i)

    if len(eligible) > 0:
        smallest = 999.0
        smallest_idx = None
        for i, path in enumerate(path_dict.keys()):
            if i in eligible:
                print(am_values[i])
                if am_values[i] < smallest:
                    smallest = am_values[i]
                    smallest_idx = i

        for i, path in enumerate(path_dict.keys()):
            if i == smallest_idx:
                return path
    else:
        print('None of the checkpoints are within 2% of baseline')




path_dict = {'/penalty_mean_lambda_0.1/model-epoch=00-val_acc=1.00-val_attention_mass=34.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.90-val_attention_mass=35.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.89-val_attention_mass=36.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.20-val_attention_mass=37.99.ckpt': 1,
            '/penalty_mean_lambda_0.1/model-epoch=00-val_acc=0.10-val_attention_mass=38.99.ckpt': 1}

baseline = 0.90


# obtain ckpt path with lowest dev AM
best_model_path = get_lowest(path_dict, baseline)
print(best_model_path)