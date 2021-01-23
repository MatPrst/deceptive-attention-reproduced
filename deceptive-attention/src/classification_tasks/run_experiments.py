


def main(config):

    if config.mode == 'anon':

        """
        Code to train model on specified task, with specified seed, with anonymization enabled.
        After training is done, test model on test set based on checkpoint with best dev accuracy.
        Save test evaluation statistics.
        """

    elif config.mode == 'adver':

        """
        BASELINE
        Train model with lambda = 0 for 10 epochs, save checkpoints and statistics per epoch.
        Then test model based on checkpoint with best dev accuracy. Save test statistics.
        """

        """
        ADVERSARIAL MODEL 1: lambda = 0.1
        Train model with lambda = 0.1 for 10 epochs, save checkpoints and statistics per epoch.
        """

        """
        ADVERSARIAL MODEL 2: lambda = 1.0
        Train model with lambda = 1.0 for 10 epochs, save checkpoints and statistics per epoch.
        """

        """
        MODEL SELECTION
        Load test accuracy and test AM from *baseline model*.
        for model in [model 1, model 2]
            1. determine epoch checkpoints that fall within admissible range
            2. of this subset, determine which checkpoint has largest decrease in AM
            3. test model on test set, record test statistics
        """


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='occupation', type=str,
                        help='specify task dataset')
    parser.add_argument('--mode', default='adver', type=bool,
                        help='flag for anonymization experiment or adversarial experiment')
    parser.add_argument('--seed', default=42, type=str,
                        help='set seed for experiments')
    parser.add_argument('--penalty_fn', default='mean', type=str,
                        help='set seed for experiments')
    config = parser.parse_args()

    main(config)