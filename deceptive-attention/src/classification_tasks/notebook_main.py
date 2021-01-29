# Global modules
import time
import os
import shutil
import pandas as pd

# PyTorch modules
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# Local util dependencies
from bert_util_pl import GenericDataModule

# Local model dependencies
from bert_main_pl_experiments import BERTModel, get_lowest

def del_checkpoints(path_dict, best_model_path):
    """
    This function deletes redundant checkpoint files
    after the train-val-test loop is finished
    """
    for path in path_dict.keys():
        if path != best_model_path:
            os.remove(path)

def run_experiments(SEED, TASK, MODES, PENALTY_FNS, BATCH_SIZE, MAX_EPOCHS, LR, DROPOUT,
                    MAX_LENGTH, NUM_GPUS, ACCELERATOR, NUM_WORKERS, LOG_EVERY, DEBUG,
                    TOY_RUN, PROGRESS_BAR, WARNINGS):


    # Set global Lightning seed
    seed_everything(SEED)

    # time the total duration of the experiments
    start = time.time()

    # for a given task and seed, there is a single 'anonymization' experiment, and there are 6 'adversarial' experiments.
    # we simply run the 7 experiments one after another.

    all_results = {}

    for mode in MODES:

        if mode == 'anon':
            # ---------------------------- Code for replicating anonymization experiment ---------------------------- #

            print('\n-------------- Beginning anonymization experiment for task: {} ------------\n'.format(TASK))

            # Define model
            model = BERTModel(dropout=DROPOUT,
                              lr=LR,
                              penalize=False,
                              lambeda=0,
                              penalty_fn='mean')

            # Define logger and path
            logger = pl_loggers.TensorBoardLogger('notebook_results/logs/seed_{}/task_{}/anon/'.format(SEED, TASK))

            # for the anonymization task, we want to test using the ckpt with the best dev accuracy
            # therefore we define a dedicated chkpt callback that monitors the val_acc metric
            checkpoint_callback = ModelCheckpoint(
                monitor='val_acc',
                dirpath='notebook_results/checkpoints/seed_{}/task_{}/anon/'.format(SEED, TASK),
                filename='model-{epoch:02d}-{val_acc:.2f}',
                save_top_k=1,
                mode='max')

            dm = GenericDataModule(task=TASK,
                                   anonymization=True,
                                   max_length=MAX_LENGTH,
                                   batch_size=BATCH_SIZE,
                                   num_workers=NUM_WORKERS)

            trainer = Trainer(gpus=NUM_GPUS,
                              logger=logger,
                              callbacks=[checkpoint_callback],
                              log_every_n_steps=LOG_EVERY,
                              accelerator=ACCELERATOR,
                              max_epochs=MAX_EPOCHS,
                              limit_train_batches=TOY_RUN,  # if toy_run=1, we only train for a single batch
                              limit_test_batches=TOY_RUN,  # across all the splits, which is useful when debugging
                              limit_val_batches=TOY_RUN,  # (default arg is None)
                              progress_bar_refresh_rate=PROGRESS_BAR,
                              weights_summary=None)  # don't print a summary

            # train model
            trainer.fit(model, dm)
            # load checkpoint with best dev accuracy
            checkpoint_callback.best_model_path
            # evaluate on test set
            print('Test results on {} with seed {} with anonymization: '.format(TASK, SEED))
            result = trainer.test()

            result_dict = result[0]
            all_results['anon'] = result_dict

            best_model_path = None
            path_dict = checkpoint_callback.best_k_models
            del_checkpoints(path_dict, best_model_path)


        if mode == 'adversarial':

            # ----------------------------  Code for replicating adversarial experiments ---------------------------- #
            print('\n-------------- Beginning adversarial experiments for task: {} ---------- \n'.format(TASK))

            # for the 'adversarial' models, there are 2 x 3 = 6 possible experiments that need to be ran.

            lambdas = [0, 0.1, 1.0]

            # run experiments for both penalty fns
            for penalty_fn in PENALTY_FNS:

                # given a penalty fn, run experiments for all values of lambda
                for lambeda in lambdas:
                    print('Training for penalty_fn = {} and lambda = {}...'.format(penalty_fn, lambeda))

                    # Define model
                    model = BERTModel(dropout=DROPOUT,
                                      lr=LR,
                                      penalize=True,
                                      lambeda=lambeda,
                                      penalty_fn=penalty_fn)

                    # Specify logger and path
                    logger = pl_loggers.TensorBoardLogger(
                        'notebook_results/logs/seed_{}/task_{}/penalty_{}_lambda_{}/'.format(
                            SEED, TASK, penalty_fn, lambeda))
                    # logger.log_hyperparams(config)

                    # For lambda 0 (the baseline), we checkpoint based on dev accuracy
                    checkpoint_callback = ModelCheckpoint(
                        monitor='val_acc',
                        dirpath='notebook_results/checkpoints/seed_{}/task_{}/penalty_{}_lambda_{}/'.format(
                            SEED, TASK, penalty_fn, lambeda),
                        filename='model-{epoch:02d}-{val_acc:.2f}-{val_attention_mass:.2f}',
                        save_top_k=10,
                        mode='max', )

                    # Initialise DataModule
                    dm = GenericDataModule(task=TASK,
                                           anonymization=False,
                                           max_length=MAX_LENGTH,
                                           batch_size = BATCH_SIZE,
                                           num_workers = NUM_WORKERS)

                    # Initialise Trainer
                    trainer = Trainer(
                        gpus=NUM_GPUS,
                        logger = logger,
                        callbacks = [checkpoint_callback],
                        log_every_n_steps = LOG_EVERY,
                        accelerator = ACCELERATOR,
                        max_epochs = MAX_EPOCHS,
                        limit_train_batches = TOY_RUN,
                        limit_test_batches = TOY_RUN,
                        limit_val_batches = TOY_RUN,
                        progress_bar_refresh_rate = PROGRESS_BAR,
                        weights_summary = None)  # don't print a summary

                    # Train model
                    trainer.fit(model, dm)

                    # ---------------------------- Model Selection Logic ---------------------------------------

                    # For the lambda = 0 baseline, we evaluate on test set using ckpt with highest dev accuracy
                    # For the lambda!= 0 models, we consider all models whose dev acc is within 2% range of baseline acc
                    # From those models, we pick the model with the lowest dev attention mass

                    path_dict = checkpoint_callback.best_k_models  # extract a dictionary of all k checkpoint paths
                    if lambeda != 0:  # comparison only holds for lambda 0.1, 1.0

                        # access baseline test accuracy
                        key = penalty_fn + '_test_acc'
                        baseline_acc = results_dict[key]

                        # obtain ckpt path with lowest dev AM
                        best_model_path = get_lowest(path_dict, baseline_acc)  # obtain path with lowest dev AM
                        if best_model_path is not None:  # we only overwrite the path if there is a checkpoint within 2% acc
                            checkpoint_callback.best_model_path = best_model_path

                    # (re)set best model path for testing
                    checkpoint_callback.best_model_path

                    # Evaluate on test set
                    print('Test results on {} with seed {} for model with penalty_fn = {}, lambda = {}: '.format(
                        TASK, SEED, penalty_fn, lambeda))
                    result = trainer.test()

                    # if lambda = 0, we want to store the baseline test acc for later use
                    result_dict = result[0]
                    if lambeda == 0:
                        results_dict = {}
                        key = penalty_fn + '_test_acc'
                        results_dict[key] = result[0]['test_acc']


                    # then, store in dict for final output
                    key = '{}_{}'.format(penalty_fn, lambeda)
                    all_results[key] = result_dict

                    # logic to delete files that are no longer needed
                    best_model_path = checkpoint_callback.best_model_path  # determine which checkpoint to keep
                    del_checkpoints(path_dict, best_model_path)  # remove other redundant checkpoints

    # delete checkpoints after having ran all experiments for a given task
    shutil.rmtree('notebook_results/checkpoints/seed_{}/task_{}'.format(SEED, TASK))

    # end = time.time()
    # # print("\n ---------------------------- Finished running experiments ---------------------------- ")
    # elapsed = end - start
    # print('Required time to run specified experiments: {} seconds '.format(elapsed))

    return all_results

def run_all_experiments(SEEDS,TASKS, MODES,PENALTY_FNS,BATCH_SIZE,MAX_EPOCHS,LR,
                        DROPOUT, MAX_LENGTH, NUM_GPUS,ACCELERATOR,NUM_WORKERS,LOG_EVERY,
                        DEBUG,TOY_RUN,PROGRESS_BAR,WARNINGS):
    final_dict = {}
    if os.path.exists('./notebook_results'):
        shutil.rmtree('./notebook_results')

    # time the total duration of the experiments
    start = time.time()

    for TASK in TASKS:

        final_dict[TASK] = {}

        for SEED in SEEDS:
            experiment_dict = run_experiments(SEED=SEED,
                                              TASK=TASK,
                                              MODES=MODES,
                                              PENALTY_FNS=PENALTY_FNS,
                                              BATCH_SIZE=BATCH_SIZE,
                                              MAX_EPOCHS=MAX_EPOCHS,
                                              LR=LR,
                                              DROPOUT=DROPOUT,
                                              MAX_LENGTH=MAX_LENGTH,
                                              NUM_GPUS=NUM_GPUS,
                                              ACCELERATOR=ACCELERATOR,
                                              NUM_WORKERS=NUM_WORKERS,
                                              LOG_EVERY=LOG_EVERY,
                                              DEBUG=DEBUG,
                                              TOY_RUN=TOY_RUN,
                                              PROGRESS_BAR=PROGRESS_BAR,
                                              WARNINGS=WARNINGS)

            key = '{}'.format(SEED)
            final_dict[TASK][key] = experiment_dict

    end = time.time()
    # print("\n ---------------------------- Finished running experiments ---------------------------- ")
    elapsed = end - start
    print('Required time to run specified experiments: {} seconds '.format(elapsed))

    return final_dict

def generate_table(final_dict):
    columns = []
    for task in final_dict.keys():
        for statistic in ['test acc', 'test AM']:
            columns.append('{} {}'.format(task, statistic))
        task_dict = final_dict[task]

        for seed in task_dict.keys():

            seed_dict = task_dict[seed]
            rows = []

            for experiment in seed_dict.keys():
                rows.append(experiment)

    df = pd.DataFrame(0, index=rows, columns=columns)

    for task in final_dict.keys():

        task_dict = final_dict[task]
        for seed in task_dict.keys():

            seed_dict = task_dict[seed]
            for experiment in seed_dict.keys():
                df.loc[experiment, '{} test acc'.format(task)] += seed_dict[experiment]['test_acc'] / len(
                    task_dict.keys())
                df.loc[experiment, '{} test AM'.format(task)] += seed_dict[experiment]['test_attention_mass'] / len(
                    task_dict.keys())

    return df