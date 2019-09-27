import os
import sys
import logging
import datetime
import warnings
import pickle
from shutil import copyfile

import numpy as np

from preprocessing import load_scramble_data
from parameter_parser import Parameters
from decision_tree import DecisionTree
from random_forest import RandomForest
from feed_forward_nn import FeedForwardNN


# Mute CPU- and OS-specific warnings from TensorFlow backend of FeedForwardNN
os.environ['KMP_WARNINGS'] = 'off'
os.environ['KMP_AFFINITY'] = 'disabled'

# Header detailing version that's printed out at the beginning of each run and at the top of each log file
header = ['Multisource Input Model Output Security Analysis Suite (MIMOSAS) v1.0.0-release.1', 'Copyright (C) 2019 University of California, Berkeley', 'https://complexity.berkeley.edu/mimosas/']


def start_logger(path):
    """
    Start and return a new logger. Add console and file outputs.
    
    @params:
        path   - Required  : path to desired file output of logger (Str)
    """

    # Create new logger
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    
    # Create output options for logger (file & console)
    file_handler = logging.FileHandler(path)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # Formate output
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Catch all outputs
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG)
    
    # Add output options to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def stop_logger(logger):
    """
    Detatch console and file outputs from logger and delete logger
    
    @params:
        logger    - Required  : logger object to be stopped (Logger)
    """

    for handler in logger.handlers:
        logger.removeHandler(handler)
        del handler
    del logger


def main(main_path):
    """
    Parse command line arguments and run either train or test mode
    """

    # Output program header
    with open('mimosas.txt', 'r') as f:
        contents = f.read()
    print(contents)
    print()
    for line in header:
        print(line)
    print('\n')

    # Parse config file
    parameters = Parameters(main_path)

    # Run indicated models
    if ('DECISION_TREE' in parameters.config.sections()):
        run_decision_tree(parameters)
    if ('RANDOM_FOREST' in parameters.config.sections()):
        run_random_forest(parameters)
    if ('FEED_FORWARD' in parameters.config.sections()):
        run_feed_forward_nn(parameters)


def run_decision_tree(parameters):
    """
    Run decision tree - including train, test, validation if applicable as indicated in the config file

    @params:
        parameters   - Required  : parameter object containing parameters loaded from config file (Parameter)
    """

    # If running in Test mode, check to make sure Load_Model_Path in CONFIG points to a loadable .pkl
    if parameters.config['MAIN']['Mode'] == 'Test':
        try:
            pickle.load(open(parameters.config['DECISION_TREE']['Load_Model_Path'], 'rb'))
        except:
            print('There is no loadable model at', parameters.config['DECISION_TREE']['Load_Model_Path'])
            print('Exiting.')
            exit()

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = os.path.join('.', 'saved_models', 'decision_tree', session)
    if not os.path.exists(path):
        os.makedirs(path)

    # Copy config file used as backup
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # If running in Test mode, copy in the loaded model and its training history; any results will be appended to this
    if parameters.config['MAIN']['Mode'] == 'Test':
        copyfile(parameters.config['DECISION_TREE']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['DECISION_TREE']['Load_Model_Path'])))
        copyfile(os.path.join(os.path.dirname(parameters.config['DECISION_TREE']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

    # Create session logger
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')
        
    # Initialize model object
    logger.info('Running Decision Tree')
    logger.info('')
    models = DecisionTree(parameters, path, logger)

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # In Test mode, MIMOASAS will load a pre-trained model, log its parameters, and score it.
    if ('Test' in parameters.config['MAIN']['Mode']):
        models.load_models(parameters.config['DECISION_TREE']['Load_Model_Path'])

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)
        
        # Exit execution
        logger.info('DONE')
        logger.info('')
        logger.info('')
        stop_logger(logger)
        return None

    # If not in Test mode, do everything
    models.train(X_train, y_train)
    models.test(X_test, y_test)
    models.save_models()
    logger.info('DONE')
    logger.info('')
    logger.info('')
    stop_logger(logger)


def run_random_forest(parameters):
    """
    Run random forest - including train, test, validation if applicable as indicated in the config file
    
    @params:
        parameters   - Required  : parameter object containing parameters loaded from config file (Parameter)
    """

    # If running in Test mode, check to make sure Load_Model_Path in CONFIG points to a loadable .pkl
    if parameters.config['MAIN']['Mode'] == 'Test':
        try:
            pickle.load(open(parameters.config['RANDOM_FOREST']['Load_Model_Path'], 'rb'))
        except:
            print('There is no loadable model at', parameters.config['RANDOM_FOREST']['Load_Model_Path'])
            print('Exiting.')
            exit()

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = os.path.join('.', 'saved_models', 'random_forest', session)
    if not os.path.exists(path):
        os.makedirs(path)

    # Copy config file used as backup
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # If running in Test mode, copy in the loaded model and its training history; any results will be appended to this
    if parameters.config['MAIN']['Mode'] == 'Test':
        copyfile(parameters.config['RANDOM_FOREST']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['RANDOM_FOREST']['Load_Model_Path'])))
        copyfile(os.path.join(os.path.dirname(parameters.config['RANDOM_FOREST']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

    # Create session logger
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')

    # Initialize model object
    logger.info('Running Random Forest')
    logger.info('')
    models = RandomForest(parameters, path, logger)

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # In Test mode, MIMOASAS will load a pre-trained model, log its parameters, and score it.
    if ('Test' in parameters.config['MAIN']['Mode']):
        models.load_models(parameters.config['RANDOM_FOREST']['Load_Model_Path'])

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)
        
        # Exit execution
        logger.info('DONE')
        logger.info('')
        logger.info('')
        stop_logger(logger)
        return None

    # If not in Test mode, do everything
    models.train(X_train, y_train)
    models.test(X_test, y_test)
    models.save_models()
    logger.info('DONE')
    logger.info('')
    logger.info('')
    stop_logger(logger)


def run_feed_forward_nn(parameters):
    """
    Run feed-forward NN - including train, test, validation if applicable as indicated in the config file
    
    @params:
        parameters   - Required  : parameter object containing parameters loaded from config file (Parameter)
    """

    # If running in Test mode, check to make sure Load_Model_Path in CONFIG points to a loadable .pkl
    if parameters.config['MAIN']['Mode'] == 'Test':
        try:
            pickle.load(open(parameters.config['FEED_FORWARD']['Load_Model_Path'], 'rb'))
        except:
            print('There is no loadable model at', parameters.config['FEED_FORWARD']['Load_Model_Path'])
            print('Exiting.')
            exit()

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = os.path.join('.', 'saved_models', 'feed_forward_nn', session)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Copy config file used for records
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # If running in Test mode, copy in the loaded model and its training history; any results will be appended to this
    if parameters.config['MAIN']['Mode'] == 'Test':
        copyfile(parameters.config['FEED_FORWARD']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['FEED_FORWARD']['Load_Model_Path'])))
        copyfile(os.path.join(os.path.dirname(parameters.config['FEED_FORWARD']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

    # Create session logger; if in Test mode, the logger appends to the existing logfile.
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')
        
    # Initialize model object
    logger.info('Running Feed-Forward Neural Network')
    logger.info('')
    models = FeedForwardNN(parameters, path, logger)

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # In Test mode, MIMOASAS will load a pre-trained model, log its parameters, and score it.
    if ('Test' in parameters.config['MAIN']['Mode']):
        models.load_models(parameters.config['FEED_FORWARD']['Load_Model_Path'])

        # # Nicely-formatted model parameters
        # logger.info('Models successfully loaded with the following parameters:')
        # for param, value in models.results['optimized_model'].get_params().items():
        #     logger.info('{}: {}'.format(param, value))
        #     logger.info('')

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)

        # If feature selection is indicated in the CONFIG file, calculate the permutation importance
        # of the input features.
        # NOTE: A lower score means a higher permutation importance (model performed worse with that feature's values shuffled)
        if (parameters.config['FEED_FORWARD']['Feature_Selection'] == 'True'):
            perm_imps = models.permutation_importance(estimator=models.models.best_estimator_, X=X_test, y=y_test)
            perm_imps = perm_imps.sort_values(by=[perm_imps.columns.values[1]], ascending=False)

            # Log/report permutation importance scores for the partially-reduced feature set
            pad = max(max(map(len, perm_imps.index.values)), len('Permuted Feature'))
            logger.info('Permutation Importances of the input features:')
            logger.info('{}  {}  {}'.format('Permuted Feature'.ljust(pad), 'Score', 'Confusion Matrix'))
            for i in perm_imps.index:
                logger.info(
                    '{}: {}  {}'.format(
                        str(i).rjust(pad),
                        ('{:0.4f}'.format(perm_imps.loc[i, perm_imps.columns.values[1]])).rjust(6),
                        str(perm_imps.loc[i, 'confusion_matrix']).replace('\n', ' ').replace('\r', '')
                    )
                )
            logger.info('')

        # Exit function execution
        logger.info('DONE')
        logger.info('')
        logger.info('')
        stop_logger(logger)
        return None

    models.train(X_train, y_train)
    models.test(X_test, y_test)
    models.save_models()

    # If feature selection is indicated in the CONFIG file, use the RFA and RFE
    # algorithms to quantify model performance as a function of input feature set
    if (parameters.config['FEED_FORWARD']['Feature_Selection'] == 'True'):
        models.recursive_feature_addition(X_train, X_test, y_train, y_test)
        models.recursive_feature_elimination(X_train, X_test, y_train, y_test)

    logger.info('DONE')
    logger.info('')
    logger.info('')
    stop_logger(logger)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main_path = os.path.dirname(sys.argv[0]);
        main(main_path)
