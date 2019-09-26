import os
import sys
import logging
import datetime
import warnings
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
header = ['Multi-Modal Analysis Suite (MMAS) v1.0.0-release.1', 'Copyright (C) 2019 University of California, Berkeley', 'https://complexity.berkeley.edu/mimosas/']


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

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = './saved_models/decision_tree/' + session + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Create session logger
    logger = start_logger(path + 'training.log')
    for line in header:
        logger.info(line)
    logger.info('')
    
    # Copy config file used as backup
    copyfile(parameters.config_file, path + 'used_conf.config')
        
    # Initialize and run model
    logger.info('Running Decision Tree')
    logger.info('')
    model = DecisionTree(parameters, path, logger)
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)
    if (parameters.config['DECISION_TREE']['Load_Model_Path']):
        model.load_model(arameters.config['DECISION_TREE']['Load_Model_Path'])
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    model.save_model()
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

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = './saved_models/random_forest/' + session + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Create session logger
    logger = start_logger(path + 'training.log')
    for line in header:
        logger.info(line)
    logger.info('')
    
    # Copy config file used as backup
    copyfile(parameters.config_file, path + 'used_conf.config')
        
    # Initialize and run model
    logger.info('Running Random Forest')
    logger.info('')
    model = RandomForest(parameters, path, logger)
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)
    if (parameters.config['RANDOM_FOREST']['Load_Model_Path']):
        model.load_model(arameters.config['RANDOM_FOREST']['Load_Model_Path'])
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    model.save_model()
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

    # Generate session folder
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = './saved_models/feed_forward_nn/' + session + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Create session logger
    logger = start_logger(path + 'training.log')
    for line in header:
        logger.info(line)
    logger.info('')
    
    # Copy config file used as backup
    copyfile(parameters.config_file, path + 'used_conf.config')
        
    # Initialize and run model
    logger.info('Running Feed-Forward Neural Network')
    logger.info('')
    model = FeedForwardNN(parameters, path, logger)

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    if (parameters.config['FEED_FORWARD']['Load_Model_Path']):
        model.load_model(parameters.config['FEED_FORWARD']['Load_Model_Path'])

        print(model.results['optimized_model'].get_params())
        model.test(X_test, y_test)
        logger.info('DONE')
        logger.info('')
        logger.info('')
        stop_logger(logger)
        exit()

    model.train(X_train, y_train)
    model.test(X_test, y_test)
    model.save_model()

    # If feature selection is indicated in the CONFIG file, use the RFA and RFE
    # algorithms to quantify model performance as a function of input feature set
    if (parameters.config['FEED_FORWARD']['Feature_Selection'] == 'True'):
        model.recursive_feature_addition(X_train, X_test, y_train, y_test)
        model.recursive_feature_elimination(X_train, X_test, y_train, y_test)

    logger.info('DONE')
    logger.info('')
    logger.info('')
    stop_logger(logger)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main_path = os.path.dirname(sys.argv[0]);
        main(main_path)
