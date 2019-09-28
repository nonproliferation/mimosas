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
from algorithms.decision_tree import DecisionTree
from algorithms.random_forest import RandomForest
from algorithms.feed_forward_nn import FeedForwardNN


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


def create_session_directory(algorithm):
    """
    Creates directory to hold files for this algorithm's execution session; returns its filepath
    
    @params:
        algorithm  - Required  : name of the algorithm being run this session
    """

    # Path string based on UTC time (second precision) at the time of creation
    session = datetime.datetime.utcnow().strftime('%Y_%b_%d_%Hh_%Mm_%Ss')
    path = os.path.join('.', 'saved_models', algorithm, session)

    # Create directory at path if it doesn't already exist
    if not os.path.exists(path):
        os.makedirs(path)

    return path


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

    # Create session directory
    path = create_session_directory('decision_tree')

    # Copy session config file used as backup
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # Create session logger
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')

    # Initialize models object
    models = DecisionTree(parameters, path, logger)

    # Check for loadable models
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        # If the Load_Model_Path parameter in CONFIG is left blank, don't bother trying to load a model.
        if parameters.config['DECISION_TREE']['Load_Model_Path']:

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['DECISION_TREE']['Load_Model_Path'])

            # If load fails, train from scratch instead.
            except:
                logger.info('There is no loadable models object at: ' + parameters.config['DECISION_TREE']['Load_Model_Path'])
                logger.info('MIMOSAS will train new model(s) from scratch.')
                logger.info('')

            # If load is successful, prepare to continue training.
            else:

                # Report successful load to terminal; this session's logfile will be overwritten with log from model's history.
                print('Loadable models object found at', parameters.config['DECISION_TREE']['Load_Model_Path'])

                # Copy loaded model to the session directory
                copyfile(parameters.config['DECISION_TREE']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['DECISION_TREE']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['DECISION_TREE']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

    else:
        if 'Test' in parameters.config['MAIN']['Mode'].split(','):

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['DECISION_TREE']['Load_Model_Path'])

            # If load fails, exit function because there is nothing to test.
            except:
                logger.info('There is no loadable models object to test at: ' + parameters.config['DECISION_TREE']['Load_Model_Path'])
                logger.info('Exiting Decision Tree.')
                stop_logger(logger)
                return None

            # If load is successful, prepare to test.
            else:

                # Copy loaded model to the session directory
                copyfile(parameters.config['DECISION_TREE']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['DECISION_TREE']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['DECISION_TREE']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

                # Report successful load
                logger.info('Models object successfully loaded.')

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # Train, if specified in CONFIG
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        logger.info('Running DecisionTree in Train Mode.')
        logger.info('')

        models.train(X_train, y_train)

        # Save model, if specified in CONFIG
        # Note: Overwrites saved model (check if True)
        if parameters.config['MAIN']['Save'] == 'True':
            models.save_models()

        # Exit Train Mode execution
        logger.info('DONE TRAINING.')
        logger.info('')
        logger.info('')

    # Test, if specified in CONFIG
    if ('Test' in parameters.config['MAIN']['Mode']):

        logger.info('Running DecisionTree in Test Mode.')
        logger.info('')

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)
        
        # Exit Test Mode execution
        logger.info('DONE TESTING.')
        logger.info('')
        logger.info('')

    # Stop logger
    stop_logger(logger)


def run_random_forest(parameters):
    """
    Run random forest - including train, test, validation if applicable as indicated in the config file
    
    @params:
        parameters   - Required  : parameter object containing parameters loaded from config file (Parameter)
    """

    # Create session directory
    path = create_session_directory('random_forest')

    # Copy session config file used as backup
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # Create session logger
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')

    # Initialize models object
    models = RandomForest(parameters, path, logger)

    # Check for loadable models
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        # If the Load_Model_Path parameter in CONFIG is left blank, don't bother trying to load a model.
        if parameters.config['RANDOM_FOREST']['Load_Model_Path']:

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['RANDOM_FOREST']['Load_Model_Path'])

            # If load fails, train from scratch instead.
            except:
                logger.info('There is no loadable models object at: ' + parameters.config['RANDOM_FOREST']['Load_Model_Path'])
                logger.info('MIMOSAS will train new model(s) from scratch.')
                logger.info('')

            # If load is successful, prepare to continue training.
            else:

                # Copy loaded model to the session directory
                copyfile(parameters.config['RANDOM_FOREST']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['RANDOM_FOREST']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['RANDOM_FOREST']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

    else:
        if 'Test' in parameters.config['MAIN']['Mode'].split(','):

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['RANDOM_FOREST']['Load_Model_Path'])

            # If load fails, exit function because there is nothing to test.
            except:
                logger.info('There is no loadable models object to test at: ' + parameters.config['RANDOM_FOREST']['Load_Model_Path'])
                logger.info('Exiting Random Forest.')
                stop_logger(logger)
                return None


            # If load is successful, prepare to test.
            else:

                # Report successful load to terminal; this session's logfile will be overwritten with log from model's history.
                print('Loadable models object found at', parameters.config['RANDOM_FOREST']['Load_Model_Path'])

                # Copy loaded model to the session directory
                copyfile(parameters.config['RANDOM_FOREST']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['RANDOM_FOREST']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['RANDOM_FOREST']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

                # Report successful load
                logger.info('Models object successfully loaded.')

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # Train, if specified in CONFIG
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        logger.info('Running RandomForest in Train Mode.')
        logger.info('')

        models.train(X_train, y_train)

        # Save model, if specified in CONFIG
        # Note: Overwrites saved model (check if True)
        if parameters.config['MAIN']['Save'] == 'True':
            models.save_models()

        # Exit Train Mode execution
        logger.info('DONE TRAINING.')
        logger.info('')
        logger.info('')

    # Test, if specified in CONFIG
    if ('Test' in parameters.config['MAIN']['Mode']):

        logger.info('Running RandomForest in Test Mode.')
        logger.info('')

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)
        
        # Exit Test Mode execution
        logger.info('DONE TESTING.')
        logger.info('')
        logger.info('')

    # Stop logger
    stop_logger(logger)


def run_feed_forward_nn(parameters):
    """
    Run feed-forward NN - including train, test, validation if applicable as indicated in the config file
    
    @params:
        parameters   - Required  : parameter object containing parameters loaded from config file (Parameter)
    """

    # Create session directory
    path = create_session_directory('feed_forward_nn')

    # Copy session config file used as backup
    copyfile(parameters.config_file, os.path.join(path, 'used_conf.config'))

    # Create session logger
    logger = start_logger(os.path.join(path, 'training.log'))
    for line in header:
        logger.info(line)
    logger.info('')

    # Initialize models object
    models = FeedForwardNN(parameters, path, logger)

    # Check for loadable models
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        # If the Load_Model_Path parameter in CONFIG is left blank, don't bother trying to load a model.
        if parameters.config['FEED_FORWARD']['Load_Model_Path']:

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['FEED_FORWARD']['Load_Model_Path'])

            # If load fails, train from scratch instead.
            except:
                logger.info('There is no loadable models object at: ' + parameters.config['FEED_FORWARD']['Load_Model_Path'])
                logger.info('MIMOSAS will train new model(s) from scratch.')
                logger.info('')

            # If load is successful, prepare to continue training.
            else:

                # Copy loaded model to the session directory
#                 copyfile(parameters.config['FEED_FORWARD']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['FEED_FORWARD']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['FEED_FORWARD']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

                # Report successful load
                logger.info('Models object successfully loaded.')

    else:
        if 'Test' in parameters.config['MAIN']['Mode'].split(','):

            # Try to load a model from the specified path.
            try:
                models.load_models(parameters.config['FEED_FORWARD']['Load_Model_Path'])

            # If load fails, exit function because there is nothing to test.
            except:
                logger.info('There is no loadable models object to test at: ' + parameters.config['FEED_FORWARD']['Load_Model_Path'])
                logger.info('Exiting FeedForwardNN.')
                stop_logger(logger)
                return None


            # If load is successful, prepare to test.
            else:

                # Copy loaded model to the session directory
                copyfile(parameters.config['FEED_FORWARD']['Load_Model_Path'], os.path.join(path, os.path.basename(parameters.config['FEED_FORWARD']['Load_Model_Path'])))
                
                # Copy loaded training log
                copyfile(os.path.join(os.path.dirname(parameters.config['FEED_FORWARD']['Load_Model_Path']), 'training.log'), os.path.join(path, 'training.log'))

                # Nicely-formatted parameters of the most successful classifier in the loaded models object
                logger.info('Models successfully loaded; the best-performing model has the following parameters:')
                for param, value in models.models.best_estimator_.get_params().items():
                    logger.info('{}: {}'.format(param, value))
                logger.info('')

    # Load data
    X_train, X_test, y_train, y_test = load_scramble_data(parameters, logger)

    # Train, if specified in CONFIG
    if 'Train' in parameters.config['MAIN']['Mode'].split(','):

        logger.info('Running FeedForwardNN in Train Mode.')
        logger.info('')

        models.train(X_train, y_train)

        # Save model, if specified in CONFIG
        # Note: Overwrites saved model (check if True)
        if parameters.config['MAIN']['Save'] == 'True':
            models.save_models()

        logger.info('DONE TRAINING.')
        logger.info('')
        logger.info('')

    # Test, if specified in CONFIG
    if ('Test' in parameters.config['MAIN']['Mode']):

        logger.info('Running FeedForwardNN in Test Mode.')
        logger.info('')

        # Score the model's prediction performance on the test set.
        models.test(X_test, y_test)

        # If feature selection is indicated in the CONFIG file, use the RFA and RFE
        # algorithms to quantify model performance as a function of input feature set
        if (parameters.config['FEED_FORWARD']['Feature_Selection'] == 'True'):
            models.recursive_feature_addition(X_train, X_test, y_train, y_test)
            models.recursive_feature_elimination(X_train, X_test, y_train, y_test)
        
        # Exit execution
        logger.info('DONE TESTING.')
        logger.info('')
        logger.info('')

    # Stop logger
    stop_logger(logger)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main_path = os.path.dirname(sys.argv[0]);
        main(main_path)
