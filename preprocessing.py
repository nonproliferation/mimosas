import os
import ast
from math import sqrt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_scramble_data(parameters, logger):
    """
    Load and randomly scramble data for train and test split
    
    @params:
        parameters   - Required  : Parameter object to read config settings (Parameter)
        logger       - Required  : Logger object for logging to console and file (Logger)
    """

    # Check data configuration options for some potential pitfalls; raise warnings correspondingly.
    data_warnings(parameters, logger)

    # Load data for training/validation set and for test set
    logger.info('Preparing training/validation set data')
    logger.info('')
    train_data = load_clean_data(parameters, logger, 'TRAINING_DATA')
    logger.info('Preparing test set data')
    logger.info('')
    test_data = load_clean_data(parameters, logger, 'TEST_DATA')
    
    # Parse column headers for input features and labels from CONFIG
    train_cols_to_use = parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')
    test_cols_to_use = parameters.config['TEST_DATA']['Cols_To_Use'].split(',')
    train_label_col = parameters.config['TRAINING_DATA']['Label_Col']
    test_label_col = parameters.config['TEST_DATA']['Label_Col']
    
    # Create train/test splits for data. Separate calls allow for drawing from different datasets
    X_train, X_test_discard, y_train, y_test_discard = train_test_split(train_data[train_cols_to_use], train_data[train_label_col], stratify=train_data[train_label_col], test_size=(1 - float(parameters.config['TRAINING_DATA']['Split_Fraction'])), random_state=int(parameters.config['TRAINING_DATA']['Split_Seed']))
    X_train_discard, X_test, y_train_discard, y_test = train_test_split(test_data[test_cols_to_use], test_data[test_label_col], stratify=test_data[test_label_col], test_size=float(parameters.config['TEST_DATA']['Split_Fraction']), random_state=int(parameters.config['TEST_DATA']['Split_Seed']))
    
    return X_train, X_test, y_train, y_test


def load_clean_data(parameters, logger, mode='TRAINING_DATA'):
    """
    Load and clean data. Used as a helper for load_scramble_data
    
    @params:
        parameters    - Required  : Parameter object to read config settings (Parameter)
        logger        - Required  : Logger object for logging to console and file (Logger)
        mode          - Optional  : Pull data from different locations in config file (Str)
    """

    # Loads background-corrected data if specified in CONFIG and background data file exists; otherwise loads raw input data
    if ('background_correction' in parameters.config[mode]['Data_Options'].split(',')):

        # If the file specified in CONFIG for Background Data does not exist, use the raw input data and issue warning
        if (not os.path.exists(os.path.join(parameters.main_path, parameters.config[mode]['Background_Data']))):
            data = pd.read_csv(os.path.join(parameters.main_path, parameters.config[mode]['Data']))
            logger.warning('Background Correction has been specified in CONFIG, but the background data specified at')
            logger.warning(parameters.config[mode]['Background_Data'])
            logger.warning('does not exist.')
            logger.warning('Loading input data WITHOUT background correction.')
            logger.warning('')
        
        # Otherwise, background-correct the input data
        else:
            data = background_correction(parameters, logger, mode)
    else:
        data = pd.read_csv(os.path.join(parameters.main_path, parameters.config[mode]['Data']))

    data.loc[:, 'utc-time'] = pd.to_datetime(data['time'], unit='s', utc=True)  # convert unix epoch time to utc.

    data_options = parameters.config[mode]['Data_Options'].split(',')
    cols_to_standardize = parameters.config[mode]['Cols_To_Standardize'].split(',')
    cols_to_minmaxscale = parameters.config[mode]['Cols_To_MinMaxScale'].split(',')

    # Identify device ID column (for, e.g., an array of sensors) from config options;
    # If there is none specified, set it to 'sensor_id' and create a corresponding
    # column in the data where all entries are 1.
    if not parameters.config[mode]['Device_ID_Col']:
        parameters.config[mode]['Device_ID_Col'] = 'sensor_id'
        data.loc[:, 'sensor_id'] = 1
    
    if ('remove_outliers' in data_options):
        data = remove_outliers(data, parameters, logger, mode)
    
    for idx in data.groupby([parameters.config[mode]['Device_ID_Col']]).groups.values():
        grouped_data = data.loc[idx]

        if ('standardize' in data_options):  # standardize data based on their device id
            data.loc[idx, :] = standardize_features(grouped_data, cols_to_standardize)

        if ('minmaxscale' in data_options):  # Scale data to [0, 1] based on their device id
            data.loc[idx, :] = minmaxscale_features(grouped_data, cols_to_minmaxscale)

    return data


def data_warnings(parameters, logger):
    """
    Check for configuration options which may produce erroneous results without
    impeding execution of the code. Log any such cases.
    
    @params:
        parameters    - Required  : Parameter object to read config settings (Parameter)
        logger        - Required  : Logger object for logging to console and file (Logger)
    """

    # If the training and test sets are drawn from the same dataset, they should have no elements in common.
    if parameters.config['TRAINING_DATA']['Data'] == parameters.config['TEST_DATA']['Data']:

        # Using a different pseudoRNG seed for splits virtually guarantees common elements.
        if parameters.config['TRAINING_DATA']['Split_Seed'] != parameters.config['TEST_DATA']['Split_Seed']:
            logger.warning('The seed for the pseudoRNG is different for the training and test data configuration options.')
            logger.warning('This is likely to result in shared samples, which bias prediction performance upward.')
            logger.warning('')

        # Even with the same seed, there are some parameter configurations which result in data common to both the training and test sets.
        else:

            # If the sum of the split fractions is greater than one, there will be data common to both the training and test sets.
            split_frac_sum = float(parameters.config['TRAINING_DATA']['Split_Fraction']) + float(parameters.config['TEST_DATA']['Split_Fraction'])
            if not np.isclose(split_frac_sum, 1.0):
                if split_frac_sum > 1.0:
                    logger.warning('The split fractions for the training and test data configurations sum to greater than 1.')
                    logger.warning('This guarantees that the trainig and test datasets will share samples, which bias prediction performance upward.')
                    logger.warning('')
            del split_frac_sum

    # These warnings we want to check for in both the training and test datasets.
    for mode in ['TRAINING_DATA', 'TEST_DATA']:

        # Log/Report warning if input features are being subjected to multiple scaling/normalization functions
        if ('standardize' in parameters.config[mode]['Data_Options'].split(',')) and ('minmaxscale' in parameters.config[mode]['Data_Options'].split(',')):
            if set(parameters.config[mode]['Cols_To_Standardize'].split(',')).intersection(set(parameters.config[mode]['Cols_To_MinMaxScale'].split(','))):
                logger.warning('Input feature(s) in the {} data are being subjected to multiple scaling/normalization functions:'.format(mode))
                for feat in set(parameters.config[mode]['Cols_To_Standardize'].split(',')).intersection(set(parameters.config[mode]['Cols_To_MinMaxScale'].split(','))):
                    logger.warning(feat)
                logger.warning('')

    return None


def standardize_features(data, columns):
    """
    Standardize data in the specified columns to have zero mean and unit variance.
    
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[columns])  # return numpy array
    for idx, col in enumerate(columns):                      # replace original data
        data.loc[:, col] = standardized_data[:, idx]
    return data


def minmaxscale_features(data, columns):
    """
    Perform Min/Max Scaling on the given columns to map them linearly onto [0, 1].
    
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[columns])  # return numpy array
    for idx, col in enumerate(columns):                # replace original data
        data.loc[:, col] = scaled_data[:, idx]
    return data


def remove_outliers(data, parameters, logger, mode):
    """
    Remove outlier data likely to correspond to experimenter perturbation.
          
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """

    # Initialize local variables
    cols_to_cull_on = parameters.config[mode]['Remove_Outlier_Cols'].split(',')
    ids = list(data[parameters.config[mode]['Device_ID_Col']].unique())
    threshold = float(parameters.config[mode]['Outlier_MADs_Threshold'])
    data_by_id = {}
    full = []
    cut = []

    # Iterate through each device in the list of unique devices (e.g., in an array of sensors)
    for device in ids:
        data_by_id[device] = data[data[parameters.config[mode]['Device_ID_Col']] == device]

        # Append number of raw events for the device to 'full'
        full.append(float(len(data_by_id[device].index)))

        # Outlier culling thresholds at +/- (threshold) MADs
        lower_cut = (data_by_id[device].mean() - threshold*data_by_id[device].mad())
        upper_cut = (data_by_id[device].mean() + threshold*data_by_id[device].mad())

        # Remove the data corresponding to outlier values in each feature in
        # 'cols_to_cull_on' from the by-ID dict
        for prod in cols_to_cull_on:
            data_by_id[device] = data_by_id[device][
                (data_by_id[device][prod] >= lower_cut[prod]) &
                (data_by_id[device][prod] <= upper_cut[prod])
                ]

        # Append number of events remaining in the device's data post-cleaning
        cut.append(float(len(data_by_id[device].index)))

    # Concatenate by-device DataFrames back into a single DataFrame
    data_cleaned = pd.concat([data_by_id[device] for device in data_by_id.keys()])

    # Restore chronological indexing to match input data
    data = data_cleaned.sort_values(by=['time']).reset_index()

    # If in Verbose is set to True, report culling statistics
    if (parameters.config['MAIN']['Verbose'] == 'True'):
        logger.info('Cleaning statistics:')
        logger.info('{} total raw data points'.format(int(sum(full))))
        logger.info('{} data points after cleaning'.format(int(sum(cut))))
        logger.info('The fraction of each device\'s events removed by cleaning is:')
        for i in range(len(full)):
            logger.info('ID {}: {:0.4f}'.format(str(ids[i]).rjust(max(list(map(len, map(str, ids))))), (1.0 - cut[i]/full[i])))
        logger.info('')

    # Return data without outliers
    return data


def background_correction(parameters, logger, mode):
    """
    Background correction function which linearly interpolates background data
    and subtracts the interpolated values from the input data.
    
    @params:
        parameters    - Required  : Parameter object to read config settings (Parameter)
        logger        - Required  : Logger object for logging to console and file (Logger)
        mode          - Required  : Pull data from different locations in config file (Str)
    """

    # Load raw input data and background data
    input_data = pd.read_csv(parameters.config[mode]['Data'])
    background = pd.read_csv(parameters.config[mode]['Background_Data'])

    # Subset of background data which temporally spans the raw input data (improves runtime)
    background = background[(background.loc[:, 'time'] > input_data.loc[:, 'time'].min()) & 
                            (background.loc[:, 'time'] < input_data.loc[:, 'time'].max())]

    # Iterate through pairs specified in the CONFIG file as key, value of dict where:
    # (input_col) key = column name in raw input data
    # (bkgd_col)  value = background data column name which will be used to adjust the input_col data
    for input_col, bkgd_col in ast.literal_eval(parameters.config[mode]['Background_Correction_Cols']).items():

        # Piecewise-linear interpolation of the background data evaluated at the times in input_data
        interpolated_bkgd = np.interp(input_data['time'], background['time'], background[bkgd_col])

        # Subtract off interpolated background from measured input data
        input_data.loc[:,  input_col] = input_data.loc[:,  input_col] - interpolated_bkgd
    
    return input_data
