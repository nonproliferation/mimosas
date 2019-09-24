import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def load_scramble_data(parameters, logger):
    """
    Load and randomly scramble data for train and test split
    
    @params:
        parameters   - Required  : Parameter object to read config settings (Parameter)
        logger       - Required  : Logger object for logging to console and file (Logger)
    """
    logger.info('Preparing training data')
    logger.info('')
    train_data = load_clean_data(parameters, logger, 'TRAINING_DATA')
    logger.info('Preparing evaluation / test data')
    logger.info('')
    eval_data = load_clean_data(parameters, logger, 'EVALUATION_DATA')
    
    train_cols_to_use = parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')
    eval_cols_to_use = parameters.config['EVALUATION_DATA']['Cols_To_Use'].split(',')
    train_label_col = parameters.config['TRAINING_DATA']['Label_Col']
    eval_label_col = parameters.config['EVALUATION_DATA']['Label_Col']
    
    X_train, X_eval_discard, y_train, y_eval_discard = train_test_split(train_data[train_cols_to_use], train_data[train_label_col], stratify=train_data[train_label_col], test_size=(1 - float(parameters.config['TRAINING_DATA']['Split_Fraction'])), random_state=int(parameters.config['TRAINING_DATA']['Split_Seed']))
    X_train_discard, X_eval, y_train_discard, y_eval = train_test_split(eval_data[eval_cols_to_use], eval_data[eval_label_col], stratify=eval_data[eval_label_col], test_size=float(parameters.config['EVALUATION_DATA']['Split_Fraction']), random_state=int(parameters.config['EVALUATION_DATA']['Split_Seed']))
    
    return X_train, X_eval, y_train, y_eval

def load_clean_data(parameters, logger, mode='TRAINING_DATA'):
    """
    Load and clean data. Used as a helper for load_scramble_data
    
    @params:
        parameters    - Required  : Parameter object to read config settings (Parameter)
        logger        - Required  : Logger object for logging to console and file (Logger)
        mode          - Optional  : Pull data from different locations in config file (Str)
    """
    if (not os.path.exists(os.path.join(parameters.main_path, parameters.config[mode]['Weather_Correction']))):
        data = pd.read_csv(os.path.join(parameters.main_path, parameters.config[mode]['Data']))
    else:
        data = weather_correction(os.path.join(parameters.main_path, parameters.config[mode]['Data']), os.path.join(parameters.main_path, parameters.config['TRAINING_DATA']['Weather_Correction']))
    
    data.loc[:, 'utc-time'] = pd.to_datetime(data['time'], unit='s', utc=True)  # convert unix epoch time to utc
    # data.loc[:, 'temp'] = data['temp'] / 100  # convert temperature to floating point
    
    data_options = parameters.config[mode]['Data_Options'].split(',')
    cols_to_standardize = parameters.config[mode]['Cols_To_Standardize'].split(',')
    cols_to_normalize = parameters.config[mode]['Cols_To_Normalize'].split(',')
    
    if ('remove_outliers' in data_options):
        data = remove_outliers(data, parameters, logger, mode)
    
    for idx in data.groupby(['id']).groups.values():
        grouped_data = data.loc[idx]

        if ('standardize' in data_options):  # standardize data based on their device id
            data.loc[idx, :] = standardize_features(grouped_data, cols_to_standardize)

        if ('normalize' in data_options):  # normalize data based on their device id
            data.loc[idx, :] = normalize_features(grouped_data, cols_to_normalize)

    return data

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


def normalize_features(data, columns):
    """
    Perform normalization on the given columns.
    
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """
    normalizer = MinMaxScaler()
    normalized_data = normalizer.fit_transform(data[columns])  # return numpy array
    for idx, col in enumerate(columns):                        # replace original data
        data.loc[:, col] = normalized_data[:, idx]
    return data
        
def remove_outliers(data, parameters, logger, mode):
    """
    Remove outlier data likely to correspond to experimenter perturbation.
          
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """
    
    device_col = 'id'
    use_magnitudes = False
    cols_to_cull_on = parameters.config[mode]['Remove_Outlier_Cols'].split(',')

    # Initialize local variables
    data_by_id = {}
    full = []
    cut = []
    ids = list(data[device_col].unique())
    threshold = 3.0

    for device in ids:
        data_by_id[device] = data[data[device_col] == device]

        # Append number of raw BLOBs for the device to 'full'
        full.append(float(len(data_by_id[device].index)))

        # If we want to cull on outliers in accelerometer data only, use the
        # magnitude of the acceleration vector instead of its components. If
        # the accelerometer data have already been converted to magnitudes:
        if cols_to_cull_on == ['acc_x', 'acc_y', 'acc_z'] and use_magnitudes:
            
            # Magnitude of the accerometer measurements, expressed in Mean Average
            # Deviations (MADs) about the mean.
            acc = (data_by_id[device]['acc'] - data_by_id[device]['acc'].mean()) / data_by_id[device]['acc'].mad()

            # Keep data within culling thresholds
            data_by_id[device] = data_by_id[device].loc[acc[(acc > -1.0*threshold) & (acc < 1.0*threshold)].index]

        # If the to-magnitude conversion was not previously performed, but we
        # use accelerometer-only values to cull:
        elif cols_to_cull_on == ['acc_x', 'acc_y', 'acc_z'] and not use_magnitudes:

            # Magnitude of the accerometer measurements, expressed in Mean Average
            # Deviations (MADs) about the mean.
            acc = (data_by_id[device]['acc_x']**2 + data_by_id[device]['acc_y']**2 + data_by_id[device]['acc_z']**2).apply(sqrt)
            acc = (acc - acc.mean()) / acc.mad()

            # Keep data within culling thresholds
            data_by_id[device] = data_by_id[device].loc[acc[(acc > -1.0*threshold) & (acc < 1.0*threshold)].index]

        # If we cull on non-accelerometer values
        else:
            # Outlier culling thresholds at +/- (threshold) MADs
            lower_cut = (
                data_by_id[device].mean() - threshold*data_by_id[device].mad()
                )
            upper_cut = (
                data_by_id[device].mean() + threshold*data_by_id[device].mad()
                )

            # Remove the data corresponding to outlier values in each feature in
            # 'cull_on' from the by-ID dict
            for prod in cols_to_cull_on:
                data_by_id[device] = data_by_id[device][
                    (data_by_id[device][prod] >= lower_cut[prod]) &
                    (data_by_id[device][prod] <= upper_cut[prod])
                    ]

        # Append number of BLOBs remaining in the device's data post-cleaning
        cut.append(float(len(data_by_id[device].index)))

    # Concatenate by-device DataFrames back into a single DataFrame
    data_cleaned = pd.concat(
        [data_by_id[device] for device in data_by_id.keys()]
        )

    # Restore chronological indexing to match input data
    data = data_cleaned.sort_values(by=['time']).reset_index()

    # Report culling statistics, if flagged with -v
    if (parameters.config['MAIN']['Verbose'] == 'True'):
        logger.info('Cleaning statistics:')
        logger.info('{} total raw data points'.format(int(sum(full))))
        logger.info('{} data points after cleaning'.format(int(sum(cut))))
        logger.info('The fraction of each device\'s events removed by cleaning is:')
        for i in range(len(full)):
            logger.info('ID {:5d}: {:0.4f}'.format(ids[i], (1.0 - cut[i]/full[i])))
        logger.info('')

    # Return data without outliers
    return data
    
def weather_correction(data_path, weather_correction_path):
    """
    Weather correction function
    
    @params:
        data_path                 - Required  : Path to data (Str)
        weather_correctin_path    - Required  : Path to weather correction (Str)
    """
    weather_data = pd.read_csv(weather_correction_path)
    measured_data = pd.read_csv(data_path)
    
    weather_data['LocalPressure'] = pd.Series(weather_data['LocalPressure']*100)  # convert weather pressures in hPa to Pa
    weather_data['Temp'] = pd.Series((weather_data['Temp'] - 273.15)*100) # convert weather temps in K to degrees C*100
    weather_data['RHx'] = pd.Series((weather_data['RHx']*1024)) # convert weather %RH to %RH*1024

    measured_data['pressure'] = weather_helper(measured_data, 'pressure', weather_data, 'LocalPressure')
    measured_data['temp'] = weather_helper(measured_data, 'temp', weather_data, 'Temp')
    measured_data['humidity'] = weather_helper(measured_data, 'humidity', weather_data, 'RHx')
    
    return measured_data
    
def weather_helper(measured_data, m_v, weather_data, w_v):
    """
    Weather adjustment helper function that returns corrected and interpolated values.
    
    @params:
        measured_data   - Required  : Pandas dataframe containing data and time (Dataframe)
        m_v             - Required  : Measured values column of pandas dataframe (Str)
        weather_data    - Required  : Pandas dataframe containing data and time (Dataframe)
        w_v             - Required  : Weather values column of pandas dataframe (Str)
    """
    measured_times = pd.Series(measured_data['time'])
    weather_times = pd.Series(weather_data['Date'])

    measured_values = pd.Series(measured_data[m_v])
    weather_values = pd.Series(weather_data[w_v])

    first_measured = measured_times[0]
    weather_time_index = 0
    while True:
        if weather_times[weather_time_index + 1] > first_measured:
            break
        weather_time_index += 1

    weather_times = weather_times[weather_time_index:]
    weather_times = weather_times.reset_index(drop=True)
    weather_values = weather_values[weather_time_index:]
    weather_values = weather_values.reset_index(drop=True)

    interpolated_weather_values = np.interp(measured_times, weather_times, weather_values)
    corrected_values = measured_values - interpolated_weather_values

    return corrected_values
