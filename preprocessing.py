import os
import ast
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    if (not os.path.exists(os.path.join(parameters.main_path, parameters.config[mode]['Background_Data']))):
        data = pd.read_csv(os.path.join(parameters.main_path, parameters.config[mode]['Data']))
    else:
        data = background_correction(parameters, logger, mode)

    data.loc[:, 'utc-time'] = pd.to_datetime(data['time'], unit='s', utc=True)  # convert unix epoch time to utc

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

    # Log/Report warning if input features are being subjected to multiple scaling/normalization functions
    if ('standardize' in data_options) and ('minmaxscale' in data_options):
        if set(cols_to_standardize).intersection(set(cols_to_minmaxscale)):
            logger.warning('Input feature(s) are being subjected to multiple scaling/normalization functions:')
            for feat in set(cols_to_standardize).intersection(set(cols_to_minmaxscale)):
                logger.warning(feat)
            logger.warning('')
    
    for idx in data.groupby([parameters.config[mode]['Device_ID_Col']]).groups.values():
        grouped_data = data.loc[idx]

        if ('standardize' in data_options):  # standardize data based on their device id
            data.loc[idx, :] = standardize_features(grouped_data, cols_to_standardize)

        if ('minmaxscale' in data_options):  # Scale data to [0, 1] based on their device id
            data.loc[idx, :] = minmaxscale_features(grouped_data, cols_to_minmaxscale)

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


def minmaxscale_features(data, columns):
    """
    Perform Min/Max Scaling on the given columns to map them linearly onto [0, 1].
    
    @params:
        data        - Required  : the data from a single device which may contain remove_outliers (list)
        columns     - Required  : Params object representing model parameters (Params)
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[columns])  # return numpy array
    for idx, col in enumerate(columns):                        # replace original data
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

    # Report culling statistics, if flagged with -v
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
    
def weather_correction(data_path, weather_correction_path):
    """
    Weather correction function
    
    @params:
        data_path                 - Required  : Path to data (Str)
        weather_correctin_path    - Required  : Path to weather correction (Str)
    """

    weather_data = pd.read_csv(weather_correction_path)
    measured_data = pd.read_csv(data_path)

    # measured_data.loc[:, 'temp'] = measured_data['temp'] / 100  # convert temperature to floating point
    
    weather_data['LocalPressure'] = pd.Series(weather_data['LocalPressure']*100)  # convert weather pressures in hPa to Pa
    weather_data['Temp'] = pd.Series((weather_data['Temp'] - 273.15)*100) # convert weather temps in K to degrees C*100
    print(weather_data['Temp'].head(n=10))
    print(measured_data['temp'].head(n=10))
    exit()
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

    # Convert the canary and weather data to pandas series
    measured_values = pd.Series(measured_data[m_v])
    weather_values = pd.Series(weather_data[w_v])

    # initialize values for the scan through weather data until the beginning of canary data is found
    first_measured = measured_times[0]
    weather_time_index = 0

    # Find the index (in time) of the last background data prior to the first input data timestep
    while True:
        # If the next timestep for the weather data is greater than the first timestep for the canary data, exit the loop
        if weather_times[weather_time_index + 1] > first_measured:
            break
        weather_time_index += 1

    # Reduce the weather timesteps and data to only points occurring past the index obtained
    # with the while loop and reset their pandas indices
    weather_times = weather_times[weather_time_index:]
    weather_times = weather_times.reset_index(drop=True)
    weather_values = weather_values[weather_time_index:]
    weather_values = weather_values.reset_index(drop=True)

    # interpolate values for weather data at the canara data timesteps
    interpolated_weather_values = np.interp(measured_times, weather_times, weather_values)

    # Subtract off interpolated background data
    corrected_values = measured_values - interpolated_weather_values

    return corrected_values


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
