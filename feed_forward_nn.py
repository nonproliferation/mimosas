import os
import json
import ast
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier


# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = " "
os.environ['KMP_WARNINGS'] = 'off'
os.environ['KMP_AFFINITY'] = 'disabled'


# Create function returning a compiled network
def construct_network(layers=[8, 4], activation='elu', dropout_rate=0.0, n_features=11, optimizer='adam', loss='categorical_crossentropy', lr=0.001):
    """
    Standalone constructor function for the Feed-Forward Neural Network
    """

    # Start neural network
    network = Sequential()

    # Add layers to model
    for l in layers:

        # Add first hidden layer with n_features dimension specified
        if not network.layers:
            network.add(Dense(units=l, activation=activation, input_shape=(n_features,)))
            network.add(Dropout(dropout_rate))
        else:
            network.add(Dense(units=l, activation=activation))
            network.add(Dropout(dropout_rate))

    # Output layer with a softmax activation function to match one-hot labels
    network.add(Dense(units=2, activation='softmax'))

    # Compile neural network
    network.compile(loss=loss,                  # Loss function; default is Categorical Cross-Entropy
                    optimizer=optimizer,        # Optimization algoritym; default is Adaptive Moments (Adam)
                    metrics=['accuracy'])       # Performance metric reported during training process

    # Set initial learning rate
    keras.backend.set_value(network.optimizer.lr, lr)
    
    # Return compiled network
    return network


class FeedForwardNN:
    def __init__(self, parameters, path, logger):
        """
        Initialize Feed-Forward Neural Network with correct hyperparameters

        @params:
            parameters    - Required  : Parameter object to read config file (Parameter)
            path          - Required  : Path to where log is stored (Str)
            logger        - Required  : Logger object for logging to file and console (Logger)
        """

        # Bind external handling objects and parameters
        self.logger = logger
        self.parameters = parameters
        self.hyperparameters = self.hyperparameter_space()
        self.path = path

        # Compatibility-wrapped estimator constructor
        self.estimator = KerasClassifier(build_fn=construct_network, verbose=1)
        self.models = GridSearchCV(estimator=self.estimator, param_grid=self.hyperparameters, scoring=make_scorer(mcc), cv=int(parameters.config['FEED_FORWARD']['CV_Folds']), return_train_score=True, n_jobs=-1)
        
        # Dict to store results
        self.results = {}

    def hyperparameter_space(self):
        """
        Creates dict with (hyperparameter, values_in_config_file) as (keys, values)
        """

        hyperparameters = dict(
            n_features=[len(self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(','))],
            layers=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Layers']),
            dropout_rate=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Dropout_Rates']),
            activation=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Activation_Fns']),
            optimizer=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Optimizers']),
            loss=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Loss_Fns']),
            lr=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Learning_Rates']),
            epochs=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Max_Epochs']),
            batch_size=ast.literal_eval(self.parameters.config['FEED_FORWARD']['Batch_Sizes'])
        )

        return hyperparameters

    def train(self, X, y):
        """
        Run train mode for Feed-Forward Neural Network

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Training Feed-Forward Neural Network')
        self.logger.info('')

        # if (self.parameters.config['FEED_FORWARD']['Feature_Selection'] == 'True'):
        #     X = self.select_features(X, y, 'Train')

        # Fit models
        self.models.fit(X, y)

        # Store best parameters and optimized model
        self.results['best_params'] = self.models.best_params_  # parameter setting that gave the best results on the hold out data.
        self.results['best_cv_score'] = self.models.best_score_  # mean cross-validated score of the best_estimator
        self.results['hyper_params'] = self.models.cv_results_['params']
        self.results['cv_mean_train_score'] = self.models.cv_results_['mean_train_score']  # average cross-validation training score
        self.results['cv_mean_validate_score'] = self.models.cv_results_['mean_test_score']  # average cross-validation validation score
        self.results['optimized_model'] = self.models.best_estimator_ # Instance of the model with optimized hyperparameters trained on the entirety of the non-test data

        # Log CV Search Parameters
        for mean, params in zip(self.results['cv_mean_validate_score'], self.results['hyper_params']):
            self.logger.info('{:0.5f} with {}'.format(mean, params))
        self.logger.info('Best parameters: ' + str(self.results['best_params']))
        self.logger.info('Best CV score: ' + str(self.results['best_cv_score']))
        # self.logger.info('Hyper parameters: ' + str(self.results['hyper_params']))
        self.logger.info('CV mean train score: ' + str(self.results['cv_mean_train_score']))
        self.logger.info('CV mean validate score: ' + str(self.results['cv_mean_validate_score']))
        # for feature_importance in self.results['feature_importances']:
        #     self.logger.info('Feature importance: ' + str(feature_importance))
        self.logger.info('')

    def test(self, X, y):
        """
        Run test mode for Feed-Forward Neural Network

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Testing Feed-Forward Neural Network')
        self.logger.info('')

        # Test Score
        self.results['test_score'] = self.models.score(X, y)
        self.logger.info('Test score: ' + str(self.results['test_score']))

        # Confusion matrix on test set
        self.results['test_confusion_matrix'] = confusion_matrix(y, self.models.predict(X))
        self.logger.info('Test confusion matrix: ' + str(self.results['test_confusion_matrix']).replace('\n', ' ').replace('\r', ''))
        self.logger.info('')

    def permutation_importance(self, estimator, X, y, scorer=mcc, n_rep=5, n_jobs=1):
        """
        Calculate the permutation importance for each input feature to the estimator

        @params:
            estimator - Required : Trained model with predict method which estimates y based on X
            X         - Required : Pandas dataframe containing input data (Dataframe)
            y         - Required : Pandas dataframe containing label data (Dataframe)
            scorer    - Optional : Metric which accepts true and predicted labels as inputs
            n_rep     - Unused   : Number of times prediction should be made with permuted inputs for each feature
            n_jobs    - Unused   : Number of simultaneous jobs to attempt; -1 uses all available CPU cores
        """

        # Initialize DataFrame to hold results of permutation importance
        perm_imps = pd.DataFrame(columns=['permuted_feature', 'features', 'score ({})'.format(scorer.__name__), 'confusion_matrix'])
        
        # Populate features to be permuted and set its column as row indices
        perm_imps['permuted_feature'] = list(X.columns.values) + ['none']
        perm_imps.set_index('permuted_feature', inplace=True)

        # Iterate model evaluation through permutation of each feature
        # (including no permuted features)
        for feat in perm_imps.index:

            # Copy test set inputs, then permute values of a single column
            X_permuted = X.copy()
            if not feat == 'none':
                np.random.shuffle(X_permuted[feat].values)

            # Evaluate model predictions made with permuted inputs
            score = scorer(y, estimator.predict(X_permuted))
            cm = confusion_matrix(y, estimator.predict(X_permuted))

            # Populate row with evaluation metrics
            perm_imps.loc[feat].fillna({'score ({})'.format(scorer.__name__) : score, 'confusion_matrix' : cm, 'features' : list(X.columns.values)}, inplace=True)

        return perm_imps

    def add_candidate_feat(self, X_train, X_test, y_train, y_test, constructor_kwargs, scorer=mcc):
        """
        Build, fit, and score a model using a subset of input features plus one candidate features

        @params:
            X_train            - Required : Pandas dataframe containing training set input data (Dataframe)
            X_test             - Required : Pandas dataframe containing test set input data (Dataframe)
            y_train            - Required : Pandas dataframe containing training set labels (Dataframe)
            y_test             - Required : Pandas dataframe containing test set labels (Dataframe)
            constructor_kwargs - Required : kwargs parameterizing for the model constructor function, except for n_features
            scorer             - Optional : Metric which accepts true and predicted labels as inputs; used to score model
        """

        # Create compatibility-wrapped model with dim(X_train) input features, then fit and score it
        model = KerasClassifier(build_fn=construct_network, n_features=len(X_train.columns.values), **constructor_kwargs)
        model.fit(X_train, y_train)
        score = scorer(y_test, model.predict(X_test))
        cm = confusion_matrix(y_test, model.predict(X_test))

        return score, cm

    def recursive_feature_addition(self, X_train, X_test, y_train, y_test, scorer=mcc):
        """
        Rank input feature importance using the Recursive Feature Addition algorithm

        @params:
            X_train - Required : Pandas dataframe containing training set input data (Dataframe)
            X_test  - Required : Pandas dataframe containing test set input data (Dataframe)
            y_train - Required : Pandas dataframe containing training set labels (Dataframe)
            y_test  - Required : Pandas dataframe containing test set labels (Dataframe)
            scorer  - Optional : Metric which accepts true and predicted labels as inputs; used to score models
        """

        # Log/report RFA execution
        self.logger.info('Evaluating Feature Importance for Feed-Forward Neural Network using Recursive Feature Addition (RFA)')
        self.logger.info('')

        # Copy of model parameter dict without n_features entry (which we want to dynamically change)
        param_dict = self.results['best_params'].copy()
        del param_dict['n_features']

        # Log/report parameters of optimized model which are shared by all RFA-created models
        self.logger.info('RFA Model Hyperparameters:')
        [self.logger.info('{}: {}'.format(param, param_dict[param])) for param in param_dict]

        # Initialize lists of unused and used columns
        unused_feats = self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')
        used_feats = []

        # DataFrame to hold RFA results
        self.results['recursive_feature_addition'] = pd.DataFrame(columns=['n_features', 'features', 'score ({})'.format(scorer.__name__), 'confusion_matrix'])
        self.results['recursive_feature_addition'].set_index('n_features', inplace=True)

        # Maximum length of feature strings (used during logger column alignment)
        pad = max(max(map(len, unused_feats)), len('Candidate Feature')+1)

        # Until the model has the number of inputs specified in the config file, iterate by adding the best candidate feature
        while len(used_feats) < int(self.parameters.config['FEED_FORWARD']['Features_To_Select']):

            # DataFrame to hold scores of trial additions to existing input features
            candidate_scores = pd.DataFrame(columns=self.results['recursive_feature_addition'].columns, index=unused_feats)

            # Iterate through currently-unincorporated input features,
            # trying each one alongside the already-incorporated ones.
            for feat in candidate_scores.index:

                # Clear keras session variables
                keras.backend.clear_session()

                # Print to terminal (not to logger) the current feature set
                print('Training and scoring model with the following features:', used_feats + [feat])

                # Build, fit, and score model using incorporated features plus the current candidate feature
                score, cm = self.add_candidate_feat(
                    X_train=X_train[used_feats + [feat]],
                    X_test=X_test[used_feats + [feat]],
                    y_train=y_train,
                    y_test=y_test,
                    constructor_kwargs=param_dict,
                    scorer=scorer
                )

                # Record score when candidate feature is includedmda_df = mda_df.sort_values(by=['accuracy'], ascending=False)
                candidate_scores.loc[feat].fillna({'features' : used_feats + [feat], 'score ({})'.format(scorer.__name__) : score, 'confusion_matrix' : cm}, inplace=True)

            # Sort candidate features by model prediction score and identify the best candidate feature
            candidate_scores.sort_values(by=['score ({})'.format(scorer.__name__)], ascending=False, inplace=True)
            best_feat = candidate_scores.head(n=1).index.values[0]

            # Log/report scores of models trained with the candidate features
            self.logger.info('Model prediction scores with remaining candidate features:')
            self.logger.info('{}  Score ({})  {}'.format('Candidate Feature'.ljust(pad), scorer.__name__, 'Confusion Matrix'))
            for i in candidate_scores.index:
                self.logger.info(
                    '{}: {}  {}'.format(
                        i.rjust(pad),
                        ('{:0.4f}'.format(candidate_scores.loc[i, 'score ({})'.format(scorer.__name__)])).rjust(len(scorer.__name__)+8),
                        str(candidate_scores.loc[i, 'confusion_matrix']).replace('\n', ' ').replace('\r', '')
                    )
                )
            self.logger.info('Incorporating the best candidate feature: {}.'.format(best_feat))
            self.logger.info('')

            # Incorporate the candidate feature and record its score
            used_feats.append(best_feat)
            unused_feats.remove(best_feat)
            self.results['recursive_feature_addition'].loc[len(candidate_scores.loc[best_feat, 'features'])] = candidate_scores.loc[best_feat]            

        # Log/report RFA scores
        self.logger.info('Recursive Feature Addition Results:')
        self.logger.info('{}  Score ({})  {}  {}'.format('No. Features'.ljust(pad-7), scorer.__name__, 'Confusion Matrix', 'Features'.ljust(len(', '.join(self.results['recursive_feature_addition'].loc[len(used_feats), 'features'])))))
        for i in self.results['recursive_feature_addition'].index:
            self.logger.info(
                '{}: {}  {}  {}'.format(
                    str(i).rjust(pad-6),
                    ('{:0.4f}'.format(self.results['recursive_feature_addition'].loc[i, 'score ({})'.format(scorer.__name__)])).rjust(len(scorer.__name__)+8),
                    str(self.results['recursive_feature_addition'].loc[i, 'confusion_matrix']).replace('\n', ' ').replace('\r', ''),
                    ', '.join(self.results['recursive_feature_addition'].loc[i, 'features']).ljust(len(', '.join(self.results['recursive_feature_addition'].loc[len(used_feats), 'features'])))
                )
            )
        self.logger.info('')

        return self.results['recursive_feature_addition']

    def recursive_feature_elimination(self, X_train, X_test, y_train, y_test, scorer=mcc):
        """
        Rank input feature importance using the Recursive Feature Elimination algorithm

        @params:
            X_train - Required : Pandas dataframe containing training set input data (Dataframe)
            X_test  - Required : Pandas dataframe containing test set input data (Dataframe)
            y_train - Required : Pandas dataframe containing training set labels (Dataframe)
            y_test  - Required : Pandas dataframe containing test set labels (Dataframe)
            scorer  - Optional : Metric which accepts true and predicted labels as inputs; used to score models
        """

        # Log/report RFA execution
        self.logger.info('Evaluating Feature Importance for Feed-Forward Neural Network using Recursive Feature Elimination (RFE)')
        self.logger.info('')

        # Copy of model parameter dict without n_features entry (which we want to dynamically change)
        param_dict = self.results['best_params'].copy()
        del param_dict['n_features']

        # Log/report parameters of optimized model which are shared by all RFA-created models
        self.logger.info('RFE Model Hyperparameters:')
        [self.logger.info('{}: {}'.format(param, param_dict[param])) for param in param_dict]

        # Initialize list of incorporated input features
        used_feats = self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')

        # DataFrame to hold RFE results
        self.results['recursive_feature_elimination'] = pd.DataFrame(columns=['n_features', 'features', 'score ({})'.format(scorer.__name__), 'confusion_matrix'])
        self.results['recursive_feature_elimination'].set_index('n_features', inplace=True)

        # Maximum length of feature strings (used during logger column alignment)
        pad = max(max(map(len, used_feats)), len('Permuted Feature')+1)

        # Until the model has the number of inputs specified in the config file, iterate by removing the least-informative feature
        while len(used_feats) >= int(self.parameters.config['FEED_FORWARD']['Features_To_Select']):

            # Print to terminal (not to logger) the current feature set
            print('RFE iteration using reduced feature set: {}'.format(used_feats))

            # Training and Evaluation sets with partially-reduced feature set
            X_train_reduced = X_train.copy()[used_feats]
            X_test_reduced = X_test.copy()[used_feats]

            # Clear keras session variables
            keras.backend.clear_session()

            # Create compatibility-wrapped model with dim(X_train_reduced) input features, then fit and score it
            rfe_model = KerasClassifier(build_fn=construct_network, n_features=len(X_train_reduced.columns.values), **param_dict)
            rfe_model.fit(X_train_reduced, y_train)

            # Calculate permutation importance for the input features and sort by score (highest to lowest)
            # of the model when it is used to predict with the data from the index feature shuffled
            perm_imps = self.permutation_importance(estimator=rfe_model, X=X_test_reduced, y=y_test, scorer=mcc, n_rep=5, n_jobs=1)
            perm_imps = perm_imps.sort_values(by=['score ({})'.format(scorer.__name__)], ascending=False)

            # Log/report permutation importance scores for the partially-reduced feature set
            self.logger.info('Permutation Importances of remaining candidate features:')
            self.logger.info('{}  {}  {}'.format('Permuted Feature'.ljust(pad), 'Score ({})'.format(scorer.__name__), 'Confusion Matrix'))
            for i in perm_imps.index:
                self.logger.info(
                    '{}: {}  {}'.format(
                        str(i).rjust(pad),
                        ('{:0.4f}'.format(perm_imps.loc[i, 'score ({})'.format(scorer.__name__)])).rjust(len(scorer.__name__)+8),
                        str(perm_imps.loc[i, 'confusion_matrix']).replace('\n', ' ').replace('\r', '')
                    )
                )
            self.logger.info('')

            # Identify least-informative feature (prediction score suffered the least from shuffling its values;
            # 'none' corresponds to no data shuffled).
            if perm_imps.index[0] == 'none':
                worst_feat = perm_imps.index[1]
            else:
                worst_feat = perm_imps.index[0]

            # Record RFE results for model with n_features (from evaluation with no inputs permuted)
            self.results['recursive_feature_elimination'].loc[len(used_feats)] = perm_imps.loc['none']

            # Remove worst-performing feature from next elimination iteration (if there will be one)
            if len(used_feats) == int(self.parameters.config['FEED_FORWARD']['Features_To_Select']):
                break
            else:
                self.logger.info('Removing least-informative informative input feature: {}'.format(worst_feat))
                self.logger.info('')
                used_feats.remove(worst_feat)

        # Log/report RFE scores
        self.logger.info('Recursive Feature Elimination Results:')
        self.logger.info('{}  Score ({})  {}  {}'.format('No. Features'.ljust(pad-7), scorer.__name__, 'Confusion Matrix', 'Features'))
        for i in self.results['recursive_feature_elimination'].index:
            self.logger.info(
                '{}: {}  {}  {}'.format(
                    str(i).rjust(pad-6),
                    ('{:0.4f}'.format(self.results['recursive_feature_elimination'].loc[i, 'score ({})'.format(scorer.__name__)])).rjust(len(scorer.__name__)+8),
                    str(self.results['recursive_feature_elimination'].loc[i, 'confusion_matrix']).replace('\n', ' ').replace('\r', ''),
                    ', '.join(self.results['recursive_feature_elimination'].loc[i, 'features'])
                )
            )
        self.logger.info('')

        return self.results['recursive_feature_elimination']

    def save_models(self):
        """
        Save models object with pickle serialization
        """

        self.logger.info('Saving Models')
        self.logger.info('')
        pickle.dump(self.models, open(os.path.join(self.path, 'models.pkl'), 'wb'))

    def load_models(self, path):
        """
        Load pickle-serialized models object

        @params:
            path          - Required  : Path to where models object is stored (Str)
        """

        # Load the models object from disk
        self.logger.info('Loading Model')
        self.logger.info('')
        self.models = pickle.load(open(path, 'rb'))
