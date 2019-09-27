import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


## Random Forest algorithm
class RandomForest:
    def __init__(self, parameters, path, logger):
        """
        Initialize Random Forest with correct hyperparameters

        @params:
            parameters    - Required  : Parameter object to read config file (Parameter)
            path          - Required  : Path to where log is stored (Str)
            logger        - Required  : Logger object for logging to file and console (Logger)
        """
        ## Logger used to log to console and log file
        self.logger = logger

        # Random Forest Hyperparameters
        n_estimators = list(map(int, parameters.config['RANDOM_FOREST']['N_Estimators'].split(',')))
        max_depth = list(map(int, parameters.config['RANDOM_FOREST']['Depths'].split(',')))
        cv = int(parameters.config['RANDOM_FOREST']['CV_Folds'])
        hyper_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
        
        ## Parameters object used to load from config file
        self.parameters = parameters
        ## Path to log file
        self.path = path
        ## Scikit-Learn estimator used for classification
        self.estimator = RandomForestClassifier(criterion=parameters.config['RANDOM_FOREST']['Classifier_Criterion'])
        ## Grid Search model for hyperparameter tuning
        self.models = GridSearchCV(self.estimator, hyper_parameters, scoring=make_scorer(mcc), cv=cv, return_train_score=True)
        ## Dictionary to store training results
        self.results = {}
        
        self.logger.info('Initializing Random Forest')
        self.logger.info('Using Classifier Criterion: ' + parameters.config['RANDOM_FOREST']['Classifier_Criterion'])
        self.logger.info('')

    def train(self, X, y):
        """
        Run train mode for Random Forest

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Training Random Forest')
        self.logger.info('')

        self.models.fit(X, y)

        self.results['best_params'] = self.models.best_params_  # parameter setting that gave the best results on the hold out data.
        self.results['best_cv_score'] = self.models.best_score_  # mean cross-validated score of the best_estimator
        self.results['hyper_params'] = self.models.cv_results_['params']
        self.results['cv_mean_train_score'] = self.models.cv_results_['mean_train_score']  # average cross-validation training score
        self.results['cv_mean_validate_score'] = self.models.cv_results_['mean_test_score']  # avergage cross-validation validation score
   
        self.logger.info('Best parameters: ' + str(self.results['best_params']))
        self.logger.info('Best CV score: ' + str(self.results['best_cv_score']))
        self.logger.info('Hyper parameters: ' + str(self.results['hyper_params']))
        self.logger.info('CV mean train score: ' + str(self.results['cv_mean_train_score']))
        self.logger.info('CV mean validate score: ' + str(self.results['cv_mean_validate_score']))

        if (self.parameters.config['RANDOM_FOREST']['Feature_Selection'] == 'True'):
            self.select_features(X, y)

    def test(self, X, y):
        """
        Run test mode for Random Forest

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Testing Random Forest')
        self.logger.info('')

        self.results['test_score'] = self.models.score(X, y)
        self.results['test_confusion_matrix'] = confusion_matrix(y, self.models.predict(X))
        
        self.logger.info('Test score: ' + str(self.results['test_score']))
        self.logger.info('Test confusion matrix: ' + str(self.results['test_confusion_matrix']).replace('\n', ' ').replace('\r', ''))
        self.logger.info('')

        if (self.parameters.config['RANDOM_FOREST']['Feature_Selection'] == 'True'):
            self.select_features(X, y)

    def select_features(self, X, y):
        """
        Feature selection function - returns top n features if indicated in config file. Only runs in 'Train' mode

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        # Calculate importances for the input features using the optimized estimator, and
        # store them in a DataFrame sorted from most- to least-informative
        self.results['feature_importances'] = pd.DataFrame(columns=['feature_importance'], index=self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(','))
        self.results['feature_importances'].loc[:, 'feature_importance'] = self.models.best_estimator_.feature_importances_
        self.results['feature_importances'].sort_values(by=['feature_importance'], ascending=False, inplace=True)

        # Log/report feature importance scores
        self.logger.info('{}  {}'.format('Feature'.ljust(11), 'Importance'))
        for i in self.results['feature_importances'].index:
            self.logger.info('{}: {:0.4f}'.format(i.rjust(11), self.results['feature_importances'].loc[i, 'feature_importance']))
        self.logger.info('')

        return self.results['feature_importances']

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
        self.logger.info('Loading Models')
        self.logger.info('')
        self.models = pickle.load(open(path, 'rb'))
