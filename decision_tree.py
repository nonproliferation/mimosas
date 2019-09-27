import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


## Decision Tree algorithm
class DecisionTree:
    def __init__(self, parameters, path, logger):
        """
        Initialize Decision Tree with correct hyperparameters

        @params:
            parameters    - Required  : Parameter object to read config file (Parameter)
            path          - Required  : Path to where log is stored (Str)
            logger        - Required  : Logger object for logging to file and console (Logger)
        """

        ## Logger used to log to console and log file
        self.logger = logger

        # Decision Tree Hyperparameters
        hyper_parameters = {'max_depth': list(map(int, parameters.config['DECISION_TREE']['Max_Depth'].split(',')))}
        cv = int(parameters.config['DECISION_TREE']['CV_Folds'])

        ## Parameters object used to load from config file
        self.parameters = parameters
        ## Path to log file
        self.path = path
        ## Scikit-Learn estimator used for classification
        self.estimator = DecisionTreeClassifier(criterion=parameters.config['DECISION_TREE']['Classifier_Criterion'])
        ## Grid Search model for hyperparameter tuning
        self.model = GridSearchCV(self.estimator, hyper_parameters, scoring=make_scorer(mcc), cv=cv, return_train_score=True)
        ## Dictionary to store training results
        self.results = {}

        self.logger.info('Initializing Decision Tree')
        self.logger.info('Using Classifier Criterion: ' + parameters.config['DECISION_TREE']['Classifier_Criterion'])
        self.logger.info('')

    def train(self, X, y):
        """
        Run train mode for Decision Tree

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Training Decision Tree')
        self.logger.info('')
        if (self.parameters.config['DECISION_TREE']['Feature_Selection'] == 'True'):
            X = self.select_features(X, y, 'Train')

        self.model.fit(X, y)
        self.results['best_params'] = self.model.best_params_  # parameter setting that gave the best results on the hold out data.
        self.results['best_cv_score'] = self.model.best_score_  # mean cross-validated score of the best_estimator
        self.results['hyper_params'] = self.model.cv_results_['params']
        self.results['cv_mean_train_score'] = self.model.cv_results_['mean_train_score']  # average cross-validation training score
        self.results['cv_mean_validate_score'] = self.model.cv_results_['mean_test_score']  # avergage cross-validation validation score
        self.results['feature_importances'] = sorted(zip(self.model.best_estimator_.feature_importances_, self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')), reverse=True)

        self.logger.info('Best parameters: ' + str(self.results['best_params']))
        self.logger.info('Best CV score: ' + str(self.results['best_cv_score']))
        self.logger.info('Hyper parameters: ' + str(self.results['hyper_params']))
        self.logger.info('CV mean train score: ' + str(self.results['cv_mean_train_score']))
        self.logger.info('CV mean validate score: ' + str(self.results['cv_mean_validate_score']))
        for feature_importance in self.results['feature_importances']:
            self.logger.info('Feature importance: ' + str(feature_importance))
        self.logger.info('')
        
    def test(self, X, y):
        """
        Run test mode for Decision Tree

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
        """

        self.logger.info('Testing Decision Tree')
        self.logger.info('')

        self.results['test_score'] = self.model.score(X, y)
        self.results['test_confusion_matrix'] = confusion_matrix(y, self.model.predict(X))

        self.logger.info('Test score: ' + str(self.results['test_score']))
        self.logger.info('Test confusion matrix: ' + str(self.results['test_confusion_matrix']).replace('\n', ' ').replace('\r', ''))
        self.logger.info('')

        if (self.parameters.config['DECISION_TREE']['Feature_Selection'] == 'True'):
            self.select_features(X, y, mode='Test')

    def select_features(self, X, y, mode):
        """
        Feature selection function - returns top n features if indicated in config file. Only runs in 'Train' mode

        @params:
            X       - Required  : Pandas dataframe containing input data (Dataframe)
            y       - Required  : Pandas dataframe containing label data (Dataframe)
            mode    - Required  : Selection mode (Str)
        """

        if (mode == 'Train'):
            temp_model = self.estimator.fit(X, y)
            cols_to_use = self.parameters.config['TRAINING_DATA']['Cols_To_Use'].split(',')
            feature_importances = {cols_to_use[i]:temp_model.feature_importances_[i] for i in range(len(cols_to_use))}
            sorted_features = [k for (k, v) in sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)]
            ## n Features selected
            self.selected_features = sorted_features[:int(self.parameters.config['DECISION_TREE']['Features_To_Select'])]
            self.logger.info('Selected features: ' + str(self.selected_features))
            self.logger.info('')

        if (mode == 'Test'):
            cols_to_use = self.parameters.config['TEST_DATA']['Cols_To_Use'].split(',')
            feature_importances = {cols_to_use[i]:self.model.best_estimator_.feature_importances_[i] for i in range(len(cols_to_use))}
            sorted_features = [k for (k, v) in sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)]
            ## n Features selected
            self.selected_features = sorted_features[:int(self.parameters.config['DECISION_TREE']['Features_To_Select'])]
            self.logger.info('Top {} features: {}'.format(self.parameters.config['DECISION_TREE']['Features_To_Select'], str(self.selected_features)))
            self.logger.info('')

        return X[self.selected_features]

    def save_model(self):
        """
        Save model pickle
        """

        self.logger.info('Saving Model')
        self.logger.info('')
        pickle.dump(self.model, open(os.path.join(self.path, 'model.pkl'), 'wb'))

    def load_model(self, path):
        """
        Load model - called for test mode

        @params:
            path          - Required  : Path to where model is stored (Str)
        """

        # load the model from disk
        self.logger.info('Loading Model')
        self.logger.info('')
        self.model = pickle.load(open(path, 'rb'))
