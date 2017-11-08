# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import warnings
from copy import copy as make_copy

import numpy as np
#import pandas as pd

from scipy.stats import mode

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils import shuffle
    from sklearn.model_selection import cross_validate, check_cv
    from sklearn.metrics import accuracy_score, make_scorer
    _IS_SKLEARN_INSTALLED = True
except ImportError:
    warnings.warn("Package sklearn is not installed.")
    _IS_SKLEARN_INSTALLED = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    _IS_LIGHTGBM_INSTALLED = True
except ImportError:
    warnings.warn("Package LightGBM is not installed.")
    _IS_LIGHTGBM_INSTALLED = False
    
try:   
    import xgboost as xgb
    from xgboost import XGBClassifier
    _IS_XGBOOST_INSTALLED = True
except ImportError:
    warnings.warn("Package XGBoost is not installed.")
    _IS_XGBOOST_INSTALLED = False
    
try:   
    from keras.wrappers.scikit_learn import KerasClassifier
    _IS_KERAS_INSTALLED = True
except ImportError:
    warnings.warn("Package Keras is not installed.")
    _IS_KERAS_INSTALLED = False

class Classifier(object):

    """Wraps scikit classifiers"""

    def __init__(self, modelname, num_bagged_est=None, random_state=None, **kwargs):
        """Construct a classifier
    
        Parameters
        ----------
        modelname : str, model name to be used as classifier
            Available models:
            - "XGBoost", 
            - "LightGBM",
            - "KerasClassifier", 
            - "RandomForest", 
            - "ExtraTrees", 
            - "Tree", 
            - "Bagging", 
            - "AdaBoost" 
            - "Linear"
        num_bagged_est: int or None
            Number of estimators to be averaged after bagged fitting. 
            If None then bagged fitting is not performed. 
        random_state:  int, RandomState instance or None, optional, default=None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator; 
            If None, the random number generator is the RandomState instance used by models. 
        **kwargs : default = None
            Parameters of the corresponding classifier.
            Examples : n_estimators, max_depth, ...
        """
        if not _IS_SKLEARN_INSTALLED:
            raise ValueError('Scikit-learn is required for this module')
            
        self.__modelname = modelname
        if self.__modelname == "XGBoost" and not _IS_XGBOOST_INSTALLED:
            raise ValueError('Package XGBoost is not installed.')
        elif self.__modelname == "LightGBM" and not _IS_LIGHTGBM_INSTALLED:
            raise ValueError('Package LightGBM is not installed.')
        elif self.__modelname == "Keras" and not _IS_KERAS_INSTALLED:
            raise ValueError('Package Keras is not installed.')
            
        self.__classif_params = {}

        self.__classifier = None
        self.__set_classifier(self.__modelname)
        self.set_params(**kwargs)
        
        self.__num_bagged_est = num_bagged_est
        if type(self.__num_bagged_est) != int and self.__num_bagged_est is not None:
            raise ValueError("num_bagged_est must be either None or an integer.")
        self.__random_state = random_state
        if type(self.__random_state) != int and self.__random_state is not None:
            raise ValueError("random_state must be either None or an integer.")
        
        self.__fitOK = False
        self.__bagged_est = None
        
    def get_params(self, deep=True):

        params = {}
        params.update({"modelname": self.__modelname,
                       "num_bagged_est": self.__num_bagged_est,
                       "random_state": self.__random_state})
        params.update(self.__classifier.get_params())
        params.update(self.__classif_params)

        return params
    
    def set_params(self, **params):

        self.__fitOK = False
        self.__bagged_est = None

        if 'modelname' in params.keys():
            self.__set_classifier(params['modelname'])
            if self.__modelname == "XGBoost" and not _IS_XGBOOST_INSTALLED:
                raise ValueError('Package XGBoost is not installed.')
            elif self.__modelname == "LightGBM" and not _IS_LIGHTGBM_INSTALLED:
                raise ValueError('Package LightGBM is not installed.')
            elif self.__modelname == "Keras" and not _IS_KERAS_INSTALLED:
                raise ValueError('Package Keras is not installed.')
                    
        if 'num_bagged_est' in params.keys():
            self.__num_bagged_est = params['num_bagged_est']
            if type(self.__num_bagged_est) != int and self.__num_bagged_est is not None:
                raise ValueError("num_bagged_est must be either None or an integer.")
                
        if 'random_state' in params.keys():
            self.__random_state = params['random_state']
            if type(self.__random_state) != int and self.__random_state is not None:
                raise ValueError("random_state must be either None or an integer.")
                
        for k, v in params.items():
            if k == 'modelname' or k == 'num_bagged_est':
                pass
            elif k == 'random_state' and k not in self.__classifier.get_params().keys():
                pass
            else:
                if k not in self.__classifier.get_params().keys():
                    warnings.warn("Invalid parameter for classifier " +
                                  str(self.__modelname) +
                                  ". Parameter IGNORED. Check the list of "
                                  "available parameters with "
                                  "`classifier.get_params().keys()`")
                else:
                    setattr(self.__classifier, k, v)
                    self.__classif_params[k] = v
                    
    def __set_classifier(self, modelname):

        self.__modelname = modelname

        if(modelname == 'XGBoost'):
            self.__classifier = XGBClassifier()

        elif(modelname == "LightGBM"):
            self.__classifier = LGBMClassifier()
        
        elif(modelname == "Keras"):
            self.__classifier = KerasClassifier()
            
        elif(modelname == 'RandomForest'):
            self.__classifier = RandomForestClassifier()

        elif(modelname == 'ExtraTrees'):
            self.__classifier = ExtraTreesClassifier()

        elif(modelname == 'Tree'):
            self.__classifier = DecisionTreeClassifier()

        elif(modelname == "Bagging"):
            self.__classifier = BaggingClassifier()

        elif(modelname == "AdaBoost"):
            self.__classifier = AdaBoostClassifier()

        elif(modelname == "Linear"):
            self.__classifier = LogisticRegression()

        else:
            raise ValueError(
                "Model name invalid. Please choose between LightGBM " +
                "(if installed), XGBoost(if installed), Keras(if installed)," +
                "RandomForest, ExtraTrees, Tree, Bagging, AdaBoost or Linear")
            
    def fit(self, X, y, **kwargs):
        """Fit model. In case num_bagged_est is not None then additionally 
        performing a type of bagging ensamble - ensamble from the same models, 
        but with different seed values/reshuffled data which aims to decrease
        variance of the predictions.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training.
        y : array-like of shape = [n_samples] or [n_samples, n_classes] for KerasClassifier.
            The numerical encoded target for classification tasks.
        **kwargs : default = None
            Additional fitting arguments accepted by model. Not tested.  
            
        Returns
        -------
        object
            self
        """
        if y.ndim > 1:
            self.__num_classes = y.shape[1]
        else:
            self.__num_classes = 2
            
        if self.__num_bagged_est is None:
            self.__classifier.fit(X, y, **kwargs)
            
        else:
            if not hasattr(self.__classifier, 'random_state'):
                 warnings.warn("The classifier " + str(self.__modelname) + 
                               " has no random_state attribute and only random " +
                               " shuffling will be used.")
        
            self.__bagged_est = []
            for i in range(0, self.__num_bagged_est):
                X_shuff, y_shuff = shuffle(X, y, random_state=self.__random_state+i)
                est = self.get_estimator()
                if hasattr(est, 'random_state'):
                    est.set_params(random_state=self.__random_state+i)
                est.fit(X_shuff, y_shuff, **kwargs)
                self.__bagged_est.append(est)
                
        self.__fitOK = True
        
        return self
    
    def predict(self, X):

        """Predicts the target.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            
        Returns
        -------
        array of shape = (n, n_classes)
            The encoded classes to be predicted.
        """

        try:
            if not callable(getattr(self.__classifier, "predict")):
                raise ValueError("predict attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:
            if self.__num_bagged_est is None:
                return self.__classifier.predict(X)
            else:
                bagged_pred = np.zeros((X.shape[0], self.__num_classes, self.__num_bagged_est))
                for c, est in enumerate(self.__bagged_est): 
                    bagged_pred[:,:,c] = est.predict(X)
                    
        else:
            raise ValueError("You must call the fit function before !")
        
        return np.squeeze(mode(bagged_pred, axis=-1)[0], axis=-1)

    def predict_log_proba(self, X):

        """Predicts class log-probabilities for X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Returns
        -------
        y : array of shape = (n, n_classes)
            The log-probabilities for each class
        """

        try:
            if not callable(getattr(self.__classifier, "predict_log_proba")):
                raise ValueError("predict_log_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:
            if self.__num_bagged_est is None:
                return self.__classifier.predict_log_proba(X)
            else:
                bagged_pred_log_proba = np.zeros((X.shape[0], self.__num_classes))
                for est in self.__bagged_est: 
                    bagged_pred_log_proba += est.predict_log_proba(X) / self.__num_bagged_est

        else:
            raise ValueError("You must call the fit function before !")

        return bagged_pred_log_proba

    def predict_proba(self, X):

        """Predicts class probabilities for X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Returns
        -------
        array of shape = (n, n_classes)
            The probabilities for each class
        """

        try:
            if not callable(getattr(self.__classifier, "predict_proba")):
                raise ValueError("predict_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:
            if self.__num_bagged_est is None:
                return self.__classifier.predict_proba(X)
            else:
                bagged_pred_proba = np.zeros((X.shape[0], self.__num_classes))
                for est in self.__bagged_est: 
                    bagged_pred_proba += est.predict_proba(X) / self.__num_bagged_est
                    
        else:
            raise ValueError("You must call the fit function before !")
 
        return bagged_pred_proba
    
    def cross_val_predict_proba(self, X, y, cv=None, scoring=None, **kwargs):
        
        """Performing cross validation hold out predictions for stacking.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training and cv.
        y : array-like of shape = [n_samples] or [n_samples, n_classes] for KerasClassifier.
            The numerical encoded target for classification tasks.
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a StratifiedKFold,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.
        scoring : callable, default: None
                A callable to evaluate the predictions on the cv set.
                None, accuracy score
        **kwargs : default = None
            Additional fitting arguments accepted by model. Not tested.         
        Returns
        -------
        array of shape = (n, n_classes)
            The hold out probabilities for each class
        """
        if y.ndim > 1:
            self.__num_classes = y.shape[1]
        else:
            self.__num_classes = 2
            
        y_pred_proba = np.zeros((X.shape[0], self.__num_classes)) 
        
        cv = check_cv(cv, y, classifier=True)
        n_splits = cv.get_n_splits(X, y)
           
        if scoring is None:
            scoring = make_scorer(accuracy_score)
            
        i = 0 
        score_mean = 0.0
        print("Starting hold out prediction with {} splits.".format(n_splits))
        for train_index, cv_index in cv.split(X, y): 
            X_train = X[train_index]    
            y_train = y[train_index]
            X_cv = X[cv_index]
            y_cv = y[cv_index]
            
            est = self.get_estimator()
            est.fit(X_train, y_train, **kwargs)
            y_pred_proba_cv = est.predict_proba(X_cv)
            
            score = scoring(y_cv, y_pred_proba_cv)                        
            
            print("Train size: {} ::: cv size: {} score (fold {}/{}): {:.4f}".format(len(train_index), len(cv_index), i + 1, n_splits, score)) 
            score_mean += score / float(n_splits)
            
            y_pred_proba[cv_index] = y_pred_proba_cv
            
            i += 1 
        
        print("Mean score: {:.4f}".format(score_mean))    

        return y_pred_proba
        
    def cross_validate(self, X, y, cv=None, scoring=None, **kwargs):
        """Performing a cross validation method.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training.
        y : array-like of shape = [n_samples]
            The numerical encoded target for classification tasks.
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a StratifiedKFold,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.
        scoring : 
            For scikit learn models:
                string, callable, list/tuple, dict or None, default: None
                A single string or a callable to evaluate the predictions on the test set.
                None, the estimator’s default scorer (if available) is used.
            For LightGBM:
                callable or None, optional (default=None)
                Customized evaluation function.
                Note: should return (eval_name, eval_result, is_higher_better) or list of such tuples.
            For XGBoost:
                callable or None, optional (default=None)
                Customized evaluation function.  
        **kwargs : default = None
            Additional fitting arguments.  
            
        Returns
        -------
        object
            self
        """                        
        if isinstance(self.__classifier, LGBMClassifier):
            params = self.__classifier.get_params()
            data = lgb.Dataset(X, label=y)
            #cv = check_cv(cv, y, classifier=True)
            ret = lgb.cv(params, data, feval=scoring, nfold=cv, **kwargs)
        
        elif isinstance(self.__classifier, XGBClassifier):
            params = self.__classifier.get_xgb_params()
            data = xgb.DMatrix(X, label=y)
            #cv = check_cv(cv, y, classifier=True)
            ret = xgb.cv(params, data, feval=scoring, nfold=cv, **kwargs)
            
        elif isinstance(self.__classifier, KerasClassifier):
            ret = self.__cross_validate_keras(X, y, cv=cv, scoring=scoring)

        else:  
            ret = cross_validate(self.__classifier, X, y, cv=cv, scoring=scoring)
        
        return ret
    
    def get_estimator(self):

        return make_copy(self.__classifier)
    
#    def __cross_validate_keras(self, X, y, cv=None, scoring=None):
#        cv = check_cv(cv, y, classifier=True)
#        n_splits = cv.get_n_splits(X, y)
#               
#        for train_index, cv_index in cv.split(X, y): 
#            X_train = X[train_index]    
#            y_train = y[train_index]
#            X_cv = X[cv_index]
#            y_cv = y[cv_index] 
#            
#            self.fit(X_train, y_train, validation_data=(X_cv, y_cv))
#            
#            y_pred = self.predict_proba(X_cv)
#            
#            ret = {'val-score': scoring(y_cv, y_pred)}
#             
#        return ret
