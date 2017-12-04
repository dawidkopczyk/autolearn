# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import warnings
from copy import copy as make_copy

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.utils import shuffle
    from sklearn.model_selection import cross_validate, check_cv
    from sklearn.metrics import accuracy_score, make_scorer
    _IS_SKLEARN_INSTALLED = True
except ImportError:
    warnings.warn("Package sklearn is not installed.")
    _IS_SKLEARN_INSTALLED = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    _IS_LIGHTGBM_INSTALLED = True
except ImportError:
    warnings.warn("Package LightGBM is not installed.")
    _IS_LIGHTGBM_INSTALLED = False
    
try:   
    import xgboost as xgb
    from xgboost import XGBRegressor
    _IS_XGBOOST_INSTALLED = True
except ImportError:
    warnings.warn("Package XGBoost is not installed.")
    _IS_XGBOOST_INSTALLED = False
    
try:   
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    _IS_KERAS_INSTALLED = True
except ImportError:
    warnings.warn("Package Keras is not installed.")
    _IS_KERAS_INSTALLED = False

class Regressor(object):

    """Wraps scikit regressors"""

    def __init__(self, modelname='Linear', num_bagged_est=None, random_state=None, **kwargs):
        """Construct a regressor
    
        Parameters
        ----------
        modelname : str, model name to be used as regressor
            Available models:
            - "XGBoost", 
            - "LightGBM",
            - "Keras", 
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
            Parameters of the corresponding regressor.
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

        self.__regressor = None
        self.__set_regressor(self.__modelname)
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
        params.update(self.__regressor.get_params())

        return params
    
    def set_params(self, **params):

        self.__fitOK = False
        self.__bagged_est = None

        if 'modelname' in params.keys():
            self.__set_regressor(params['modelname'])
            del params['modelname']
            if self.__modelname == "XGBoost" and not _IS_XGBOOST_INSTALLED:
                raise ValueError('Package XGBoost is not installed.')
            elif self.__modelname == "LightGBM" and not _IS_LIGHTGBM_INSTALLED:
                raise ValueError('Package LightGBM is not installed.')
            elif self.__modelname == "Keras" and not _IS_KERAS_INSTALLED:
                raise ValueError('Package Keras is not installed.')
                    
        if 'num_bagged_est' in params.keys():
            self.__num_bagged_est = params['num_bagged_est']
            del params['num_bagged_est']
            if type(self.__num_bagged_est) != int and self.__num_bagged_est is not None:
                raise ValueError("num_bagged_est must be either None or an integer.")
                
        if 'random_state' in params.keys():
            self.__random_state = params['random_state']
            if 'random_state' not in self.__regressor.get_params().keys():
                del params['random_state']
            if type(self.__random_state) != int and self.__random_state is not None:
                raise ValueError("random_state must be either None or an integer.")
        
        if 'build_fn' in params.keys() and self.get_estimator_name == 'Keras':
            setattr(self.__regressor, 'build_fn', params['build_fn'])
            del params['build_fn']
            
        self.__regressor.set_params(**params)
                    
    def __set_regressor(self, modelname):

        self.__modelname = modelname

        if(modelname == 'XGBoost'):
            self.__regressor = XGBRegressor()

        elif(modelname == "LightGBM"):
            self.__regressor = LGBMRegressor()
        
        elif(modelname == "Keras"):
            self.__regressor = KerasRegressor(build_fn=Sequential())
            
        elif(modelname == 'RandomForest'):
            self.__regressor = RandomForestRegressor()

        elif(modelname == 'ExtraTrees'):
            self.__regressor = ExtraTreesRegressor()

        elif(modelname == 'Tree'):
            self.__regressor = DecisionTreeRegressor()

        elif(modelname == "Bagging"):
            self.__regressor = BaggingRegressor()

        elif(modelname == "AdaBoost"):
            self.__regressor = AdaBoostRegressor()

        elif(modelname == "Linear"):
            self.__regressor = Ridge()

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
        y : array-like of shape = [n_samples, ]
            The numerical encoded target for regression tasks.
        **kwargs : default = None
            Additional fitting arguments accepted by model. Not tested.  
            
        Returns
        -------
        object
            self
        """
        y = self.__process_target(y)
            
        if self.__num_bagged_est is None:
            self.__regressor.fit(X, y, **kwargs)
            
        else:
            if not hasattr(self.__regressor, 'random_state'):
                 warnings.warn("The regressor " + str(self.__modelname) + 
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
        array of shape = [n_samples, ] 
            The target to be predicted.
        """

        try:
            if not callable(getattr(self.__regressor, "predict")):
                raise ValueError("predict attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:
            if self.__num_bagged_est is None:
                return self.__regressor.predict(X)
            else:
                bagged_pred = np.zeros(X.shape[0])
                for c, est in enumerate(self.__bagged_est): 
                    bagged_pred += est.predict(X) / self.__num_bagged_est
                    
        else:
            raise ValueError("You must call the fit function before !")
        
        return bagged_pred
 
    def transform(self, X):

        """Transforms X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        array-like or sparse matrix of shape = [n_samples, n_features]
            The transformed X.
        """

        try:
            if not callable(getattr(self.__regressor, "transform")):
                raise ValueError("transform attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            return self.__regressor.transform(X)
        else:
            raise ValueError("You must call the fit function before !")


    def score(self, X, y, sample_weight=None):

        """Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training and cv.
        y : array-like of shape = [n_samples, ]
            The numerical encoded target for regression tasks.

        Returns
        -------
        float
            R^2 of self.predict(df) wrt. y.
        """

        try:
            if not callable(getattr(self.__regressor, "score")):
                raise ValueError("score attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            return self.__regressor.score(X, y, sample_weight)
        else:
            raise ValueError("You must call the fit function before !")
            
    def cross_val_predict(self, X, y, cv=None, scoring=None, **kwargs):
        
        """Performing cross validation hold out predictions for stacking.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training and cv.
        y : array-like of shape = [n_samples, ]
            The numerical encoded target for regression tasks.
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
        array of shape = [n_samples, ]
            The hold out target
        """
        y = self.__process_target(y)
        
        y_pred = np.zeros(X.shape[0]) 
        
        cv = check_cv(cv, y, classifier=False)
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
            y_pred_cv = est.predict(X_cv)
            
#            score = scoring(y_cv, y_pred_proba_cv)                        
            
#            print("Train size: {} ::: cv size: {} score (fold {}/{}): {:.4f}".format(len(train_index), len(cv_index), i + 1, n_splits, score)) 
#            score_mean += score / float(n_splits)
            
            y_pred[cv_index] = y_pred_cv
            
            i += 1 
        
#        print("Mean score: {:.4f}".format(score_mean))    

        return y_pred
        
    def cross_validate(self, X, y, cv=None, scoring=None, **kwargs):
        """Performing a cross validation method.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix used for training.
        y : array-like of shape = [n_samples, ]
            The numerical encoded target for regression tasks.
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
        y = self.__process_target(y)
        
        if self.get_estimator_name == 'LightGBM':
            params = self.__regressor.get_params()
            data = lgb.Dataset(X, label=y)
            cv = check_cv(cv, y, classifier=False)
            ret = lgb.cv(params, data, feval=scoring, folds=cv, **kwargs)
        
        elif self.get_estimator_name == 'XGBoost':
            params = self.__regressor.get_xgb_params()
            data = xgb.DMatrix(X, label=y)
            cv = check_cv(cv, y, classifier=False)
            ret = xgb.cv(params, data, feval=scoring, folds=cv, **kwargs)

        else:  
            ret = cross_validate(self.__regressor, X, y, cv=cv, scoring=scoring)
        
        return ret
    
    def __process_target(self, y):
        
        y = np.array(y, dtype='float') 
               
        return y
    
    def get_estimator(self):

        return self.__classifier 
    
    def get_estimator_copy(self):

        return make_copy(self.__classifier)
    
    @property
    def feature_importances_(self):  
        if self.__fitOK:
            
            if hasattr(self.__regressor, 'feature_importances_'):
                return self.__regressor.feature_importances_
            else:
                raise ValueError('The regressor ' + self.get_estimator_name + 
                                 ' does not have feature_importances_ attribute.')
                
        else:
            
            raise ValueError("You must call the fit function before !")
            
    @property
    def get_estimator_name(self):
        
        return self.__modelname
    