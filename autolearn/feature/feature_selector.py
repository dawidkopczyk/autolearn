# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np

try:
    from boruta import BorutaPy
    _IS_BORUTA_INSTALLED = True
except ImportError:  
    _IS_BORUTA_INSTALLED = False

#from ..model.regressor import Regressor
#from ..model.classifier import Classifier

class FeatureSelector():

    """Feature selector using Boruta."""

    def __init__(self, classifier=True,
                 estimator=None, **kwargs):
        """Selects important features using Boruta.
    
        Parameters
        ----------
        classifier: bool, default=True
            Flag indicating classification or regression task.
        estimator : object, default=None
            A Classifier or Regressor, with a 'fit' method that returns the
            feature_importances_ attribute. Important features must correspond to
            high absolute values in the feature_importances_.
            If None for classifier=True, then RandomForestClassifier is selected,
            if None for classifier=False, then RandomForestRegressor is selected.
        **kwargs : : default = None
            Parameters of Boruta.
            
        Attributes
        ----------
        n_features_ : int
            The number of selected features.
        support_ : array of shape [n_features]
            The mask of selected features - only confirmed ones are True.
        support_weak_ : array of shape [n_features]
            The mask of selected tentative features, which haven't gained enough
            support during the max_iter number of iterations..
        ranking_ : array of shape [n_features]
            The feature ranking, such that ``ranking_[i]`` corresponds to the
            ranking position of the i-th feature. Selected (i.e., estimated
            best) features are assigned rank 1 and tentative features are assigned
            rank 2.
        """
        
        if not _IS_BORUTA_INSTALLED:
            raise ValueError('Boruta is required for this module')
            
        self.classifier = classifier
        if type(self.classifier) != bool:
            raise ValueError('classifier flag must a boolean')
            
        self.estimator = estimator
        if self.estimator is None and self.classifier:
            self.estimator = Classifier(modelname='RandomForest')
        elif self.estimator is None:
            self.estimator = Regressor(modelname='RandomForest')
        
        if not isinstance(self.estimator, Classifier) and self.classifier:
            raise ValueError('Classifier is required for classifier=True')
        elif not isinstance(self.estimator, Regressor) and not self.classifier:
            raise ValueError('Regressor is required for classifier=False')
            
        self.__selector = BorutaPy(self.estimator.get_estimator(), **kwargs)
        self.__fitOK = False

    def get_params(self, deep=True):

        return {'classifier': self.classifier,
                'estimator': self.estimator,
                'n_estimators': self.__selector.n_estimators, 
                'perc': self.__selector.perc, 
                'alpha': self.__selector.alpha,
                'two_step': self.__selector.two_step, 
                'max_iter': self.__selector.max_iter, 
                'random_state': self.__selector.random_state, 
                'verbose': self.__selector.verbose}


    def set_params(self, **params):

        self.__fitOK = False
        
        if 'classifier' in params.keys():
            self.classifier = params['classifier']
            del params['classifier']
            if type(self.classifier) != bool:
                raise ValueError('classifier flag must a boolean')
        
        if 'estimator' in params.keys():
            self.estimator = params['estimator']
            del params['estimator']           
            
            if self.estimator is None and self.classifier:
                self.estimator = Classifier(modelname='RandomForest')
            elif self.estimator is None:
                self.estimator = Regressor(modelname='RandomForest')
            
            if not isinstance(self.estimator, Classifier) and self.classifier:
                raise ValueError('Classifier is required for classifier=True')
            elif not isinstance(self.estimator, Regressor) and not self.classifier:
                raise ValueError('Regressor is required for classifier=False')
            
            self.__selector = BorutaPy(self.estimator.get_estimator())
            
        for k, v in params.items():
            if k not in ['n_estimators', 'perc', 'alpha',
                         'two_step', 'max_iter', 'random_state', 'verbose']:
                warnings.warn("Invalid parameter a for feature selector"
                              ". Parameter IGNORED. Check"
                              "the list of available parameters with"
                              "`feature_selector.get_params().keys()`")
            else:
                setattr(self.__selector, k, v)
                
    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        
        Returns
        -------
        object
            self
        """
        err = np.geterr()['invalid']
        np.seterr(invalid='ignore')
        
        self.__selector.fit(X, y)
        self.__fitOK = True
        
        np.seterr(invalid=err)
        
        return self
     
    def transform(self, X, weak=False):
        """
        Reduces the input X to the features selected by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """
        if self.__fitOK:
            
            return self.__selector.transform(X, weak)    
        
        else:

            raise ValueError("Call fit function before")
        
         
    def fit_transform(self, X, y, weak=False):
        """
        Fits Boruta, then reduces the input X to the selected features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self.fit(X, y)
        return self.transform(X, weak)     
    
    @property
    def n_features_(self):
        if self.__fitOK: 
            return self.__selector.n_features_
        else:
            raise ValueError("Call fit function before")
            
    @property
    def support_(self):
        if self.__fitOK: 
            return self.__selector.support_ 
        else:
            raise ValueError("Call fit function before")
            
    @property
    def support_weak_(self):
        if self.__fitOK: 
            return self.__selector.support_weak_ 
        else:
            raise ValueError("Call fit function before")
            
    @property
    def ranking_(self):
        if self.__fitOK: 
            return self.__selector.ranking_ 
        else:
            raise ValueError("Call fit function before")