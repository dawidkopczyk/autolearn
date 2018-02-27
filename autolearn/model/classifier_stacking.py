# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

from copy import copy as make_copy

import numpy as np

from sklearn.model_selection import check_cv

from .classifier import Classifier

class ClassifierStacking(Classifier):
    """Stacking classifier"""

    def __init__(self, modelname='Linear',
                 base_estimators=[Classifier(modelname="Linear"),
                                  Classifier(modelname="RandomForest"),
                                  Classifier(modelname="ExtraTrees")],
                 num_bagged_est=None, random_state=None,
                 base_cv=None, base_scoring = None, 
                 base_copy_idx=None, base_save=False, base_save_files=None,
                 base_drop_first=True, stacking_verbose=True, **kwargs):
        
        """Construct a stacking classifier
        
        A stacking classifier is a classifier that uses the predictions of
        base layer estimators (generated with a cross validation method).
        
        Parameters
        ----------
        modelname : str, model name to be used as stacking classifier
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
        base_estimators : list of estimators objects/tuples
            List of estimators to fit in the stacking level using a cross validation. 
            The items of list could be:
            - Classifier/ClassifierStacking objects 
            - tuples with hold out and test predictions of base estimators
        stacking_estimator : object, default = Classifier(modenlame="Linear")
            The estimator used in stacking level.
        base_cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy used in hold out 
            predictions. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a StratifiedKFold,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.
        base_scoring : callable, default: None
            A callable to evaluate the predictions on the cv set in hold out predictions.
            None, accuracy score
        base_copy_idx : list, default = None
            The list of original features added to meta features
        base_save : bool, default = False
            Saves hold out and test predictions of each base estimator to pickle files
        base_save_files : list of tuples, default = None
            File refs of saved hold out and test predictions 
        base_drop_first : bool, default = True
            If True, each base estimator uses n-1 classes without depending class
        random_state:  int, RandomState instance or None, optional, default=None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator; 
            If None, the random number generator is the RandomState instance used by models. 
        stacking_verbose : bool, default = True
            Verbose mode.
        **kwargs : default = None
            Parameters of the corresponding stacking classifier.
            Examples : n_estimators, max_depth, ...
        """
    
        super(ClassifierStacking, self).__init__(modelname, 
             num_bagged_est=num_bagged_est, random_state=random_state, **kwargs)
        
        self.base_estimators = base_estimators
        if type(self.base_estimators) != list:
            raise ValueError("base_estimators must be a list.")
        for i, est in enumerate(self.base_estimators):
            if type(est) == tuple:
                self.base_estimators[i] = est
            elif isinstance(est, Classifier) or isinstance(est, ClassifierStacking):
                self.base_estimators[i] = make_copy(est)
            else:
               raise ValueError("Elements of base_estimators must be either Classifier, ClassifierStacking or tuple.")
              
        self.base_cv = base_cv
        self.base_scoring = base_scoring
        
        self.base_copy_idx = base_copy_idx
        if type(self.base_copy_idx) != list and self.base_copy_idx is not None:
            raise ValueError("base_copy_idx must be either None or a list of integers.")

        self.base_save = base_save
        if type(self.base_save) != bool:
            raise ValueError("base_save must be a boolean.")

        self.base_save_files = base_save_files
        if type(self.base_save_files) != list and self.base_save_files is not None:
            raise ValueError("base_save_files must be either None or a list of tuples.")
            
        if self.base_save_files is not None:
            if len(self.base_save_file) != len(self.base_estimators):
                raise ValueError("base_save_files must be the same size as base_estimators.")
            
        self.base_drop_first = base_drop_first
        if type(self.base_drop_first) != bool:
            raise ValueError("base_drop_first must be a boolean.")

        self.stacking_verbose = stacking_verbose
        if type(self.stacking_verbose) != bool:
            raise ValueError("stacking_verbose must be a boolean.")

        self.__X_meta_test = None
        self.__X_meta_train = None
        self.__fittransformOK = False
        self.__transformOK = False
        
    def get_params(self, deep=True):

        params = {}
        params.update(super(ClassifierStacking, self).get_params())
        params.update({'base_estimators': self.base_estimators,
                       'base_cv': self.base_cv,
                       'base_scoring': self.base_scoring,
                       'base_copy_idx': self.base_copy_idx,
                       'base_save': self.base_save,
                       'base_drop_first': self.base_drop_first,
                       'stacking_verbose': self.stacking_verbose})
    
        return params


    def set_params(self, **params):

        self.__X_meta_test = None
        self.__X_meta_train = None
        self.__fittransformOK = False
        self.__transformOK = False

        if 'base_estimators' in params.keys():
            self.base_estimators = params['base_estimators']
            del params['base_estimators']
            if type(self.base_estimators) != list:
                raise ValueError("base_estimators must be a list.")
            for i, est in enumerate(self.base_estimators):
                if type(est) == tuple:
                    self.base_estimators[i] = est
                elif isinstance(est, Classifier) or isinstance(est, ClassifierStacking):
                    self.base_estimators[i] = make_copy(est)
                else:
                   raise ValueError("Elements of base_estimators must be either Classifier, ClassifierStacking or tuple.")
                   
        if 'base_cv' in params.keys():
            self.base_cv = params['base_cv']
            del params['base_cv']

        if 'base_scoring' in params.keys():
            self.base_scoring = params['base_scoring']
            del params['base_scoring']
            
        if 'base_copy_idx' in params.keys():
            self.base_copy_idx = params['base_copy_idx']
            del params['base_copy_idx']
            if type(self.base_copy_idx) != list and self.base_copy_idx is not None:
                raise ValueError("base_copy_idx must be either None or a list of integers.")
                
        if 'base_save' in params.keys():
            self.base_save = params['base_save']
            del params['base_save']
            if type(self.base_save) != bool:
                raise ValueError("base_save must be a boolean.")   
        
        if 'base_save_files' in params.keys():
            self.base_save_files = params['base_save_files']
            if type(self.base_save_files) != list and self.base_save_files is not None:
                raise ValueError("base_save_files must be either None or a list of tuples.")
            if self.base_save_files is not None:
                if len(self.base_save_file) != len(self.base_estimators):
                    raise ValueError("base_save_files must be the same size as base_estimators.")
            
        if 'base_drop_first' in params.keys():
            self.base_drop_first = params['base_drop_first']
            del params['base_drop_first']
            if type(self.base_drop_first) != bool:
                raise ValueError("base_drop_first must be a boolean.")
                
        if 'stacking_verbose' in params.keys():
            self.stacking_verbose = params['stacking_verbose']
            del params['stacking_verbose']
            if type(self.stacking_verbose) != bool:
                raise ValueError("stacking_verbose must be a boolean.")
                
        super(ClassifierStacking, self).set_params(**params)

    def fit_transform(self, X=None, y=None, **kwargs):

        """Creates training meta-features for the stacking procedure
        and fits the base models.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features], default = None
            Input feature matrix used for training.
        y : array-like of shape = [n_samples, ] or [n_samples, n_classes] for KerasClassifier, default = None
            The numerical encoded target for classification tasks.
        **kwargs : default = None
            Additional fitting arguments accepted by model. Not tested.  
            
        Returns
        -------
        self.__X_meta_train : array-like or sparse matrix of shape = [n_samples, n_base_estimators * (n_classes - int(self.base_drop_first))]
            Training meta-features 
        """

        self.__X_meta_train = None
        
        if X is not None and y is not None:
            cv = check_cv(self.base_cv, y, classifier=True)
            scoring = self.base_scoring

        for c, est in enumerate(self.base_estimators):
            
            if type(est) == tuple:
                if(self.stacking_verbose):
                    print("\n" + "Loading estimator n°" + str(c+1))
                
                y_pred_proba = np.load(est[0])  
  
            elif X is not None and y is not None:
                if(self.stacking_verbose):
                    print("\n" + "Fitting estimator n°" + str(c+1))
                
                y_pred_proba = est.cross_val_predict_proba(X, y, cv=cv, scoring=scoring, **kwargs)
                est.fit(X, y, **kwargs)
                
                if self.base_save:
                    if self.base_save_files is not None:
                        np.save(self.base_save_files[c][0], y_pred_proba)
                    else:
                        np.save('est' + str(c) + '_train', y_pred_proba)
                    
            else:
                raise ValueError("X and y must be specified to fit_transform base estimators.")
                
            for i in range(int(self.base_drop_first), y_pred_proba.shape[1]):
                if self.__X_meta_train is None:
                    self.__X_meta_train = y_pred_proba[:,i]
                else:
                    self.__X_meta_train = np.column_stack((self.__X_meta_train, y_pred_proba[:,i]))

        if self.base_copy_idx is not None:
            self.__X_meta_train = np.column_stack((self.__X_meta_train, X[:,self.base_copy_idx]))
            
        self.__y = y
        self.__fittransformOK = True    
        
        return self.__X_meta_train

    def transform(self, X=None):

        """Creates testing meta-features for the stacking procedure.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features], default = None
            Input feature matrix used for training.
            
        Returns
        -------
        self.__X_meta_test : array-like or sparse matrix of shape = n_samples, n_base_estimators * (n_classes - int(self.base_drop_first))]
            Testing meta-features 
        """

        self.__X_meta_test = None
        
        if not self.__fittransformOK:
            raise ValueError("Call fit_transform before !")

        for c, est in enumerate(self.base_estimators):
            
            if type(est) == tuple:
                if(self.stacking_verbose):
                    print("\n" + "Loading estimator n°" + str(c+1))
                    
                y_pred_proba = np.load(est[1])   
  
            elif X is not None:
                if(self.stacking_verbose):
                    print("\n" + "Predicting estimator n°" + str(c+1))
    
                y_pred_proba = est.predict_proba(X)
                
                if self.base_save:
                    if self.base_save_files is not None:
                        np.save(self.base_save_files[c][1], y_pred_proba)
                    else:
                        np.save('est' + str(c) + '_test', y_pred_proba)
            
            else:
                raise ValueError("X must be specified to transform.")
                
            for i in range(int(self.base_drop_first), y_pred_proba.shape[1]):
                if self.__X_meta_test is None:
                    self.__X_meta_test = y_pred_proba[:,i]
                else:
                    self.__X_meta_test = np.column_stack((self.__X_meta_test, y_pred_proba[:,i]))

        if self.base_copy_idx is not None:
            self.__X_meta_test = np.column_stack((self.__X_meta_test, X[:,self.base_copy_idx]))
        
        self.__transformOK = True
        
        return self.__X_meta_test
    
    def fit(self, X=None, y=None, **kwargs):
        if X is not None and y is not None:
            X = self.fit_transform(X, y)
        else:
            if y is None:
                y = self.get_y
            X = self.get_meta_train  
        
        if(self.stacking_verbose):
            print("Fitting stacking estimator...")
        return super(ClassifierStacking, self).fit(X, y, **kwargs)
 
    def predict(self, X=None):
        if X is not None:
            X = self.transform(X)
        else:
            X = self.get_meta_test
            
        return super(ClassifierStacking, self).predict(X)

    def predict_log_proba(self, X=None):
        if X is not None:
            X = self.transform(X)
        else:
            X = self.get_meta_test
        
        return super(ClassifierStacking, self).predict_log_proba(X)

    def predict_proba(self, X=None):
        if X is not None:
            X = self.transform(X)
        else:
            X = self.get_meta_test
        
        return super(ClassifierStacking, self).predict_proba(X)
    
    def cross_val_predict_proba(self, X=None, y=None, cv=None, scoring=None, **kwargs):  
        if X is not None and y is not None:
            X = self.fit_transform(X, y)
        else:
            if y is None:
                y = self.get_y
            X = self.get_meta_train  
        
        return super(ClassifierStacking, self).cross_val_predict_proba(X, y, cv=cv, scoring=scoring, **kwargs)
    
    def cross_validate(self, X=None, y=None, cv=None, scoring=None, **kwargs):
        if X is not None and y is not None:
            X = self.fit_transform(X, y)
        else:
            if y is None:
                y = self.get_y
            X = self.get_meta_train  
        
        return super(ClassifierStacking, self).cross_validate(X, y, cv=cv, scoring=scoring, **kwargs)
    
    @property
    def get_meta_train(self):
        """Get training meta-features."""
        if not self.__fittransformOK:
            raise ValueError('No training meta-features found. Need to call fit_transform beforehand.')
        return self.__X_meta_train

    @property
    def get_y(self):
        """Get training meta-features."""
        if not self.__fittransformOK:
            raise ValueError('No y found. Need to call fit_transform beforehand.')
        return self.__y
    
    @property
    def get_meta_test(self):
        """Get training meta-features."""
        if not self.__transformOK:
            raise ValueError('No test meta-features found. Need to call transform beforehand.')
        return self.__X_meta_test