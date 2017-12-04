# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import OneHotEncoder

class EncoderCategorical():

    """Categorical features encoder."""

    def __init__(self, strategy=None, verbose=False):
        """Encodes categorical features
    
        Several strategies are possible (supervised or not). Works for both
        classification and regression tasks.
    
        Parameters
        ----------
        strategy : dict, optional (default=None)
            Strategy dictory for categorical encoding.
            Available keys :
                - "one_hot"             -> One hot encoding
                - "target_encoding"     -> Target encoding (only for classification)
            Values :
                - "all"             -> transform all columns
                - lst               -> list with numeric indices of columns to be encoded
    
        verbose : bool, default = False
            Verbose mode. Useful for entity embedding strategy.
        """
    
        self.strategy = strategy
        self.verbose = verbose
        self.__enc = dict()
        self.__fitOK = False


    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'verbose': self.verbose}


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder "
                              "Categorical_encoder. Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)


    def fit(self, X, y=None, **kwargs):

        """Fits Categorical Encoder.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
        y : array-like, shape = [n_samples], optional (default=None)
            The target values
        **kwargs : default = None
            Parameters of the corresponding encoder.
            Examples : min_samples_leaf, smoothing, ...

        Returns
        -------
        object
            self
        """

        if self.strategy is None:
            pass
        
        for k, v in self.strategy.items():
            
            if len(v) == 0:
                pass
            
            else:
                if X.ndim == 1:
                    X = np.expand_dims(X, axis=1)
                    
                if v == "all":
                    v = range(X.shape[1])
                
                v = np.array(v).astype(np.int64)
                
                #################################################
                #                One Hot 
                #################################################
                
                if k == 'one_hot':

                    self.__enc['one_hot'] = OneHotEncoder(categorical_features=v, sparse=False)
                    self.__enc['one_hot'].fit(X)
 
                #################################################
                #                Target Encoding
                #################################################
                   
                elif k == 'target_encoding':
                    
                    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
                    smoothing = kwargs.get('smoothing', 1)
                    noise_level = kwargs.get('noise_level', 0)
                    
                    target = pd.Series(y, name='target') 
                    self.__prior = target.mean()
                    
                    for col in v:
                      
                        train = pd.Series(X[:,col], name='train')
                        temp = pd.concat((train, target), axis=1)
                    
                        averages = temp.groupby(by='train')['target'].agg(['mean', 'count'])
                        
                        # Compute smoothing
                        smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
                    
                        # The bigger the count the less full_avg is taken into account
                        averages['target_avg'] = self.__prior * (1 - smoothing) + (averages["mean"] *
                                                                           smoothing)
                        averages.drop(['mean', 'count'], axis=1, inplace=True)  
                        
                        self.__enc[col] = averages
                        self.__noise_level = noise_level  
                        
                else:
    
                    raise ValueError("Strategy for categorical encoding is not valid")
                    
        self.__fitOK = True
        
        return self


    def fit_transform(self, X, y=None):

        """Fits Categorical Encoder and transforms the dataset

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
        y : array-like, shape = [n_samples], optional (default=None)
            The target values

        Returns
        -------
        array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix with encoded categorical features
        """

        self.fit(X, y)

        return self.transform(X)


    def transform(self, X):

        """Transforms the dataset

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.

        Returns
        -------
        array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix with encoded categorical features
        """

        if self.__fitOK:

            if self.strategy is None:
                pass
        
            for k, v in self.strategy.items():
                
                if len(v) == 0:
                    pass
                
                else:
                    if X.ndim == 1:
                        X = np.expand_dims(X, axis=1)
                    
                    if v == "all":
                        v = range(X.shape[1])
                    
                    v = np.array(v).astype(np.int64)
                
                    #################################################
                    #                One Hot
                    #################################################
                    
                    if k == 'one_hot':

                        return self.__enc['one_hot'].transform(X)

                    #################################################
                    #                Target Encoding
                    #################################################
                       
                    elif k == 'target_encoding':
                    
                        def get_encoded(col):
                            train = pd.Series(X[:,col], name='train')
                            
                            encoded = pd.merge(train.to_frame(train.name),
                                self.__enc[col].reset_index().rename(columns={'index': 'target',
                                                                       'target_avg': 'average'}),
                                on=train.name,
                                how='left')['average'].fillna(self.__prior)
                        
                            return encoded * (1 + self.__noise_level * np.random.randn(len(encoded)))
                            
                        if v.size == X.shape[1]:
                            return np.column_stack(tuple([get_encoded(col) for col in v]))
                        else:
                            return np.column_stack(tuple([get_encoded(col) for col in v]) + (X[:,~v], ))
                        
            else:
    
                raise ValueError("Call fit or fit_transform function before")
