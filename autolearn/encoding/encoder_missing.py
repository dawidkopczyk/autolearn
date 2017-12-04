# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import numpy as np
import warnings

from sklearn.preprocessing import Imputer

class EncoderMissing():

    """Missing values Encoder."""

    def __init__(self, strategy=None, missing_values='NaN'):
        """Encodes missing values.
    
        Several strategies are available.
    
        Parameters
        ----------
        strategy : dict, optional (default=None)
            Strategy dictory for missing value replacement.
            Available keys :
                - "mean"            -> Missing value is filled with mean of column
                - "median"          -> Missing value is filled with median of column
                - "most_frequent"   -> Missing value is filled with most frequent value of column
                - value             -> Missing value is filled with a value
            Values :
                - "all"             -> transform all columns
                - lst               -> list with numeric indices of columns to be encoded
        missing_values : integer or "NaN", optional (default="NaN")
            The placeholder for the missing values. All occurrences of
            `missing_values` will be imputed. For missing values encoded as np.nan,
            use the string value "NaN".
        """
     
        self.strategy = strategy
        self.missing_values = missing_values
        self.__imp = {}
        self.__fitOK = False

    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'missing_values': self.missing_values}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder EncoderMissing. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`EncoderMissing.get_params().keys()`")
            else:
                setattr(self, k, v)

    def fit(self, X):

        """Fits missing values encoder.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
            
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
                if v == "all":
                    v = range(X.shape[1])
                
                v = np.array(v).astype(np.int64)
                    
                if k in ['mean', 'median', 'most_frequent']:
                    enc = Imputer(missing_values=self.missing_values, strategy=k)
                    if v.size == 1:
                        self.__imp[k] = enc.fit(X[:,v].reshape(-1,1))
                    else:
                        self.__imp[k] = enc.fit(X[:,v])
                    
                else:
                    pass
                
        self.__fitOK = True

        return self

    def fit_transform(self, X):

        """Fits missing values encoder and transforms the X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.

        Returns
        -------
        array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix with no missing values.
        """

        self.fit(X)

        return self.transform(X)

    def transform(self, X):

        """Transforms the X

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.

        Returns
        -------
        array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix with no missing values.
        """

        if self.strategy is None:
            return X
        
        if self.__fitOK:

            X_trans = X
            
            for k, v in self.strategy.items():
                
                if len(v) == 0:
                    pass
                
                else:
                    if v == "all":
                        v = range(X.shape[1])
                    
                    v = np.array(v).astype(np.int64)
                    
                    if k in ['mean', 'median', 'most_frequent']:
                        
                        if v.size == 1:
                            X_trans[:,v] = self.__imp[k].transform(X[:,v].reshape(-1,1)).ravel()
                        else:
                            X_trans[:,v] = self.__imp[k].transform(X[:,v])
                            
                    else:
                        
                        if self.missing_values == 'NaN' or np.isnan(self.missing_values):
                            inds = np.where(np.isnan(X[:,v]))
                        else:
                            inds = np.where(X[:,v] == self.missing_values)
                        if v.size == 1:
                            X_trans[inds,v] = k
                        else:
                            X_trans[inds] = k
             
            return X_trans
        
        else:

            raise ValueError("Call fit function before")
