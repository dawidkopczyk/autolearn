# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import numpy as np
import os
import warnings
import time

from sklearn.pipeline import Pipeline

#from ..encoding.encoder_missing import EncoderMissing
#from ..encoding.encoder_categorical import EncoderCategorical
#from ..encoding.feature_selector import FeatureSelector
#from ..model.classifier_stacking import ClassifierStacking
#from ..model.classifier import Classifier
#from ..model.regressor_stacking import RegressorStacking
#from ..model.regressor import Regressor

class Predictor():

    """Fits and predicts the target on the test dataset.

    The test dataset must not contain the target values.

    Parameters
    ----------
    transformers : dict, default = None
        Dictonary of other transformers than built-in:
        - "em" = EncoderMissing
        - "ec" = EncoderCategorical
        - "fs" = FeatureSelector
        Transformer should containt fit() and transform() methods
        to be properly used. Key is the name of tfr used in a pipeline.
    to_path : str, default = "save"
        Name of the folder where feature importances and
        predictions are saved (.png and .csv formats).
        Must contain target encoder object (for classification task only).
    verbose : bool, default = True
        Verbose mode
    """

    def __init__(self, tranformers=None, to_path="save", verbose=True):

        self.tranformers = tranformers
        self.to_path = to_path
        self.verbose = verbose

    def get_params(self, deep=True):

        return {'to_path': self.to_path,
                'verbose': self.verbose
                }

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for predictor Predictor. "
                              "Parameter IGNORED. "
                              "Check the list of available parameters with "
                              "`predictor.get_params().keys()`")
            else:
                setattr(self,k,v)

    def fit_predict(self, params, X, y, X_test):


        """Fits the model and predicts on the test set.

        Also outputs the submission file
        
        Parameters
        ----------
        params : dict
            Hyper-parameters dictionary for the whole pipeline.
            - The keys must respect the following syntax : "tfr__param".
                - "tfr" = "em" for encoder missing [OPTIONAL]
                - "tfr" = "ec" for encoder categorical [OPTIONAL]
                - "tfr" = "fs" for feature selector [OPTIONAL]
                - "tfr" = key in transformer argument directory for other transformers [OPTIONAL]
                - "tfr" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
                - "tfr" = "est" for the final estimator
                - "param" : a correct associated parameter for each step. Ex: "max_depth" for "tfr"="est", ...
            - The values are those of the parameters. Ex: 4 for key = "est__max_depth", ...
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (with dtype='float' for a regression or dtype='int' for a classification).
        X_test : array-like, shape = [n_samples, n_features]
            The testing input samples.
        Returns
        -------
        object
            self.
        """

        if self.to_path is None:
            raise ValueError("You must specify a path to save your model "
                             "and your predictions")

        else:

            ##########################################
            #             Classification
            ##########################################

            if y.dtype == 'int':

                # Estimator
                est = Classifier()
    
                # Feature selection if specified
                fs = None
                for p in params.keys():
                    if p.startswith("fs__"):
                        fs = FeatureSelector(classifier=True)
                    else:
                        pass
                        
                # Stacking if specified
                STCK = {}
                for p in params.keys():
                    if p.startswith("stck"):
                        STCK[p.split("__")[0]] = ClassifierStacking(stacking_verbose=False)
                    else:
                        pass

            ##########################################
            #               Regression
            ##########################################

            elif y.dtype == 'float':
    
                # Estimator
                est = Regressor()
    
                fs = None
                for p in params.keys():
                    if p.startswith("fs__"):
                        fs = FeatureSelector(classifier=False)
                    else:
                        pass
                    
                # Stacking if specified
                STCK = {}
                for p in params.keys():
                    if p.startswith("stck"):
                        STCK[p.split("__")[0]] = RegressorStacking(stacking_verbose=False)
                    else:
                        pass

            else:
                raise ValueError("Impossible to determine the task. "
                                 "Please check that your target is encoded.")

            ##########################################
            #          Creating the Pipeline
            ##########################################
            pipe = []
            
            # Encoder missing if specified
            for p in params.keys():
                if p.startswith("em__"):
                    pipe.append(("em", EncoderMissing()))   
    
            # Encoder categorical if specified
            for p in params.keys():
                if p.startswith("ec__"):
                    pipe.append(("ec", EncoderCategorical()))   
    
            # Feature selector
            if fs is not None:
                pipe.append(("fs", fs))
    
            # Other transformers
            if self.tranformers is not None:
                for k, v in self.tranformers.items():
                    pipe.append((k, v))
            
            # Stacking estimator
            for stck in np.sort(list(STCK)):
                pipe.append((stck, STCK[stck]))
    
            # Estimator
            pipe.append(("est", est))
    
            # Do we need to cache transformers?
            cache = False
    
            if "ce__strategy" in params:
                if params["ce__strategy"] == "entity_embedding":
                    cache = True
                else:
                    pass
            else:
                pass
            
            if len(STCK) != 0:
                cache = True
            else:
                pass
            
            if cache:
                pp = Pipeline(pipe, memory=self.to_path)
            else:
                pp = Pipeline(pipe)
            
            ##########################################
            #          Fitting the Pipeline
            ##########################################

            start_time = time.time()
    
            # No params : default configuration
            if params is None:
                set_params = True
                print('No parameters set. Default configuration is tested')
    
            else:
                try:
                    pp = pp.set_params(**params)
                    set_params = True
                except:
                    set_params = False
    
            if set_params:
    
                if self.verbose:
                    print("")
                    print("fitting the pipeline ...")
                    
                try:
                    pp.fit(X, y)
                except:
                    raise ValueError("Pipeline cannot be fitted")

                if self.verbose:
                    print("Time: %s seconds"%(time.time() - start_time))

                try:
                    os.mkdir(self.to_path)
                except OSError:
                    pass
    
            else:
                raise ValueError("Pipeline cannot be set with these parameters."
                                 " Check the name of your stages.")

            ##########################################
            #               Predicting
            ##########################################

            if X_test.shape[0] == 0:
                warnings.warn("You have no test dataset. Cannot predict !")

            else:

                start_time = time.time()

                ##########################################
                #             Classification
                ##########################################

                if y.dtype == 'int':

                    if self.verbose:
                        print("")
                        print("predicting ...")
                            
                    try:
                        
                        pred_proba = pp.predict_proba(X_test)
                        pred = np.argmax(pred_proba, axis=1)
                        
                    except:
                        raise ValueError("Can not predict")

                ##########################################
                #               Regression
                ##########################################

                elif y.dtype == 'float':

                    if self.verbose:
                        print("")
                        print("predicting...")

                    try:

                        pred= pp.predict(X_test)

                    except:
                        raise ValueError("Can not predict")

                else:
                    pass

                if self.verbose:
                    print("Time: %s seconds" % (time.time() - start_time))

                ##########################################
                #           Dumping predictions
                ##########################################

                if self.verbose:
                    print("")
                    print("dumping predictions into directory : " + self.to_path + " ...")

                np.savetxt(self.to_path + "/" + "_pred.csv", pred, delimiter=',')

        return pred
