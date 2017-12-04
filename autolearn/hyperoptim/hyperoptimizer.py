# coding: utf-8
# author: dawidkopczyk <dawid.kopczyk@gmail.com>
# License: BSD 3 clause

import numpy as np
import warnings
import time

from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score, check_cv
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

#from ..encoding.encoder_missing import EncoderMissing
#from ..encoding.encoder_categorical import EncoderCategorical
#from ..encoding.feature_selector import FeatureSelector
#from ..model.classifier_stacking import ClassifierStacking
#from ..model.classifier import Classifier
#from ..model.regressor_stacking import RegressorStacking
#from ..model.regressor import Regressor

class HyperOptimiser():

    """Optimises hyper-parameters of the whole Pipeline.
    
    - Encoder missing (missing values encoder, OPTIONAL)
    - Encoder categorical (categorical features encoder,  OPTIONAL)
    - Feature selector (OPTIONAL)
    - Stacking estimator - feature engineer (OPTIONAL)
    - other valid transformers
    - Estimator (Classifier or Regressor)
    Works for both regression and classification (multiclass or binary) tasks.
    
    Parameters
    ----------
    transformers : dict, default = None
        Dictonary of other transformers than built-in:
        - "em" = EncoderMissing
        - "ec" = EncoderCategorical
        - "fs" = FeatureSelector
        Transformer should containt fit() and transform() methods
        to be properly used. Key is the name of tfr used in a pipeline.
    scoring : str, callable or None. default: None
        A string or a scorer callable object.
        If None, "log_loss" is used for classification and
        "mean_squared_error" for regression
        Available scorings for classification : {"accuracy","roc_auc", "f1",
        "log_loss", "precision", "recall"}
        Available scorings for regression : {"mean_absolute_error",
        "mean_squared_error","median_absolute_error","r2"}
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a StratifiedKFold,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
    random_state : int, default = 1
        Pseudo-random number generator state used for shuffling
    to_path : str, default = "save"
        Name of the folder where models are saved
    verbose : bool, default = True
        Verbose mode
    """

    def __init__(self, tranformers=None,
                 scoring=None,
                 cv=None,
                 random_state=1,
                 to_path="save",
                 verbose=True):

        self.tranformers = tranformers
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.to_path = to_path
        self.verbose = verbose

        warnings.warn("HyperOptimiser will save all your fitted models into directory '"
                      +str(self.to_path)+"/joblib'. Please clear it regularly.")        
        
    def get_params(self, deep=True):

        return {'tranformers': self.tranformers,
                'scoring': self.scoring,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'to_path': self.to_path,
                'verbose': self.verbose}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for optimiser HyperOptimiser. "
                              "Parameter IGNORED. Check the list of available "
                              "parameters with `optimiser.get_params().keys()`")
            else:
                setattr(self, k, v)


    def evaluate(self, params, X, y):

        """Evaluates the data.
        Evaluates the data with a given scoring function and given hyper-parameters
        of the whole pipeline. 
        
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
            
        Returns
        -------
        float.
            The score. The higher the better.
            Positive for a score and negative for a loss.
            
        Examples
        --------
        >>> from quanteelearn.hyperoptim import *
        >>> from sklearn.datasets import load_boston
        >>> #load data
        >>> dataset = load_boston()
        >>> #evaluating the pipeline
        >>> hopt = HyperOptimiser()
        >>> params = {
        ...     "ne__strategy" : 0,
        ...     "ce__strategy" : "label_encoding",
        ...     "fs__threshold" : 0.1,
        ...     "stck__base_estimators" : [Regressor(strategy="RandomForest"), Regressor(strategy="ExtraTrees")],
        ...     "est__strategy" : "Linear"
        ... }
        >>> df = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> opt.evaluate(params, df)
        """

        ##########################################
        #             Classification
        ##########################################

        if y.dtype == 'int':

            # Cross validation
            cv = check_cv(self.cv, y, classifier=True)
            n_splits = cv.get_n_splits()
            
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

            # Default scoring for classification
            if self.scoring is None:
                self.scoring = 'neg_log_loss'

            else:
                if type(self.scoring) == str:
                    if self.scoring in ["accuracy", "roc_auc", "f1",
                                         "neg_log_loss", "precision", "recall"]:
                        pass
                    else:
                        warnings.warn("Invalid scoring metric. "
                                      "neg_log_loss is used instead.")
                        self.scoring = 'neg_log_loss'

                else:
                    pass

        ##########################################
        #               Regression
        ##########################################

        elif y.dtype == 'float':

            # Cross validation
            cv = check_cv(self.cv, y, classifier=False)
            n_splits = cv.get_n_splits()

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

            # Default scoring for regression
            if (self.scoring is None):
                self.scoring = "mean_squared_error"
            else:
                if type(self.scoring) == str:
                    if self.scoring in ["mean_absolute_error",
                                         "mean_squared_error",
                                         "median_absolute_error",
                                         "r2"]:
                        pass
                    else:
                        warnings.warn("Invalid scoring metric. "
                                      "mean_squarred_error is used instead.")
                        self.scoring = 'mean_squared_error'
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
                print("#####################################################"
                      " testing hyper-parameters... "
                      "#####################################################")
                print("")
#                print(">>> NA ENCODER :" + str(ne.get_params()))
#                print("")
#                print(">>> CA ENCODER :" + str({'strategy': ce.strategy}))

                for i, stck in enumerate(np.sort(list(STCK))):

                    stck_params = STCK[stck].get_params().copy()

                    print("")
                    print(">>> STACKING LAYER n°"
                          + str(i + 1) + " :" + str(STCK[stck].get_estimator_name))

                    for j, model in enumerate(stck_params["base_estimators"]):
                        print("")
                        print("    > base_estimator n°" + str(j + 1) + " :"
                              + str(model.get_estimator_name))

                print("")
                print(">>> ESTIMATOR :" + str(est.get_estimator_name))
                print("")

            try:

                # Computing the mean cross validation score across the folds
                scores = cross_val_score(estimator=pp,
                                         X=X,
                                         y=y,
                                         scoring=self.scoring,
                                         cv=cv)
                score = np.mean(scores)

            except:

                scores = [-np.inf for _ in range(n_splits)]
                score = -np.inf

        else:
            raise ValueError("Pipeline cannot be set with these parameters."
                             " Check the name of your stages.")

        if score == -np.inf:
            warnings.warn("An error occurred while computing the cross "
                          "validation mean score. Check the parameter values "
                          "and your scoring function.")

        ##########################################
        #             Reporting scores
        ##########################################

        out = " ("

        for i, s in enumerate(scores[:-1]):
            out = out + "fold " + str(i + 1) + " = " + str(s) + ", "

        if self.verbose:
            print("")
            print("MEAN SCORE : " + str(self.scoring) + " = " + str(score))
            print("VARIANCE : " + str(np.std(scores))
                  + out + "fold " + str(i + 2) + " = " + str(scores[-1]) + ")")
            print("Time: %s seconds" % (time.time() - start_time))
            print("")

        return score


    def optimise(self, space, X, y, max_evals=40):

        """Optimises the Pipeline.
        Optimises hyper-parameters of the whole Pipeline with a given scoring
        function. Algorithm used to optimize : Tree Parzen Estimator.
        IMPORTANT : Try to avoid dependent parameters and to set one feature
        selection strategy and one estimator strategy at a time.
        Parameters
        ----------
        space : dict, default = None.
            Hyper-parameters space:
            - The keys must respect the following syntax : "tfr__param".
                - "tfr" = "em" for encoder missing [OPTIONAL]
                - "tfr" = "ec" for encoder categorical [OPTIONAL]
                - "tfr" = "fs" for feature selector [OPTIONAL]
                - "tfr" = key in transformer argument directory for other transformers [OPTIONAL]
                - "tfr" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
                - "tfr" = "est" for the final estimator
                - "param" : a correct associated parameter for each step. Ex: "max_depth" for "tfr"="est", ...
            - The values must respect the syntax: {"search":strategy,"space":list}
                - "strategy" = "choice" or "uniform". Default = "choice"
                - list : a list of values to be tested if strategy="choice". Else, list = [value_min, value_max].
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (with dtype='float' for a regression or dtype='int' for a classification).
        max_evals : int, default = 40.
            Number of iterations.
            For an accurate optimal hyper-parameter, max_evals = 40.
        Returns
        -------
        dict.
            The optimal hyper-parameter dictionary.
        Examples
        --------
        >>> from mlbox.optimisation import *
        >>> from sklearn.datasets import load_boston
        >>> #loading data
        >>> dataset = load_boston()
        >>> #optimising the pipeline
        >>> opt = Optimiser()
        >>> space = {
        ...     'fs__strategy':{"search":"choice","space":["variance","rf_feature_importance"]},
        ...     'est__colsample_bytree':{"search":"uniform", "space":[0.3,0.7]}
        ... }
        >>> df = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> best = opt.optimise(space, df, 3)
        """

        hyperopt_objective = lambda params: -self.evaluate(params, X, y)

        # Creating a correct space for hyperopt

        if space is None:
            warnings.warn(
                "Space is empty. Please define a search space. "
                "Otherwise, call the method 'evaluate' for custom settings")
            return dict()

        else:

            if len(space) == 0:
                warnings.warn(
                    "Space is empty. Please define a search space. "
                    "Otherwise, call the method 'evaluate' for custom settings")
                return dict()

            else:

                hyper_space = {}

                for p in space.keys():

                    if "space" not in space[p]:
                        raise ValueError("You must give a space list ie values"
                                         " for hyper parameter " + p + ".")

                    else:

                        if "search" in space[p]:

                            if space[p]["search"] == "uniform":
                                hyper_space[p] = hp.uniform(p, np.sort(space[p]["space"])[0],
                                                            np.sort(space[p]["space"])[-1])

                            elif space[p]["search"] == "choice":
                                hyper_space[p] = hp.choice(p, space[p]["space"])
                            else:
                                raise ValueError(
                                    "Invalid search strategy "
                                    "for hyper parameter " + p + ". Please"
                                    " choose between 'choice' and 'uniform'.")

                        else:
                            hyper_space[p] = hp.choice(p, space[p]["space"])

                best_params = fmin(hyperopt_objective,
                                   space=hyper_space,
                                   algo=tpe.suggest,
                                   max_evals=max_evals)

                # Displaying best_params

                for p, v in best_params.items():
                    if ("search" in space[p]):
                        if (space[p]["search"] == "choice"):
                            best_params[p] = space[p]["space"][v]
                        else:
                            pass
                    else:
                        best_params[p] = space[p]["space"][v]

                if (self.verbose):
                    print("")
                    print("")
                    print("~" * 137)
                    print("~" * 57 + " BEST HYPER-PARAMETERS " + "~" * 57)
                    print("~" * 137)
                    print("")
                    print(best_params)

                return best_params
