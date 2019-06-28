import numpy as np
from scipy.stats import mode
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

# Classification metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

# Regresion metrics
from sklearn.metrics import explained_variance_score, median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

import time

class BaseAverageRF():
    """
    Abstract class for an averaged Random Forest [classifier / regressor]. 
    
    Builds n_forests RFs, then predicions are the average (or mode) of the obtained for each RF.

    It is also possile to obtain an array with the individual predictions for each forest,
    by adding a underscore before the name of the function (e.g. "_predict")
    
    """
    def __init__(self,  n_forests, verbosity, *kargs, **kwarg):

        # Variables 
        self.n_forests = n_forests
        self.verbosity = verbosity

        self.models = []
        self.stats = {}
        self.skf = None
        self.RFModel = None

        for _ in range(self.n_forests):
            self.models.append(self.RFModel(*kargs, **kwargs))

    def fit(self,X,y):
        """
        Train the RF models usinig stratified K-fold splits, 
        with K = n_forests
        """
        train_times= []
        scores = []

        #Splits the data into stratified k-folds
        skf = self.skf
        folds = skf.split(X,y)
        
        # Train and test the models
        for fold in enumerate(folds):
            this_rf = fold[0]
            train, test = fold[1]
            _model = self.models[this_rf]
            _start = time.time()
            _model.fit(X[train],y[train])
            _elapsed = time.time() - _start

            # Calculate stats for each RF
            _y_pred = _model.predict(X[test])
            _y_true = y[test]

            train_times.append(_elapsed)

            _model.calc_stats()

            if self.verbosity > 0:
                print(f"SPLIT: {this_rf}  [TRAIN: {len(train)} \tTEST: {len(test)}] \t Training Time:{_elapsed:.3f} seconds.")
                _model.print_stats()

        return self
    
    def score(self, X, y, sample_weight=None):
        return np.average(self._score(X, y, sample_weight), axis=0)
    
    def calc_stats(self):
        pass
    
    def print_stats(self):
        pass

    ###############################################################################################
    #                                         LOCAL FUNCTIONS                                     #
    ###############################################################################################

    def _evaluate_models(self, func_name, *args, **kwargs):
        """
        Evaluates the required property for each RandomForest
        This is only a wrapper around the RF calls, to evaluate
        the desired function to each RF and return a list
        of the results for each RF.
        """
        results = []
        for model in self.models:
            func = getattr(model, func_name)
            result = func(*args,**kwargs)
            results.append(result)
        return results
    
    # The following functions available for each RF, and return
    # arrays with the results for each RF.
    def _predict(self, X):
        func = "predict"
        return self._evaluate_models(func, X)

    def _score(self,X, y, sample_weight=None):
        func = "score"
        return self._evaluate_models(func, X, y, sample_weight)

    def _set_params(self,**params):
        for model in self.models:
            model.set_params(**params)

    def _get_params(self, deep=True):
        params = []
        for model in self.models:
            params.append(model.get_params(deep))
        return params


class AverageRFRegressor(BaseAverageRF):

    def __init__(self, n_forests=5,verbosity=0, *kargs, **kwargs):
        self.RFModel = RandomForestRegressor
        super().__init__(n_forests,verbosity, *kargs, **kwargs)

    exp_var_sc, mean_abs_err, mean_sq_err, median_abs_err = [], [], [], []
    _stats = {"Training Time (s)":train_times,
              "Explained Variance":exp_var_sc,
              "Mean Absolute Error": mean_abs_err,
              "Mean Squared Error": mean_sq_err,
              "Median Absolute Error": median_abs_err,
              "R^2":scores} 
    skf = KFold(n_splits=self.n_forests, random_state=None, shuffle=True)

    def predict(self, X):
        return np.average(self._predict(X), axis=0)
    
    def calc_stats():
        exp_var_sc.append(explained_variance_score(_y_true, _y_pred))
        mean_abs_err.append(mean_absolute_error(_y_true,_y_pred))
        mean_sq_err.append(mean_squared_error(_y_true,_y_pred))
        median_abs_err.append(median_absolute_error(_y_true,_y_pred))
        scores.append(_model.score(X[test],_y_true))
        self.stats = _stats

    def print_stats():
        print(f"\t  Explained Variance    = {exp_var_sc[this_rf]:.4f}")
        print(f"\t  Mean Absolute Error   = {mean_abs_err[this_rf]:.4f}")
        print(f"\t  Mean Squared Error    = {mean_sq_err[this_rf]:.4f}")
        print(f"\t  Median Absolute Error = {median_abs_err[this_rf]:.4f}")
        print(f"\t  R^2 Score             = {scores[this_rf]:.4f} \n")


class AverageRFClassifier(BaseAverageRF):

    RFModel = RandomForestClassifier

    conf_mats, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []
    _stats = {"Training Time (s)":train_times,
              "Confusion Matrix":conf_mats,
              "Precision":precisions,
              "Recall":recalls,
              "F1":f1_scores,
              "Accuracy Score": scores,
              "ROC_AUC":roc_aucs}
    skf = StratifiedKFold(self.n_forests, random_state=None, shuffle=True)

    def predict(self, X):
        return mode(self._predict(X), axis=0)[0][0]

    def _predict_proba(self, X):
        func = "predict_proba"
        return self._evaluate_models(func, X)
    
    def _predict_log_proba(self,X):
        func = "predict_log_proba"
        return self._evaluate_models(func, X)

    def calc_stats():
        scores.append(_model.score(X[test],_y_true))
        conf_mats.append(confusion_matrix(_y_true, _y_pred))
        precisions.append(precision_score(_y_true, _y_pred))
        recalls.append(recall_score(_y_true, _y_pred))
        f1_scores.append(f1_score(_y_true, _y_pred))
        _y_prob = _model.predict_proba(X[test])[:, 1]  # For ROC_RUC
        roc_aucs.append(roc_auc_score(_y_true, _y_prob))
        self.stats = _stats

    
    def print_stats():
        print("Confusion Matrix = \n", conf_mats[this_rf])
        print(f"\t  Precision = {precisions[this_rf]:.4f}" )
        print(f"\t  Recall    = {recalls[this_rf]:.4f}"  )
        print(f"\t  F1-score  = {f1_scores[this_rf]:.4f}")
        print(f"\t  ROC AUC   = {roc_aucs[this_rf]:.4f}")
        print(f"\t  Score     = {scores[this_rf]:.4f} \n")

