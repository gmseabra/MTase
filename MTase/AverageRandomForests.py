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

# Chemistry
from rdkit import Chem
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Progress Bar
from tqdm import trange, tqdm_notebook
from tqdm.auto import tqdm

class ActivityClassifier():
    """
    Class to hold a classifier object, that determines if molecules
    should be active or not based on the SMILES string.
    """

    def __init__(self, classifier, fp_type='rdkit'):
        self.classifier = classifier

        assert(fp_type.lower() in ['rdkit', 'morgan']), f"Fingerprint type not recognized: '{fp_type}'"
        self.fp_type = fp_type.lower()

    def predict(self, smiles):
        """
        Predicts the probability that a certain SMILES string 
        corresponds to an inactive or active molecule.
        """

        if isinstance(smiles,str):
            predictions = self.classify_one(smiles)

        else:
            predictions = []
            for smi in tqdm(smiles, desc="Generating predictions:"):
                predictions.append(self.classify_one(smi))
        
        return predictions

    def classify_one(self, smi):
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return -1
        else:
            fp = self.get_fingerprint(mol)
            return self.classifier.predict_proba([fp])[0][1]

    def get_fingerprint(self, mol):
        """
        Gets the fingerprint according to the chosen fingerprint type
        """
        if self.fp_type == 'rdkit':
            fp = RDKFingerprint(mol)
        elif self.fp_type == 'morgan':
            fp = GetMorganFingerprintAsBitVect(mol,2)

        return np.array(list(map(int,fp.ToBitString())))




class AverageRF():
    """
    An averaged Random Forest [classifier / regressor]. 
    
    Builds n_forests RFs, then predicions are the average (or mode) of the obtained for each RF.

    It is also possile to obtain an array with the individual predictions for each forest,
    by adding a underscore before the name of the function (e.g. "_predict")
    
    """
    def __init__(self, model_type, n_forests=5,verbosity=0, *kargs, **kwargs):
        # n_estimators="warn", verbose=0,
        # Variables 
        self.model_type = model_type.lower()
        self.n_forests = n_forests
        self.verbosity = verbosity

        self.models = []
        self.stats = {}

        assert (model_type in ["classifier","regressor"]), f"Unrecognized model type: {model_type}"
        if self.model_type == "classifier":
            RFModel = RandomForestClassifier
        else:
            RFModel = RandomForestRegressor

        for _ in range(n_forests):
            self.models.append(RFModel(*kargs, **kwargs))

    def fit(self,X,y):
        """
        Train the RF models usinig stratified K-fold splits, 
        with K = n_forests
        """
        train_times= []
        scores = []

        if self.model_type == "classifier":
            conf_mats, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []
            _stats = {"Training Time (s)":train_times,
                      "Confusion Matrix":conf_mats,
                      "Precision":precisions,
                      "Recall":recalls,
                      "F1":f1_scores,
                      "Accuracy Score": scores,
                      "ROC_AUC":roc_aucs}
            skf = StratifiedKFold(self.n_forests, random_state=None, shuffle=True)
        else:
            exp_var_sc, mean_abs_err, mean_sq_err, median_abs_err = [], [], [], []
            _stats = {"Training Time (s)":train_times,
                      "Explained Variance":exp_var_sc,
                      "Mean Absolute Error": mean_abs_err,
                      "Mean Squared Error": mean_sq_err,
                      "Median Absolute Error": median_abs_err,
                      "R^2":scores} 
            skf = KFold(n_splits=self.n_forests, random_state=None, shuffle=True)

        #Splits the data into stratified k-folds
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

            if self.model_type == "classifier":
                scores.append(_model.score(X[test],_y_true))
                conf_mats.append(confusion_matrix(_y_true, _y_pred))
                precisions.append(precision_score(_y_true, _y_pred))
                recalls.append(recall_score(_y_true, _y_pred))
                f1_scores.append(f1_score(_y_true, _y_pred))
                _y_prob = _model.predict_proba(X[test])[:, 1]  # For ROC_RUC
                roc_aucs.append(roc_auc_score(_y_true, _y_prob))
            else:
                exp_var_sc.append(explained_variance_score(_y_true, _y_pred))
                mean_abs_err.append(mean_absolute_error(_y_true,_y_pred))
                mean_sq_err.append(mean_squared_error(_y_true,_y_pred))
                median_abs_err.append(median_absolute_error(_y_true,_y_pred))
                scores.append(_model.score(X[test],_y_true))

            if self.verbosity > 0:
                # Print info
                print(f"SPLIT: {this_rf}  [TRAIN: {len(train)} \tTEST: {len(test)}] \t Training Time:{_elapsed:.3f} seconds.")
                if self.model_type == "classifier":
                    print("Confusion Matrix = \n", conf_mats[this_rf])
                    print(f"\t  Precision = {precisions[this_rf]:.4f}" )
                    print(f"\t  Recall    = {recalls[this_rf]:.4f}"  )
                    print(f"\t  F1-score  = {f1_scores[this_rf]:.4f}")
                    print(f"\t  ROC AUC   = {roc_aucs[this_rf]:.4f}")
                    print(f"\t  Score     = {scores[this_rf]:.4f} \n")
                else:
                    print(f"\t  Explained Variance    = {exp_var_sc[this_rf]:.4f}")
                    print(f"\t  Mean Absolute Error   = {mean_abs_err[this_rf]:.4f}")
                    print(f"\t  Mean Squared Error    = {mean_sq_err[this_rf]:.4f}")
                    print(f"\t  Median Absolute Error = {median_abs_err[this_rf]:.4f}")
                    print(f"\t  R^2 Score             = {scores[this_rf]:.4f} \n")
                            
            self.stats = _stats
        return self

    # The following functions return the average of the
    # results obtained for all the RFs.
    # Those are the functions that should be called externally.
    
    def predict(self, X):
        if self.model_type == "classifier":
            _pred = mode(self._predict(X), axis=0)[0][0]
        else:
            _pred = np.average(self._predict(X), axis=0)
        return _pred
    
    def score(self, X, y, sample_weight=None):
        return np.average(self._score(X, y, sample_weight), axis=0)

    # Those are only available for classifiers
    def predict_log_proba(self,X):
        assert (self.model_type == "classifier"), "predict_log_proba is only available for classifiers."
        return np.average(self._predict_log_proba(X), axis=0)

    def predict_proba(self, X):
        assert (self.model_type == "classifier"), "predict_proba is only available for classifiers."
        return np.average(self._predict_proba(X), axis=0)

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

    # The following methods are only available for classifiers:
    def _predict_proba(self, X):
        assert (self.model_type == "classifier"), "predict_proba is only available for classifiers."
        func = "predict_proba"
        return self._evaluate_models(func, X)
    
    def _predict_log_proba(self,X):
        assert (self.model_type == "classifier"), "predict_log_proba is only available for classifiers."
        func = "predict_log_proba"
        return self._evaluate_models(func, X)