import numpy as np
from sklearn.model_selection import train_test_split

# Classification scores
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, zero_one_loss

# Regression scores
from sklearn.metrics import explained_variance_score, median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

def train_test_split_data(X,y,rate_inactives=0.9):
    """
    Splits the data into train & test sets, taking into account a 
    desired proportion of actives / inactives.
    
    IMPORTANT: The data (X,y) should come ordered as 
               all actives, followed by all inactives.
    
    Arguments:
      X: The independent variables
      y: The dependent (True/False labels for actives/inactives)
      rate_inactive (Float [0,1]): Proportion of inactives desired in the final data set.

    """
    assert(X.shape[0] == y.shape[0]), "Size of arrays do not match!"

    n_actives = sum(y)
    rate_actives = 1 - rate_inactives
    n_inactives = int(n_actives * rate_inactives / rate_actives)
    data_size = n_actives + n_inactives

    if data_size > y.shape[0]:
        data_size = y.shape[0]
    
    # Slice the data with the right proportion of inactives. 
    # Assumes the data is ordered with actives first, then inactives.
    X_data, y_data = X[:data_size], y[:data_size]

    # Randomize data
    shuffle_index = np.random.permutation(data_size)
    X_data, y_data = X_data[shuffle_index], y_data[shuffle_index]

    # split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, stratify=y_data)
    
    print("X_train: ", len(X_train), "\ty_train: ",len(y_train),
         "\t(",sum(y_train),"actives and",len(y_train)-sum(y_train),"inactives)" )
    print("X_test:  ", len(X_test), " \ty_test:  ",len(y_test),
         " \t(",sum(y_test),"actives and",len(y_test)-sum(y_test),"inactives)" )

    return X_train, X_test, y_train, y_test

def evluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression stats for the model
    """
    y_pred = model.predict(X_test)

    evs = explained_variance_score(y_test,y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    med_ae = median_absolute_error(y_test, y_pred)
    r2sc = r2_score(y_test, y_pred)

    # Regression stats
    print(f"Explained Variance Score = {evs:.2f}")
    print(f"Mean Absolute Error      = {mae:.2f}")
    print(f"Mean SQUARED error       = {mse:.2f}")
    print(f"MEDIAN ansolute error    = {med_ae:.2f}")
    print(f"R^2 Score                = {r2sc:.2f}")

    return y_pred, {"Explained Variance":evs, "MAE":mae, "MSE":mse, "Med_AE":med_ae,"R2":r2sc}


def evaluate_classifier_model(model, X_test, y_test, title="Statistics"):
    # Evaluate metrics from test set
    # Gather model predictions for test set
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except:
        print("WARNING: Using binary labels as probabilities")
        y_proba = y_pred

    # Calculate perfomance measures
    # ROC
    fpr, tpr, thr_roc = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test,y_proba)

    # Precision x Recall
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Precision x Recll curves
    precisions, recalls, pr_thr = precision_recall_curve(y_test,y_pred)

    #import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,9))
    plot_precision_recall_vs_thresholds(precisions, recalls, pr_thr, ax[0,0])
    plot_precision_recall(precisions,recalls,ax[0,1])
    plot_roc_curve(fpr,tpr,ax[1,0])

    font={'fontfamily': 'monospace',
          'size': 'x-large'}
    res = ax[1,1]
    res.set_axis_off()
    res.annotate(title,(0.2,0.46), size='xx-large', fontweight='bold')
    res.annotate(f"Precision     = {precision:.4f}",(0.2,0.36), **font)
    res.annotate(f"Recall        = {recall:.4f}",   (0.2,0.28), **font)
    res.annotate(f"F1-score      = {f1:.4f}",       (0.2,0.20), **font)
    res.annotate(f"ROC AUC       = {auc:.4f}",      (0.2,0.12), **font)

    plt.show()
 
    return {"Precision": precision, "Recall":recall, "F1_Score":f1, "ROC_AUC":auc}

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds, ax=plt):      
    ax.plot(thresholds, precisions[:-1], linestyle="-", color="b", label='Precision')
    ax.plot(thresholds, recalls[:-1], linestyle="-", color="g", label='Recall')
    ax.set_xlabel("Threshold")
    ax.legend(loc='center left')
    ax.set_ylim([0,1])
    return
    
def plot_precision_recall(precisions, recalls, ax=plt):
    ax.plot(recalls, precisions, linestyle="-")
    ax.plot([0,1], [1,0], 'k--')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.axis([0,1,0,1])
    return

def plot_roc_curve(fpr,tpr,ax=plt):
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.plot(fpr, tpr, linewidth=1, linestyle="-")
    ax.plot([0,1], [0,1], 'k--')
    ax.axis([0,1,0,1])
    return
