import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix'); plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.show()
    return cm

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC'); ax.legend()
    plt.show()
    return auc

def reliability_diagram(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0,1],[0,1],'--', label='Perfectly calibrated')
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
    ax.set_title('Reliability Diagram'); ax.legend()
    plt.show()
    return prob_true, prob_pred

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0; total = len(y_true)
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask): continue
        conf = y_prob[mask].mean()
        acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return ece

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return np.mean((y_prob - y_true) ** 2)

def entropy_hist(y_prob, eps=1e-12, bins=30):
    p = np.clip(y_prob, eps, 1 - eps)
    ent = -(p*np.log(p) + (1-p)*np.log(1-p))
    fig, ax = plt.subplots()
    ax.hist(ent, bins=bins)
    ax.set_title('Predictive Entropy'); ax.set_xlabel('Entropy'); ax.set_ylabel('Count')
    plt.show()
    return ent
