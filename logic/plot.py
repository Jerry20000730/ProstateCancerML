import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_train(y_train, train_probs):
    fpr, tpr, thresholds = roc_curve(y_train, train_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.scatter(best_fpr, best_tpr, color='black')
    plt.text(best_fpr, best_tpr, f"{best_thresh:.3f} ({best_fpr:.3f}, {best_tpr:.3f})",
             fontsize=9, ha='left')
    plt.text(0.6, 0.2, f"AUC: {auc_score:.3f}", fontsize=10)
    plt.title("A")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(-0.5, 1.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_test(y_test, test_probs):
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.scatter(best_fpr, best_tpr, color='black')
    plt.text(best_fpr, best_tpr, f"{best_thresh:.3f} ({best_fpr:.3f}, {best_tpr:.3f})",
             fontsize=9, ha='left')
    plt.text(0.6, 0.2, f"AUC: {auc_score:.3f}", fontsize=10)
    plt.title("B")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(-0.5, 1.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_dca(y_test, test_probs):
    thresholds = np.linspace(0.01, 0.4, 100)
    net_benefit = []
    treat_all = []
    treat_none = np.zeros_like(thresholds)
    y = y_test.values

    for thresh in thresholds:
        pred_positive = test_probs >= thresh
        tp = ((pred_positive == 1) & (y == 1)).sum()
        fp = ((pred_positive == 1) & (y == 0)).sum()
        n = len(y)
        nb = tp / n - fp / n * (thresh / (1 - thresh))
        net_benefit.append(nb)
        treat_all.append(y.mean() - (1 - y.mean()) * (thresh / (1 - thresh)))

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, net_benefit, color="brown", label="regression")
    plt.plot(thresholds, treat_all, linestyle=":", color="black", label="All")
    plt.plot(thresholds, treat_none, linestyle="--", color="gray", label="None")
    plt.xlabel("High Risk Threshold")
    plt.ylabel("Net Benefit")
    plt.title("C")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    