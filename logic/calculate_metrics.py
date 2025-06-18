from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

def calculate_metrics(y_true, y_probs, y_preds, dataset_type="train"):
    """
    Calculate and return various metrics for model evaluation.
    
    :param y_true: True labels.
    :param y_probs: Predicted probabilities.
    :param y_preds: Predicted labels.
    :param dataset_type: Type of dataset ('train' or 'test').
    :return: Dictionary containing calculated metrics.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    specificity = 1 - best_fpr
    auc_score = auc(fpr, tpr)
    acc = accuracy_score(y_true, y_preds)
    
    metrics = {
        "分组": dataset_type,
        "AUC": round(auc_score, 3),
        "敏感性": f"{round(best_tpr * 100, 1)}%",
        "特异性": f"{round(specificity * 100, 1)}%",
        "准确性": f"{round(acc * 100, 1)}%"
    }
    
    return metrics