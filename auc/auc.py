import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    a = 0
    for i in range(len(fpr) - 1):
        a += (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i])/2

    return a