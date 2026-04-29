def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    tp = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
    fn = sum([1 if y_pred[i] != y_true[i] else 0 for i in range(len(y_pred))])

    return 2 * tp / (2 * tp + fn * 2)