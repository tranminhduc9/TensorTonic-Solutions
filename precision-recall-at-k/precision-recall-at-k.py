def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    topk = set(recommended[:k])
    precision = len(topk & set(relevant)) / k
    recall = len(topk & set(relevant)) / len(set(relevant))
    return [precision, recall]