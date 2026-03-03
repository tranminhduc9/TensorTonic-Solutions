import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    c = np.unique(y, return_counts=True)
    p = c[1]/sum(c[1])
    return - np.sum(p * np.log2(p))