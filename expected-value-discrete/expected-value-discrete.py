import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if np.sum(p) != 1:
        raise ValueError("Probabilities must sum to 1")

    return np.sum(np.array(x) * np.array(p))
