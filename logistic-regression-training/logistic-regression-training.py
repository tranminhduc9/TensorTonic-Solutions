import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # X là n x d
    
    N, d = X.shape
    
    W = np.zeros(d)
    b = 0.0
    
    for _ in range(steps):
        z = X @ W + b
        p = _sigmoid(z)
        
        # Loss
        L = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        
        # Gradient
        gw = (X.T @ (p - y)) / N
        gb = np.mean(p - y)
        
        # Update
        W -= lr * gw
        b -= lr * gb
        
    return W, b