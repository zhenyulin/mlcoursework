import numpy as np
from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    m = X.shape[0]

    def label(n):
        if n < 0.5:
            return 0
        else:
            return 1

    p = np.zeros(m)

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#


# =========================================================================
    Z = np.dot(X, theta)
    H = sigmoid(Z)
    label = np.vectorize(label)
    p = label(H)
    
    return p