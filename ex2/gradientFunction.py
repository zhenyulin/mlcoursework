from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples
    n = len(X[0])

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    z = np.dot(X, theta)
    H = sigmoid(z)
    D = np.subtract(H,y)
    D = np.transpose(D)
    D = np.repeat(D, n, axis=0)
    D = D.reshape(m,n)
    D = np.multiply(D,X)
    D = np.mean(D, axis=0)
    grad = D
    
    return grad
