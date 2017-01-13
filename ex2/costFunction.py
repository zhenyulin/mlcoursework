import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    z = np.dot(X, theta)
    H = sigmoid(z)
    H0 = 1 - H
    H = np.log(H)
    H0 = np.log(H0)
    y0 = 1 - y
    H = np.multiply(H, y)
    H0 = np.multiply(H0, y0)
    H = np.add(H, H0)
    H = np.mean(H)
    J = -H
    return J
