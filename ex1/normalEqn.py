import numpy as np


def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------
    X_transpose = np.transpose(X)
    A = np.dot(X_transpose, X)
    A = np.linalg.inv(A)
    B = np.dot(A, X_transpose)
    theta = np.dot(B, y)

# -------------------------------------------------------------

    return theta

# ============================================================

