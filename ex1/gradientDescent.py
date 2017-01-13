from computeCost import computeCost
import numpy as np


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        H = np.dot(X, theta)
        D = np.subtract(H,y)
        D = np.transpose(D)
        D = np.repeat(D, 2, axis=0)
        D = D.reshape(m,2)
        D = np.multiply(D,X)
        D = np.mean(D, axis=0)
        D = alpha*D
        theta = np.subtract(theta, D)

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
