import numpy as np

from ex2.sigmoid import sigmoid

def h(theta, x):
    return sigmoid(x.dot(theta))

def predictOneVsAll(thetas, X):
    """will return a vector of predictions
  for each example in the matrix X. Note that X contains the examples in
  rows. all_theta is a matrix where the i-th row is a trained logistic
  regression theta vector for the i-th class. You should set p to a vector
  of values from 1..K (e.g., p = [1 3 1 2] predicts classes 1, 3, 1, 2
  for 4 examples) """

    m = X.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters (one-vs-all).
#               You should set p to a vector of predictions (from 1 to
#               num_labels).
#
# Hint: This code can be done all vectorized using the max function.
#       In particular, the max function can also return the index of the 
#       max element, for more information see 'help max'. If your examples 
#       are in rows, then, you can use max(A, [], 2) to obtain the max 
#       for each row.
#       
    num_labels = len(thetas)
    for digit in range(num_labels):
        result = h(thetas[digit], X)
        p = np.column_stack((p, result))

    p = np.delete(p, 0, 1)
    p = np.argmax(p, axis=1)

# =========================================================================

    return p    # add 1 to offset index of maximum in A row
