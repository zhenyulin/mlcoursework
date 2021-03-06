import numpy as np

from ex2.sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#
    X = np.insert(X, 0, np.ones(m), 1)
    h1 = sigmoid(X.dot(Theta1.transpose()))
    h1 = np.insert(h1, 0, np.ones(m), 1)
    h2 = sigmoid(h1.dot(Theta2.transpose()))
    p = np.argmax(h2, axis=1)

# =========================================================================

    return p        # add 1 to offset index of maximum in A row

