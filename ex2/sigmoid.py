import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
    z = np.negative(z)
    g = np.exp(z)
    g = 1 / (g + 1)
# =============================================================
    return g