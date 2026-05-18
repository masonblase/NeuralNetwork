import numpy as np

# Network architecture
L = 3
n = [2, 3, 3, 1]

# Random weights and biases
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# Training data and labels
def prepare_data():
    # Sample training data
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])
    # Sample labels (0 for healthy, 1 for unhealthy)
    y = np.array([[0], [1], [1], [0], [0], [1], [1], [0], [1], [0]])
    m = 10
    A_theta = X.T
    Y = y.reshape(n[L], m)

    return A_theta, Y, m

def cost(y_hat, y):
    """y_hat and y should be a n^L x m matrix"""
    # Losses is n^L x m matrix
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))
    m = y_hat.reshape(-1).shape[0]

    # Summing across axis = 1 makes this a n^L x 1 matrix
    summed_losses = (1/m) * np.sum(losses, axis = 1)

    return np.sum(summed_losses)

# Activation function
def g(z):
    return 1 / (1 + np.exp(-z))

# Feed forward process
def feed_forward(A_theta):
    # Layer 1 calculation
    Z1 = W1 @ A_theta + b1
    A1 = g(Z1)

    # Layer 2 calculation
    Z2 = W2 @ A1 + b2
    A2 = g(Z2)

    # Layer 3 calculation
    Z3 = W3 @ A2 + b3
    A3 = g(Z3)

    cache = {
        "A_theta": A_theta,
        "A1": A1,
        "A2": A2
    }

    return A3, cache

# Backpropagation process
def backprop_layer_3(y_hat, Y, m, A2, W3): # Layer L
    A3 = y_hat

    # Calculate dC/dZ3
    dC_dZ3 = (1/m) * (A3 - Y)
    assert dC_dZ3.shape == (n[3], m)

    # Calculate dC/dW3
    dZ3_dW3 = A2
    assert dZ3_dW3.shape == (n[2], m)

    dC_dW3 = dC_dZ3 @ dZ3_dW3.T
    assert dC_dW3.shape == (n[3], n[2])

    # Calculate dC/db3
    dC_db3 = np.sum(dC_dZ3, axis = 1, keepdims = True)
    assert dC_db3.shape == (n[3], 1)

    # Calculate propagator dC/dA2
    dZ3_dA2 = W3
    dC_dA2 = W3.T @ dC_dZ3
    assert dC_dA2.shape == (n[2], m)

    return dC_dW3, dC_db3, dC_dA2

def backprop_layer_2(propagator_dC_dA2, A1, A2, W2):
    # Calculate dC/dZ2
    dA2_dZ2 = A2 * (1 - A2)
    dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
    assert dC_dZ2.shape == (n[2], m)

    # Calculate dC/dW2
    dZ2_dW2 = A1
    assert dZ2_dW2.shape == (n[1], m)

    dC_dW2 = dC_dZ2 @ dZ2_dW2.T
    assert dC_dW2.shape == (n[2], n[1])

    # Calculate dC/db2
    dC_db2 = np.sum(dC_dZ2, axis = 1, keepdims = True)
    assert dC_db2.shape == (n[2], 1)

    # Calculate propagator dC/dA1
    dZ2_dA1 = W2
    dC_dA1 = W2.T @ dC_dZ2
    assert dC_dA1.shape == (n[1], m)

    return dC_dW2, dC_db2, dC_dA1

def backprop_layer_1(propagator_dC_dA1, A1, A_theta, W1):
    # Calculate dC/dZ1
    dA1_dZ1 = A1 * (1 - A1)
    dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
    assert dC_dZ1.shape == (n[1], m)

    # Calculate dC/dW1
    dZ1_dW1 = A_theta
    assert dZ1_dW1.shape == (n[0], m)

    dC_dW1 = dC_dZ1 @ dZ1_dW1.T
    assert dC_dW1.shape == (n[1], n[0])

    # Calculate dC/db1
    dC_db1 = np.sum(dC_dZ1, axis = 1, keepdims = True)
    assert dC_db1.shape == (n[1], 1)

    return dC_dW1, dC_db1

A_theta, Y, m = prepare_data()
y_hat = feed_forward(A_theta)