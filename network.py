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

    return A_theta, Y

# Activation function (sigmoid)
def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

# Feed forward process
def feed_forward(A_theta):
    # Layer 1 calculation
    Z1 = W1 @ A_theta + b1
    A1 = sigmoid(Z1)

    # Layer 2 calculation
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # Layer 3 calculation
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    y_hat = A3 # Predicted output
    return y_hat

A_theta, Y = prepare_data()
y_hat = feed_forward(A_theta)