import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def forward(X, W_1, b_1, W_2, b_2): 
    z_1 = np.matmul(X, W_1) + b_1
    a_1 = np.tanh(z_1)
    z_2 = np.matmul(a_1, W_2) + b_2
    a_2 = sigmoid(z_2)
    
    return z_1, a_1, z_2, a_2


def backward(a_1, a_2, z_1, W_2, X, y, num_samples):
    dz_2 = a_2 - y
    dw_2 = np.matmul(a_1.T, dz_2) / num_samples
    db_2 = np.sum(dz_2, axis=0) / num_samples
    
    da_1 = np.matmul(dz_2, W_2.T)
    dz_1 = np.multiply(da_1, tanh_derivative(z_1))
    dw_1 = np.matmul(X.T, dz_1) / num_samples
    db_1 = np.sum(dz_1, axis=0) / num_samples
    
    return dw_1, db_1, dw_2, db_2


def compute_y(x, W, bias):
    return (-x*W[0] - bias) / (W[1] + 1e-10)


def plot_decision(X_, W_1, W_2, b_1, b_2):
    plt.clf()
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)


X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y = np.expand_dims(np.array([0, 1, 1, 0]), 1)

no_hidden = 5
no_out = 1

W_1 = np.random.normal(0, 1, (2, no_hidden))
b_1 = np.zeros((no_hidden))
W_2 = np.random.normal(0, 1, (no_hidden, no_out)) 
b_2 = np.zeros((no_out))

N = X.shape[0]

epochs = 70
learning_rate = 0.5

for _ in range(epochs):
    X, y = shuffle(X, y)
    
    z_1, a_1, z_2, a_2 = forward(X, W_1, b_1, W_2, b_2)
    loss = np.mean(-(y * np.log(a_2) + (1 - y) * np.log(1 - a_2)))
    
    accuracy = np.mean((np.round(a_2) == y))
    print(accuracy)
    
    dw_1, db_1, dw_2, db_2 = backward(a_1, a_2, z_1, W_2, X, y, N)
    
    W_1 -= learning_rate * dw_1
    b_1 -= learning_rate * db_1
    W_2 -= learning_rate * dw_2
    b_2 -= learning_rate * db_2
    
    plot_decision(X, W_1, W_2, b_1, b_2)
    
    
