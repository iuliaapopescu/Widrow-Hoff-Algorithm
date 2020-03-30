import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def compute_y(x, W, bias):
     return (-x * W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
     x1 = -0.5
     y1 = compute_y(x1, W, b)
     x2 = 0.5
     y2 = compute_y(x2, W, b)
     plt.clf()
     color = 'r'
     if(current_y == -1):
         color = 'b'
     plt.ylim((-1, 2))
     plt.xlim((-1, 2))
     plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
     plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
     plt.plot(current_x[0], current_x[1], color+'s')
     plt.plot([x1, x2] ,[y1, y2], 'black')
     plt.show(block=False)
     plt.pause(0.3)

def train_perceptron(X, y, epochs, learning_rate):
    no_features = X.shape[1]
    
    weights = np.zeros(no_features)
    bias = 0
    
    no_samps = X.shape[0]
    
    accuracy = 0.0
    
    for _ in range(epochs):
        X, y = shuffle(X, y)
        
        for idx_samps in range(no_samps):
            y_hat = np.dot(X[idx_samps][:], weights) + bias
            loss = (y_hat - X[idx_samps]) ** 2
            weights -= learning_rate * (y_hat - y[idx_samps]) * X[idx_samps][:]
            bias -= learning_rate * (y_hat - y[idx_samps])
            accuracy = np.mean(np.sign(np.dot(X, weights) + bias) == y)
            
            plot_decision_boundary(X, y, weights, bias, X[idx_samps][:], y[idx_samps])
            
    return weights, bias, accuracy
        

X = np.array([[0,0], [1,0], [0,1], [1,1]])

# Training a perceptron with the Widrow-Hoff algorithm with labels [-1, 1, 1, 1]

y = np.array([-1, 1, 1, 1])

epochs = 70
learning_rate = 0.1

weights_OR, bias_OR, accuracy_OR = train_perceptron(X, y, epochs, learning_rate)

print(weights_OR)
print(bias_OR)
print(accuracy_OR)

# Training a perceptron with the Widrow-Hoff algorithm with labels [-1, 1, 1, -1]

y = np.array([-1, 1, 1, -1])

weights_XOR, bias_XOR, accuracy_XOR = train_perceptron(X, y, epochs, learning_rate)

print(weights_XOR)
print(bias_XOR)
print(accuracy_XOR)


