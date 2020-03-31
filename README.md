# Training a perceptron

>Step 1

Get number of features

>Step 2

Initialize weights with an array of zeros

>Step 3

In each epoch:

>>Step 3.1

Shuffle the training data `X, y = shuffle(X, y)`

>>Step 3.2

For each example in the training data:

>>>Step 3.2.1

Calculate prediction `y_hat = np.dot(X[idx_samps][:], weights) + bias`

>>>Step 3.2.2

Calculate error for current example `loss = (y_hat - X[idx_samps]) ** 2`

>>>Step 3.2.3

Update weights using derivative(loss) / derivative(weights) `weights -= learning_rate * (y_hat - y[idx_samps]) * X[idx_samps][:]`

>>>Step 3.2.4

Update bias using derivative(loss) / derivative(bias) `bias -= learning_rate * (y_hat - y[idx_samps])`

>>>Step 3.2.5

Calculate accuracy `accuracy = np.mean(np.sign(np.dot(X, weights) + bias) == y)`

# Training a neural network

>Step 1

Initialize weights of the hidden layer, bias of the hidden layer, weights of the output layer and bias of the output layer

>Step 2

In each epoch:

>>Step 2.1

Shuffle the training data

>>Step 2.2

The forward method: 
  - calculate z 
    - for each layer multiply the training data with the corespondent weights and add the bias to it
  - calculate a 
    - apply the activation function for z
    
>>Step 2.3

Calculate: 
  - the logistic loss
   `loss = np.mean(-(y * np.log(a_2) + (1 - y) * np.log(1 - a_2)))`
  - the accuracy
   `accuracy = np.mean((np.round(a_2) == y))`

>>Step 2.4

The backward method using the chain rule

>>Step 2.5

Update weights and biases using the learning rate

`
W_1 -= learning_rate * dw_1
b_1 -= learning_rate * db_1
W_2 -= learning_rate * dw_2
b_2 -= learning_rate * db_2
`
  

