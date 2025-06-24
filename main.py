import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

targets = np.array([[0],[1],[1],[0]])

input_size = 2
hidden_size = 2
output_size = 1

rng = np.random.default_rng(42)

W1 = rng.normal(0, 1,(input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = rng.normal(0, 1,(hidden_size, output_size))
b2 = np.zeros((1, output_size))

lr = 2
epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(inputs, W1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    error = targets - a2
    
    d_output = error * sigmoid_derivative(z2)
    
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(z1)
    
    W2 += np.dot(a1.T, d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    
    W1 += np.dot(inputs.T, d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f'Epoch {epoch} - Loss {loss}')
print("\n--- Risultati finali ---")
for x, y in zip(inputs, targets):
    out = sigmoid(np.dot(sigmoid(np.dot(x, W1) + b1), W2) + b2)
    print(f"Input: {x} => Predetto: {round(out.item(), 3)} (Atteso: {y[0]})")