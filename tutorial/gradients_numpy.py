import numpy as np

# Linear Regression model f = w * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2.5, 5, 7.5, 10], dtype=np.float32)

# weight initialization
w = 0.0


# feed forward vals
def feed_forward(x):
    return w * x


# Mean Squared Error
def loss(y, y_hat):
    return ((y_hat - y)**2).mean()


# Gradient Calc
# LOSS = MSE = 1/N * (w * x - y ) ^ 2
# dLoss/dw = 1/N * 2 * (wx - y) * x
def gradient(x, y, y_hat):
    return np.dot(2 * x, y_hat - y).mean()


learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # Predict
    y_pred = feed_forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient
    g = gradient(X, Y, y_pred)
    # update weight
    w = w - g*learning_rate

    print(f'epoch: {epoch} | Weight: {w:.3f} | Loss: {l:.8f} | dw: {g:.8f}')

print(f'Prediction after training: {feed_forward(np.array([6, 7, 8, 9], dtype=np.float32))}')
