import torch

# Linear Regression model f = w * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2.5, 5, 7.5, 10], dtype=torch.float32)

# weight initialization
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# feed forward vals
def feed_forward(x):
    return w * x


# Mean Squared Error
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()


learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # Predict
    y_pred = feed_forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient using backward propagation
    l.backward()

    """In PyTorch, the computation of gradients is automatically enabled within the default context, which means that 
    every operation performed within this context will track operations to compute gradients. However, 
    when you update the model parameters, you typically don't want these operations to be tracked, as they are part 
    of the optimization process and not the forward or backward pass of your model. Tracking these operations would 
    lead to unnecessary computations and increased memory usage."""

    # update weight
    with torch.no_grad():
        w -= w.grad * learning_rate

    """PyTorch accumulates gradients by default. This means that if you don't reset the gradients, the new gradient 
    will be added to the existing gradient in every backward pass. This can lead to incorrect gradient values and, 
    consequently, improper updates of the model parameters."""

    # Zero the gradients
    w.grad.zero_()
    print(f'epoch: {epoch} | Weight: {w:.8f} | Loss: {l:.5f} |')

print(f'Prediction after training: {feed_forward(torch.tensor([6, 7, 8, 9], dtype=torch.float32))}')
