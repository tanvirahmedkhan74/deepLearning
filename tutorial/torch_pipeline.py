"""
Step - 1: Design Model (Input and Output size, forward pass and layers)
Step - 2: Construct loss and optimizer
Step - 3: Training Loop - Compute the gradients using backward pass and update weight with each epoch
"""

import torch
import torch.nn as nn

# Linear Regression model f = w * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2.5], [5], [7.5], [10]], dtype=torch.float32)

no_of_samples, no_of_features = X.shape

input_size = no_of_features
output_size = no_of_features

# model = nn.Linear(input_size, output_size)
# We can also create a class for this model


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

# Mean Squared Error
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_iters = 100

for epoch in range(n_iters):
    # Predict
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # gradient using backward propagation
    l.backward()
    # Update the weights
    optimizer.step()
    # Empty the grads
    optimizer.zero_grad()

    [w, b] = model.parameters()
    print(f'epoch: {epoch} || Loss: {l:.5f} || W: {w[0][0].item()} || Bias: {b.item()}')

print(f'Prediction after training: {model(torch.tensor([10], dtype=torch.float32)).item()}')
