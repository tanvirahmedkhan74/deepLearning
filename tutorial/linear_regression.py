import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0 > Prepare the data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# y_numpy is an 1-D array thus needs to be reshaped into a column vector
# Reshaping into a column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1 > Model Initialization
model = nn.Linear(in_features=n_features, out_features=n_features)

# 2 > Loss and optimizer
learning_rate = 0.01
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop - forward pass > backward pass > update weights
n_epochs = 300
for epoch in range(n_epochs):
    y_predicted = model(X)
    loss = loss_function(y_predicted, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | loss: {loss.item():.5f}')

# Plot the datas
predictions = model(X).detach().numpy()     # removed this step from the computational graph
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predictions, 'b')
plt.show()



