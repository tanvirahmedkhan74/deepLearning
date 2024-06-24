import torch

"""
The network -
    x , w -> (w.x) -> y_hat , y -> (y_hat - y) -> s -> (s ** 2) -> loss
    
    Using Backward propagation or chain rule to calculate gradients - 
        dloss/dw = dloss/ds * ds/dy_hat * dy_hat/dw

    * Each step, pytorch calculates the local gradients and then when backward passing
    * It calculates the gradient using backward propagation
"""

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# step 1 - forward pass the vals and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)

# Then we update the weights and continue forward and backward pass
