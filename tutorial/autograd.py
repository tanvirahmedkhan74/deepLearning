import torch

x = torch.randn(3, requires_grad=True)

y = x + 2

print('Y ', y)
z = y * y * 2
print('Z ', z)
z = z.mean()
print('Z mean ', z)

v = torch.tensor([0.1, 1.2, 0.0001], dtype=torch.float32)

z.backward()  # dz/dx , when scalar
z.backward(v)  # dz/dx , when not scalar

print(x.grad)

# For preventing Gradient History
"""
1. x.requires_grad(false)
2. x.detach()
3. with torch.no_grad():
"""

# for optimization
# weights.grad.zero_()
