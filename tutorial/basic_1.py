"""This file covers topics of Tensors Basics"""

# Tensors are the DSA for datas similar like arrays and vectors for numpy
import torch
import numpy as np

# 1d Tensor
one_dim_tensor = torch.empty(4)
two_dim_tensor = torch.empty(3, 4)
three_dim_tensor = torch.empty(2, 3, 4)

# Tensor with Random Values
rand_tensor = torch.rand(2, 3)

# Tensor with zeros and ones
zero_tensor = torch.zeros(2, 3)
one_tensor = torch.ones(2, 3)

# Tensor Datatypes
float_tensor = torch.ones(1, dtype=torch.float)  # float32
double_tensor = torch.ones(1, dtype=torch.double)  # float64

complex_float_tensor = torch.ones(1, dtype=torch.complex64)
complex_double_tensor = torch.ones(1, dtype=torch.complex128)

int_tensor = torch.ones(1, dtype=torch.int)  # int32
long_tensor = torch.ones(1, dtype=torch.long)  # int64

uint_tensor = torch.ones(1, dtype=torch.uint8)  # unsigned 8-bit int
int8_tensor = torch.ones(1, dtype=torch.int8)  # Signed 8-bit int

bool_tensor = torch.ones(1, dtype=torch.bool)  # boolean

float_16_tensor_pre = torch.ones(1, dtype=torch.float16)  # Good For Precisions
float_16_tensor_rng = torch.ones(1, dtype=torch.bfloat16)  # Good For Range

# Tensor Size
tensor_size = zero_tensor.size()

# Python List to Tensor Conversion
vals = [2.5, 3.8, 2.2, 6.9]
converted_tensor = torch.tensor(vals)

# Tensor Maths
x = torch.rand(2, 3)
y = torch.rand(2, 3)

add = torch.add(x, y)  # or x + y
sub = torch.sub(x, y)  # or x - y

# Inplace Operation (func with _())
y.add_(x)
y.mul_(x)

# Tensor Slicing
all_row_col_0 = x[:, 0]
row_0_all_col = x[0, :]

# Precision output
precise_output = x[1, 1].item()

# Reshaping Tensor
x = torch.rand(8, 8)
one_dim_16_ele = x.view(16)
auto_dim_tensor = x.view(-1, 8)  # Creates a 2x8 tensor

# Numpy and Tensor Conversion
x = torch.ones(5)
np_arr = x.numpy()

"""x and np_arr shares same memory location (CPU or GPU), if one changed with inplace, others value will also change"""

y = np.ones(5)
pt_tensor = torch.from_numpy(y)

# We can move Tensor and NP arrays in CPU and GPU if we have CUDA support
# requires_grad=True is specified for futhure optimization of the tensor
