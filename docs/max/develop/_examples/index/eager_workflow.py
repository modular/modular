# DOC: max/develop/index.mdx

from max import functional as F
from max import random
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

# Create input data
x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU())

# Create random weights
w = random.gaussian(
    [2, 2], mean=0.0, std=0.1, dtype=DType.float32, device=CPU()
)

# Forward pass - each operation executes as you write it
z = x @ w  # Matrix multiply
h = F.relu(z)  # Activation
out = h.mean()  # Reduce to scalar

# Inspect intermediate results anytime
print(f"Input shape: {x.shape}")
print(f"After matmul: {z.shape}")
print(f"Output: {out}")
