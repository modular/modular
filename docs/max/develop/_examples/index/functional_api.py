# DOC: max/develop/index.mdx

from max import functional as F
from max.driver import CPU
from max.tensor import Tensor

# Force CPU execution to avoid GPU compiler issues
x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], device=CPU())

y = F.sqrt(x)  # Element-wise square root
z = F.softmax(x, axis=-1)  # Softmax along last axis

print(f"Input: {x}")
print(f"Square root: {y}")
print(f"Softmax: {z}")
