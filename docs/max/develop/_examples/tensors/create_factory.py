# DOC: max/develop/tensors.mdx

from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

# Tensor filled with ones
ones = Tensor.ones([3, 4], dtype=DType.float32, device=CPU())
print(ones)

# Tensor filled with zeros
zeros = Tensor.zeros([2, 3], dtype=DType.float32, device=CPU())
print(zeros)
