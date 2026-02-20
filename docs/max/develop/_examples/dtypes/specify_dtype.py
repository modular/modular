# DOC: max/develop/dtypes.mdx

from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

# Create a tensor with float32 (default for most operations)
float_tensor = Tensor.ones([2, 3], dtype=DType.float32, device=CPU())
print(f"Float tensor dtype: {float_tensor.dtype}")

# Create a tensor with int32 for indices or counts
int_tensor = Tensor.constant([1, 2, 3], dtype=DType.int32, device=CPU())
print(f"Int tensor dtype: {int_tensor.dtype}")
