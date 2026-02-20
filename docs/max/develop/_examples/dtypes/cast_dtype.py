# DOC: max/develop/dtypes.mdx

from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

# Create a float32 tensor
x = Tensor.constant([1.7, 2.3, 3.9], dtype=DType.float32, device=CPU())
print(f"Original dtype: {x.dtype}")  # DType.float32

# Cast to int32 (truncates decimal values)
y = x.cast(DType.int32)
print(f"After cast to int32: {y.dtype}")  # DType.int32

# Cast to float64 for higher precision
z = x.cast(DType.float64)
print(f"After cast to float64: {z.dtype}")  # DType.float64
