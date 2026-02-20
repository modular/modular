# DOC: max/develop/tensors.mdx

from max.dtype import DType
from max.tensor import Tensor

# Float tensor (default for most operations)
floats = Tensor.ones([2, 2], dtype=DType.float32)

# Integer tensor
integers = Tensor.ones([2, 2], dtype=DType.int32)
