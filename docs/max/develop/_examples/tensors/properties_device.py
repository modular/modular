# DOC: max/develop/tensors.mdx

from max.driver import CPU
from max.tensor import Tensor

# Tensor on CPU
cpu_tensor = Tensor.ones([2, 2], device=CPU())
