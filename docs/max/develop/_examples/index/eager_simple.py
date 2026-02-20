# DOC: max/develop/index.mdx

from max import functional as F
from max.driver import CPU
from max.tensor import Tensor

# Create tensor from Python data
x = Tensor.constant([1.0, -2.0, 3.0, -4.0, 5.0], device=CPU())

y = F.relu(x)

# Results are available right away
print(y)
