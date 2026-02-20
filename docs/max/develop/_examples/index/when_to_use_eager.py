# DOC: max/develop/index.mdx

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor


def debug_forward_pass(x: Tensor) -> Tensor:
    """Forward pass with intermediate inspection."""
    # Can print/inspect at any point
    print(f"Input: {x}")

    z = x * 2
    print(f"After multiply: {z}")

    h = F.relu(z)
    print(f"After ReLU: {h}")

    return h


x = Tensor.constant([-1.0, 0.0, 1.0, 2.0], dtype=DType.float32, device=CPU())
result = debug_forward_pass(x)
