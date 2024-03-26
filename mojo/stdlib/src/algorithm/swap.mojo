# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the swap function.

```mojo
from algorithm.swap import swap
```
"""


# ===----------------------------------------------------------------------===#
# Swap
# ===----------------------------------------------------------------------===#
@always_inline
fn swap[T: Movable](inout lhs: T, inout rhs: T):
    """Swaps the two given arguments.

    Parameters:
       T: Constrained to Copyable types.

    Args:
        lhs: Argument value swapped with rhs.
        rhs: Argument value swapped with lhs.
    """
    var tmp = lhs^
    lhs = rhs^
    rhs = tmp^
