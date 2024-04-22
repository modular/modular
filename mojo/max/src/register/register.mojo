# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements decorators to register MAX custom ops."""


fn op(name: StringLiteral, priority: Int = -1):
    """Registers a function as an op implementation.

    The decorator takes the name of the op as its argument.

    For example, this registers `my_op` as an implementation of `mo.add`:

    ```mojo
    from max import register

    @register.op("mo.add")
    fn my_op[...](...):
    ```

    Args:
      name: The name of the op to register.
      priority: The priority of the op.
    """
    return


fn elementwise():
    """Declares an elementwise operator.

    This decorator marks an op as being elementwise. This implies that the op
    represents a lambda that should be executed in the inner loop of an
    elementwise function.

    For example:

    ```mojo
    from max import register

    @register.op("mo.add")
    @register.elementwise()
    fn my_add[...](x: SIMD[...], y: SIMD[...]) -> SIMD[...]:
    ```
    """
    return
