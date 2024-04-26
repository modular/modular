# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements decorators to register MAX custom ops."""


fn op(name: StringLiteral, priority: Int = 0):
    """Registers a function as a graph op.

    You can use this to override ops implemented in MAX Engine (regardless of
    whether the model is from PyTorch, ONNX, or MAX Graph), or to define
    completely custom ops for use with MAX Graph.

    For example, this registers `my_op` as an override implementation of
    the `mo.add` op:

    ```mojo
    from max import register

    @register.op("mo.add")
    fn my_op[...](...):
    ```

    Args:
      name: The name of the op to register.
      priority: The priority of the op. If there are multiple registered ops
          for the same op, this is used to determine which one to use in the
          graph (the highest value wins). You normally don't need to use this.
          It's only necessary if you're overriding another op override or
          custom op.
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
