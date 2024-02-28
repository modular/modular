# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn op(name: StringLiteral, priority: Int = -1):
    """
    This decorator registers a given mojo function as being an implementation
    of an operation.

    For instance:

    ```mojo
    import max
    @max.register.op("mo.add")
    fn my_op[...](...):
    ```

    registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
      priority: The priority of the op.
    """
    return


fn elementwise():
    """
    This decorator marks a kernel as being elementwise. This implies the
    kernel represents a lambda to be executed in the inner loop of an
    elementwise function.

    For instance:
    ```mojo
    @max.register.op("mo.add")
    @max.register.elementwise()
    fn my_add[...](x: SIMD[...], y: SIMD[...]) -> SIMD[...]:
    ```
    """
    return
