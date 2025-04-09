# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn register_internal(name: StaticString):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op.

    For instance:

    @register_internal("mo.add")
    fn my_op[...](...):

    registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


fn __mogg_intrinsic_attr(intrin: StaticString):
    """
    Attaches the given intrinsic annotation onto the function.
    """
    return
