# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn register_internal(name: StringLiteral):
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


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    """
    Attaches the given intrinsic annotation onto the function.
    """
    return


fn enforce_io_param():
    """
    Decorator meant to ease the transition fo MangedTensorSlice needing to
    specialize on the io_spec param in kernel struct execute calls.
    """
    return
