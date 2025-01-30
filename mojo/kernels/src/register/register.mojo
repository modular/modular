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


fn register_internal_override(name: StringLiteral, priority: Int):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op with an override priority.

    @register_internal("mo.add", 1)
    fn my_op[...](...):

    Args:
      name: The name of the op to register.
      priority: The priority of the op.
    """
    return


fn register_internal_shape_func(name: StringLiteral):
    """
    This decorator registers a given mojo function as being an implementation
    of a shape function for a mo op or a `mo.custom` op.

    @register_internal_custom("tf.Something")
    fn something_impl[...](...):
        pass

    @register_internal_custom_shape("tf.Something")
    fn something_shape_impl[...](...):
        pass

    Args:
      name: The name of the op to register.
    """
    return


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    """
    Attaches the given intrinsic annotation onto the function.
    """
    return


fn uses_opaque():
    """
    Indicates the function may use opaque types. As a result, additional
    information about the funciton may be needed to lower things in the graph
    compiler.

    TODO(GEX-1145): Remove the need for this.
    """
    return
