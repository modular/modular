# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn mogg_register(name: StringLiteral):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op.

    For instance:

    @mogg_register("mo.add")
    fn my_op[...](...):

    registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


fn mogg_register_override(name: StringLiteral, priority: Int):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op with an override priority.

    @mogg_register("mo.add", 1)
    fn my_op[...](...):

    Args:
      name: The name of the op to register.
      priority: The priority of the op.
    """
    return


fn mogg_register_shape_func(name: StringLiteral):
    """
    This decorator registers a given mojo function as being an implementation
    of a shape function for a mo op or a `mo.custom` op.

    @mogg_register_custom("tf.Something")
    fn something_impl[...](...):
        pass

    @mogg_register_custom_shape("tf.Something")
    fn something_shape_impl[...](...):
        pass

    Args:
      name: The name of the op to register.
    """
    return


fn mogg_elementwise():
    """
    This decorator marks a kernel as being elementwise. This implies the
    kernel represents a lambda to be executed in the inner loop of an
    elementwise function.
    """
    return


fn mogg_view_op():
    """
    This decorator marks a kernel as being a view operation. These are expected
    to return an NDBuffer with only the offset, strides, and/or shape changed.
    """
    return


fn mogg_takes_indices():
    """
    Tells the compiler that this kernel takes elementwise indices as its last
    argument.
    """
    return


fn mogg_tensor_allocator():
    """
    Marks this function as being the allocator of a tensor.
    """
    return


fn mogg_tensor_copy_constructor():
    """
    Marks this function as being the move constructor of a tensor.
    """
    return


fn mogg_tensor_deconstructor():
    """
    Marks this function as being the deconstructor of a tensor.
    """
    return


fn mogg_enable_fusion():
    """
    Marks this function as the trigger which enables fusion.
    """
    return


fn mogg_input_fusion_hook():
    """
    A hint that this is the template to look at to create the fusion lambda.
    """
    return


fn mogg_output_fusion_hook():
    """
    A hint that this is the template to look at to create the fusion lambda.
    """
    return


fn mogg_elementwise_hook():
    """
    A hint that this is the elementwise function.
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

    TODO(GRA-1145): Remove the need for this.
    """
    return
