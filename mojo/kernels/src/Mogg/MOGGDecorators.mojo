# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn mogg_register(name: StringLiteral):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op. For instance:

    @mogg_register("mo.add")
    fn my_op[...](...):

    registers `my_op` as an implementation of `mo.add`.
    """
    return


# TODO(#27757): This is a temporary decorator until kernels can be written
#               with Mojo async/await, in which case we can inspect the kernel
#               signature directly.
fn mogg_will_become_async():
    """
    This decorator indicates the decorated Mojo function should be called on a
    sub-task so as to not to block the main executor. Once kernels are
    rewritten to use Mojo async/await this responsibility will fall back to the
    kernel author, and no decorator will be necessary.
    """
    return


fn mogg_register_override(name: StringLiteral, priority: Int):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op with an override priority.

    @mogg_register("mo.add", 1)
    fn my_op[...](...):
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
