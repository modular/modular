# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs print functions."""


from memory import stack_allocation

# We need to make sure that the call to vprintf consistently uses the same type,
# otherwise you end up with signature conflicts when using external_call.


@always_inline
fn _printf(fmt: StringLiteral):
    var tmp = 0
    var arg_ptr = Pointer.address_of(tmp)
    _ = external_call["vprintf", Int32](
        fmt.data(), arg_ptr.bitcast[Pointer[NoneType]]()
    )


@always_inline
fn _printf[*Ts: AnyRegType](fmt: StringLiteral, args: ListLiteral[Ts]):
    var args_stack = stack_allocation[1, ListLiteral[Ts]]()
    args_stack[0] = args
    _ = external_call["vprintf", Int32](
        fmt.data(), args_stack.bitcast[Pointer[NoneType]]()
    )
