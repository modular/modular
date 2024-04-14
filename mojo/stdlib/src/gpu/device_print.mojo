# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs print functions."""


from builtin.builtin_list import _LITRefPackHelper


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
fn _printf[*types: AnyType](fmt: StringLiteral, *args: *types):
    var kgen_pack = _LITRefPackHelper(args._value).get_as_kgen_pack()

    # FIXME(37129): Cannot use get_loaded_kgen_pack because vtables on types
    # aren't stripped off correctly.
    var loaded_pack = __mlir_op.`kgen.pack.load`(kgen_pack)

    _ = external_call["vprintf", Int32](
        fmt.data(), UnsafePointer.address_of(loaded_pack)
    )
