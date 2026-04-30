# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: kgen -emit=llvm %s | FileCheck %s --check-prefix=IR

# `nsw`/`nuw` are property-stored, so their presence in the LLVM IR proves
# `_properties` (whether `__mlir_attr` or `__mlir_deferred_attr`) survived
# the `kgen.deferred` round trip.

from std.collections.string.string_slice import _get_kgen_string


@no_inline
def add_with_nsw[
    width: Int
](
    a: __mlir_deferred_type[`i`, +width._mlir_value],
    b: __mlir_deferred_type[`i`, +width._mlir_value],
) -> __mlir_deferred_type[`i`, +width._mlir_value]:
    # IR: add nsw i32
    return __mlir_op.`llvm.add`[
        _type=__mlir_deferred_type[`i`, +width._mlir_value],
        _properties=__mlir_attr.`{overflowFlags = #llvm.overflow<nsw>}`,
    ](a, b)


@always_inline("nodebug")
def _overflow_kind_str[signed: Bool]() -> StaticString:
    comptime if signed:
        return "nsw"
    else:
        return "nuw"


@no_inline
def add_with_deferred_props[
    width: Int, signed: Bool
](
    a: __mlir_deferred_type[`i`, +width._mlir_value],
    b: __mlir_deferred_type[`i`, +width._mlir_value],
) -> __mlir_deferred_type[`i`, +width._mlir_value]:
    # IR: add nuw i32
    return __mlir_op.`llvm.add`[
        _type=__mlir_deferred_type[`i`, +width._mlir_value],
        _properties=__mlir_deferred_attr[
            `{overflowFlags = #llvm.overflow<`,
            +_get_kgen_string[_overflow_kind_str[signed]()](),
            `>}`,
        ],
    ](a, b)


def main():
    # CHECK: ok
    comptime w: Int = 32
    var a = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](
        Int32(5)._mlir_value
    )
    var b = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](
        Int32(7)._mlir_value
    )
    _ = add_with_nsw[w](a, b)
    _ = add_with_deferred_props[w, False](a, b)
    print("ok")
