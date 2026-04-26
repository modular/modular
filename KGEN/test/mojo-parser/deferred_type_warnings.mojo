# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# When all parts of __mlir_deferred_type are trivially constructable (no
# unresolved parameters), the compiler emits a warning suggesting __mlir_type.
# expected-warning @+2 {{trivially constructable type. Use `__mlir_type` instead.}}
@always_inline
def trivially_constructable_return_type() -> __mlir_deferred_type[
    `!llvm.array<4 x f32>`
]:
    # expected-warning @+2 {{trivially constructable type. Use `__mlir_type` instead.}}
    return __mlir_op.`llvm.mlir.undef`[
        _type = __mlir_deferred_type[`!llvm.array<4 x f32>`]
    ]()
