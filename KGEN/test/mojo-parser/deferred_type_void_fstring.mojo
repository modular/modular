# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

# Bare `__mlir_deferred_type` in the f-string `__mlir_op[`...`]` form must
# force `kgen.deferred` and skip parse-time verification, matching the
# dot-syntax `__mlir_op.<name>[...]` path.


# CHECK-LABEL: lit.fn @"force_defer_fstring_void(vector<4xf32>)"
def force_defer_fstring_void(a: __mlir_type.`vector<4xf32>`):
    # CHECK: kgen.deferred "some.unregistered.op
    __mlir_op[
        `some.unregistered.op %{a} : %{type_of(a)}`,
        _type=__mlir_deferred_type,
    ]


# Sibling negative case: without the `__mlir_deferred_type` marker, the same
# f-string template must not be deferred. `llvm.store` is a registered void op
# so the bracket form is well-formed without `_type=`.
# CHECK-LABEL: lit.fn @"no_defer_fstring_void(vector<4xf32>,!llvm.ptr)"
def no_defer_fstring_void(
    a: __mlir_type.`vector<4xf32>`, p: __mlir_type.`!llvm.ptr`
):
    # CHECK-NOT: kgen.deferred
    # CHECK: llvm.store
    __mlir_op[`llvm.store %{a}, %{p} : %{type_of(a)}, !llvm.ptr`]
