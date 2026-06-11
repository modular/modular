# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

# Bare `__mlir_deferred_type` as `_type=` adds no result types but forces
# `kgen.deferred`. `-verify-diagnostics` with no `expected-*` markers
# asserts no diagnostics are emitted.

# CHECK-LABEL: lit.fn @"force_defer_no_results()"
def force_defer_no_results():
    # CHECK: kgen.deferred "some.unregistered.op"
    __mlir_op.`some.unregistered.op`[_type = __mlir_deferred_type]()


# CHECK-LABEL: lit.fn @"force_defer_alongside_concrete_op()"
def force_defer_alongside_concrete_op():
    # Op is registered and well-formed; bare marker forces deferral anyway.
    # CHECK: kgen.deferred "index.constant"
    __mlir_op.`index.constant`[
        _type = __mlir_deferred_type,
        value = __mlir_attr.`0 : index`,
    ]()
