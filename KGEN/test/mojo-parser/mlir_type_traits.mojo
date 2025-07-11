# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --kgen-print-inline-type-values %s | FileCheck %s


alias Composition = Movable & Copyable


# CHECK-LABEL: lit.fn @"mlir_type_trait_conformance
fn mlir_type_trait_conformance():
    # CHECK: !AnyType = <[{{.*}}::@__MLIRType<:type index>, index]>
    alias Any: AnyType = __mlir_type.index
    # CHECK: !Copyable = <[{{.*}}::@__MLIRType<:type index>, index]>
    alias Copy: Copyable = __mlir_type.index
    # CHECK: !Movable = <[{{.*}}::@__MLIRType<:type index>, index]>
    alias Move: Movable = __mlir_type.index
    # CHECK: !Movable_Copyable = <[{{.*}}::@__MLIRType<:type index>, index]>
    alias Comp: Composition = __mlir_type.index
