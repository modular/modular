# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --kgen-print-inline-type-values %s | FileCheck %s


comptime Composition = Movable & ImplicitlyCopyable


# CHECK-LABEL: lit.fn @"mlir_type_trait_conformance
fn mlir_type_trait_conformance():
    # CHECK: !AnyType = <[{{.*}}::@__MLIRType<:type index>, index]>
    comptime Any: AnyType = __mlir_type.index
    # CHECK: !ImplicitlyCopyable = <[{{.*}}::@__MLIRType<:type index>, index]>
    comptime Copy: ImplicitlyCopyable = __mlir_type.index
    # CHECK: !Movable = <[{{.*}}::@__MLIRType<:type index>, index]>
    comptime Move: Movable = __mlir_type.index
    # CHECK: !Movable_ImplicitlyCopyable = <[{{.*}}::@__MLIRType<:type index>, index]>
    comptime Comp: Composition = __mlir_type.index
