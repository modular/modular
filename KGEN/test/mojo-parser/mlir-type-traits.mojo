# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --kgen-print-inline-vtables %s | FileCheck %s


trait Composition(Movable, Copyable):
    pass


# CHECK-LABEL: lit.func @"mlir_type_trait_conformance
fn mlir_type_trait_conformance():
    # CHECK: !AnyType = <[index, {"__del__" : {{.*}}@__MLIRType::@"__del__
    alias Any: AnyType = __mlir_type.index
    # CHECK: !Copyable = <[index, {"__copyinit__" : {{.*}}@__MLIRType::@"__copyinit__
    # CHECK-SAME: "__del__"
    alias Copy: Copyable = __mlir_type.index
    # CHECK: !Movable = <[index, {"__moveinit__" : {{.*}}@__MLIRType::@"__moveinit__
    # CHECK-SAME: "__del__"
    alias Move: Movable = __mlir_type.index
    # CHECK: !Composition = <[index,
    # CHECK-SAME: "__moveinit__"
    # CHECK-SAME: "__del__"
    # CHECK-SAME: "__copyinit__"
    alias Comp: Composition = __mlir_type.index
