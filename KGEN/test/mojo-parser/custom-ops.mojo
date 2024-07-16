# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# Change the order of definitions, so we check that the resulting
# kgen.custom.op_impls is sorted.

import _mlir

@op_implementation("custom.a")
struct CustomOpA:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

@op_implementation("custom.c")
struct CustomOpC:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

@op_implementation("custom.b")
struct CustomOpB:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

@op_implementation("custom.with_canonicalize")
struct CustomOpWithCanonicalize:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

    @staticmethod
    fn canonicalize(x: _mlir.Operation):
        return

fn main():
    var x: Int32 = 4
    var y: Int32 = 6
    var res = __mlir_op.`custom.a`[_type=Int32](x, y)
    print(res)

# Check that the custom ops are registered with `custom.op_impls`

# CHECK:      kgen.custom.op_impls @__CustomOpImplSymbol [
# CHECK-SAME:   <"custom.a",
# CHECK-SAME:     impl: :!lit.signature<{{.*}}> @"custom-ops"::@CustomOpA::@"impl({{[^\)]*}})">,
# CHECK-SAME:   <"custom.b",
# CHECK-SAME:     impl: :!lit.signature<{{.*}}> @"custom-ops"::@CustomOpB::@"impl({{[^\)]*}})">,
# CHECK-SAME:   <"custom.c",
# CHECK-SAME:     impl: :!lit.signature<{{.*}}> @"custom-ops"::@CustomOpC::@"impl({{[^\)]*}})">,
# CHECK-SAME:   <"custom.with_canonicalize",
# CHECK-SAME:     impl: :!lit.signature<{{.*}}> @"custom-ops"::@CustomOpWithCanonicalize::@"impl({{[^\)]*}})",
# CHECK-SAME:     canonicalize: :!lit.signature<{{.*}}> @"custom-ops"::@CustomOpWithCanonicalize::@"canonicalize({{[^\)]*}})">]
