# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# Change the order of definitions, so we check that the resulting
# kgen.custom.op_impls is sorted.

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

fn main():
    var x: Int32 = 4
    var y: Int32 = 6
    var res = __mlir_op.`custom.a`[_type=Int32](x, y)
    print(res)

# Check that the custom ops are registered with `custom.op_impls`
# CHECK: kgen.custom.op_impls [<"custom.a",
# CHECK-SAME:                    @"custom-ops"::@CustomOpA::@"impl(stdlib::builtin::simd::SIMD[{int32}, {1}],stdlib::builtin::simd::SIMD[{int32}, {1}])">,
# CHECK-SAME:                  <"custom.b",
# CHECK-SAME:                    @"custom-ops"::@CustomOpB::@"impl(stdlib::builtin::simd::SIMD[{int32}, {1}],stdlib::builtin::simd::SIMD[{int32}, {1}])">,
# CHECK-SAME:                  <"custom.c",
# CHECK-SAME:                    @"custom-ops"::@CustomOpC::@"impl(stdlib::builtin::simd::SIMD[{int32}, {1}],stdlib::builtin::simd::SIMD[{int32}, {1}])">]
