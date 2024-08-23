# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# Change the order of definitions, so we check that the resulting
# kgen.custom.op_impls is sorted.

import _mlir


# CHECK-LABEL: lit.struct.decl @CustomOpA
# CHECK-SAME: customOpName = "custom.a"
@op_implementation("custom.a")
struct CustomOpA:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y


# CHECK-LABEL: lit.struct.decl @CustomOpC
# CHECK-SAME: customOpName = "custom.c"
@op_implementation("custom.c")
struct CustomOpC:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y


# CHECK-LABEL: lit.struct.decl @CustomOpB
# CHECK-SAME: customOpName = "custom.b"
@op_implementation("custom.b")
struct CustomOpB:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y


# CHECK-LABEL: lit.struct.decl @AddConstantOp
# CHECK-SAME: customOpName = "custom.add_constant"
@op_implementation("custom.add_constant")
struct AddConstantOp:
    @staticmethod
    fn impl[cst: Int32](x: Int32) -> Int32:
        return x + cst


# CHECK-LABEL: lit.struct.decl @CustomOpWithCanonicalize
# CHECK-SAME: customOpName = "custom.with_canonicalize"
@op_implementation("custom.with_canonicalize")
struct CustomOpWithCanonicalize:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

    @staticmethod
    fn canonicalize(x: _mlir.Operation):
        return


fn main():
    # CHECK: %{{.*}} = "custom.a"(%{{[A-Za-z0-9]+}}, %{{[A-Za-z0-9]+}})
    # CHECK-SAME: __custom_op_struct_ref = {{.*}}
    var res = CustomOpA(19, 23)
    print(res)

    # CHECK: %{{.*}} = "custom.add_constant"(%{{[A-Za-z0-9]+}})
    # CHECK-SAME: __custom_op_struct_ref = {{.*}}, _op_impl_params = {{.*}}
    var res2 = AddConstantOp[19](23)
    print(res2)


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
