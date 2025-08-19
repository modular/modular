# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


struct X_T:
    pass


struct Y_T:
    pass


struct X_N:
    fn __del__(deinit self):
        pass


struct X_T_U:
    fn __del__(deinit self):
        pass

    # User marked __del__as trivial
    alias __del__is_trivial = __mlir_attr.`1 : i1`


# CHECK-LABEL: lit.struct.decl @C
struct C[X: AnyType, Y: AnyType]:
    var x: X
    var y: Y

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::AnyType" {
    # CHECK-NEXT:    kgen.witness "__del__"
    # CHECK-NEXT:    kgen.witness "__del__is_trivial" : i1 = cond(
    # CHECK-SAME:      #kgen.get_witness<:!AnyType X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:!AnyType Y, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:!AnyType X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">


# CHECK-LABEL: lit.struct.decl @StructMLIRTypeOnly
struct StructMLIRTypeOnly:
    var x: __mlir_type.index
    var y: __mlir_type.index

    # CHECK: lit.alias.decl __del__is_trivial: i1 = <1>


# CHECK-LABEL: lit.fn @"main()"
fn main():
    alias TrivialC = C[X_T, Y_T]
    # CHECK: lit.alias.decl *"t1`{{.*}}": i1 = <1>
    alias t1 = TrivialC.__del__is_trivial
    # CHECK: lit.alias.decl *"t2`{{.*}}": i1 = <1>
    alias t2 = C[TrivialC, Y_T].__del__is_trivial

    alias TrivialCU = C[X_T_U, Y_T]
    # CHECK: lit.alias.decl *"t3`{{.*}}": i1 = <1>
    alias t3 = TrivialCU.__del__is_trivial
    # CHECK: lit.alias.decl *"t4`{{.*}}": i1 = <1>
    alias t4 = C[TrivialCU, Y_T].__del__is_trivial

    alias NonTrivialC = C[X_N, Y_T]
    # CHECK: lit.alias.decl *"n1`{{.*}}": i1 = <0>
    alias n1 = NonTrivialC.__del__is_trivial
    # CHECK: lit.alias.decl *"n2`{{.*}}": i1 = <0>
    alias n2 = C[NonTrivialC, Y_T].__del__is_trivial
