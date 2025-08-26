# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


struct X_T(Copyable, Movable):
    pass


struct Y_T(Copyable, Movable):
    pass


struct X_N(Copyable, Movable):
    fn __del__(deinit self):
        pass


struct X_T_U(Copyable, Movable):
    fn __del__(deinit self):
        pass

    # User marked __del__as trivial
    alias __del__is_trivial: Bool = True


# CHECK-LABEL: lit.struct.decl @C
#  CHECK-SAME: <X: [[X_TYPE:!.*]], Y: [[Y_TYPE:!.*]]>
struct C[X: Copyable & Movable, Y: Copyable & Movable](Copyable, Movable):
    var x: X
    var y: Y

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::AnyType" {
    # CHECK-NEXT:    kgen.witness "__del__"
    # CHECK-NEXT:    kgen.witness "__del__is_trivial" : !Bool = {value: i1 = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">)

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Copyable" {
    # CHECK-NEXT:    kgen.witness "__copyinit__"
    # CHECK-NEXT:    kgen.witness "__copyinit__is_trivial" : !Bool = {value: i1 = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">)

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Movable" {
    # CHECK-NEXT:    kgen.witness "__moveinit__"
    # CHECK-NEXT:    kgen.witness "__moveinit__is_trivial" : !Bool = {value: i1 = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">)


# CHECK-LABEL: lit.struct.decl @StructMLIRTypeOnly
struct StructMLIRTypeOnly(Copyable & Movable):
    var x: __mlir_type.index
    var y: __mlir_type.index

    # CHECK-DAG: lit.alias.decl __del__is_trivial: i1 = <1>
    # CHECK-DAG: lit.alias.decl __moveinit__is_trivial: i1 = <1>
    # CHECK-DAG: lit.alias.decl __copyinit__is_trivial: i1 = <1>
