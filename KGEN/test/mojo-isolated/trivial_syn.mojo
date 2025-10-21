# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


struct X_T(ImplicitlyCopyable, Movable):
    pass


struct Y_T(ImplicitlyCopyable, Movable):
    pass


struct X_N(ImplicitlyCopyable, Movable):
    fn __del__(deinit self):
        pass


struct X_T_U(ImplicitlyCopyable, Movable):
    fn __del__(deinit self):
        pass

    # User marked __del__as trivial
    alias __del__is_trivial: Bool = True


# CHECK-LABEL: lit.struct.decl @C
#  CHECK-SAME: <X: [[X_TYPE:!.*]], Y: [[Y_TYPE:!.*]]>
struct C[X: ImplicitlyCopyable & Movable, Y: ImplicitlyCopyable & Movable](
    ImplicitlyCopyable, Movable
):
    var x: X
    var y: Y

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::AnyType" {
    # CHECK-NEXT:    kgen.witness "__del__{{.*}}"
    # CHECK-NEXT:    kgen.witness "__del__is_trivial" : !Bool = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::AnyType", "__del__is_trivial">)

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Copyable" {
    # CHECK-NEXT:      kgen.witness "__copyinit__{{.*}}"
    # CHECK-NEXT:      kgen.witness "copy{{.*}}"
    # CHECK-NEXT:      kgen.witness "__copyinit__is_trivial" : !Bool = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Copyable", "__copyinit__is_trivial">)

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::ImplicitlyCopyable" {
    # CHECK-NEXT:   }

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Movable" {
    # CHECK-NEXT:    kgen.witness "__moveinit__{{.*}}"
    # CHECK-NEXT:    kgen.witness "__moveinit__is_trivial" : !Bool = cond(
    # CHECK-SAME:      #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[Y_TYPE]] Y, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">,
    # CHECK-SAME:        #kgen.get_witness<:[[X_TYPE]] X, "stdlib::builtin::stubs::Movable", "__moveinit__is_trivial">)


# CHECK-LABEL: lit.struct.decl @StructMLIRTypeOnly
struct StructMLIRTypeOnly(ImplicitlyCopyable & Movable):
    var x: __mlir_type.index
    var y: __mlir_type.index

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 1}>
    # CHECK-DAG: lit.alias.decl __moveinit__is_trivial: !Bool = <{:i1 1}>
    # CHECK-DAG: lit.alias.decl __copyinit__is_trivial: !Bool = <{:i1 1}>


# MOCO-2396:
# CHECK-LABEL: lit.struct.decl @NotTrivial
struct NotTrivial(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn __del__(deinit self):
        pass

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __moveinit__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __copyinit__is_trivial: !Bool = <{:i1 0}>


# CHECK-LABEL: lit.struct.decl @Wrapper
struct Wrapper(Copyable, Movable):
    var value: NotTrivial

    # Should be parser-folded.

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __moveinit__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __copyinit__is_trivial: !Bool = <{:i1 0}>
