# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


struct X_T(ImplicitlyCopyable):
    pass


struct Y_T(ImplicitlyCopyable):
    pass


struct X_N(ImplicitlyCopyable):
    def __del__(deinit self):
        pass


struct X_T_U(ImplicitlyCopyable):
    def __del__(deinit self):
        pass

    # User marked __del__as trivial
    comptime __del__is_trivial: Bool = True


# CHECK-LABEL: lit.struct.decl @C
#  CHECK-SAME: <X: [[X_TYPE:!.*]], Y: [[Y_TYPE:!.*]]>
struct C[X: ImplicitlyCopyable, Y: ImplicitlyCopyable](ImplicitlyCopyable):
    var x: Self.X
    var y: Self.Y

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::AnyType" {
    # CHECK-NEXT:   }

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Copyable" {
    # CHECK-NEXT:      kgen.witness "__init__{{.*}}(*, "copy":{{.*}}"
    # CHECK-NEXT:      kgen.witness "copy{{.*}}"
    # CHECK: kgen.witness "__copy_ctor_is_trivial" : !Bool = sugar_builtin(apply({{.*}})

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::ImplicitlyCopyable" {
    # CHECK-NEXT:   }

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::ImplicitlyDestructible" {
    # CHECK-NEXT:    kgen.witness "__del__{{.*}}"
    # CHECK: kgen.witness "__del__is_trivial" : !Bool = sugar_builtin(apply({{.*}})

    # CHECK-LABEL:  kgen.conformance @"{{.*}}::Movable" {
    # CHECK-NEXT:    kgen.witness "__init__{{.*}}(*, "take":{{.*}}"
    # CHECK: kgen.witness "__move_ctor_is_trivial" : !Bool = sugar_builtin(apply({{.*}})


# CHECK-LABEL: lit.struct.decl @StructMLIRTypeOnly
struct StructMLIRTypeOnly(ImplicitlyCopyable):
    var x: __mlir_type.index
    var y: __mlir_type.index

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 1}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:i1 1}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:i1 1}>


# MOCO-2396:
# CHECK-LABEL: lit.struct.decl @NotTrivial
struct NotTrivial(Copyable):
    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __del__(deinit self):
        pass

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:i1 0}>


# CHECK-LABEL: lit.struct.decl @Wrapper
struct Wrapper(Copyable):
    var value: NotTrivial

    # Should be parser-folded.

    # CHECK-DAG: lit.alias.decl __del__is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:i1 0}>
    # CHECK-DAG: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:i1 0}>


# CHECK-LABEL: lit.struct.decl @TrivialFieldGen
# CHECK: lit.alias.decl __del__is_trivial: !Bool = <#kgen.get_witness<:!Movable T, "{{.*}}::ImplicitlyDestructible", "__del__is_trivial">>
# CHECK: lit.alias.decl __move_ctor_is_trivial: !Bool = <#kgen.get_witness<:!Movable T, "{{.*}}::Movable", "__move_ctor_is_trivial">>
struct TrivialFieldGen[T: Movable](Movable):
    var z: Self.T
    var y: Int
    var q: Self.T

# CHECK-LABEL: lit.struct.decl @TestTrivialRegisterPassable
# CHECK: lit.alias.decl __del__is_trivial: !Bool = <{:i1 1}>
# CHECK: lit.alias.decl __move_ctor_is_trivial: !Bool = <{:i1 1}>
# CHECK: lit.alias.decl __copy_ctor_is_trivial: !Bool = <{:i1 1}>
struct TestTrivialRegisterPassable[T: TrivialRegisterPassable](Copyable):
    var _value: Self.T
