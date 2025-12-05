# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Support types
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
trait RPTTrait:
    pass


# ===----------------------------------------------------------------------=== #
# Destructor tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @DtorExample1
# Shouldn't have a registered destructor because it's trivial and not explicit.
# It does have a destructor though because of AnyType conformance.
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable("trivial")
struct DtorExample1(AnyType):
    var a: Int


# CHECK-LABEL: lit.struct.decl @DtorExample2
# Shouldn't have a registered destructor because it's trivial and not explicit
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable("trivial")
struct DtorExample2(AnyType):
    var a: Int


# CHECK-LABEL: lit.struct.decl @DtorExample3
# Should have a registered destructor because it's explicit.
# CHECK-NEXT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable
struct DtorExample3(AnyType):
    var a: Int

    fn __del__(deinit self):
        pass


# CHECK-LABEL: lit.struct.decl @DtorExample4
# Shouldn't have a registered destructor because it's trivial and not explicit
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample4[T: RPTTrait]:
    var thing: Self.T


# CHECK-LABEL: lit.struct.decl @DtorExample5
# Should have a registered destructor because T has a destructor.
# CHECK-NEXT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample5[T: AnyType]:
    var thing: Self.T


# ===----------------------------------------------------------------------=== #
# Copy/Move synthesis tests
# ===----------------------------------------------------------------------=== #


struct IntPair(ImplicitlyCopyable):
    var x: Int
    var y: Int


struct IntPairWrapper(ImplicitlyCopyable):
    var value: IntPair


# CHECK-LABEL: lit.struct.decl @IntPairWrapper
# CHECK-LABEL: lit.fn @"copy
# CHECK-SAME: (%self: !lit.ref<!IntPairWrapper{{.*}}> read_mem,
# CHECK-SAME: %__result__: !lit.ref<!IntPairWrapper{{.*}}> byref_result)
# CHECK-NEXT: lit.call {{.*}}@Copyable::@"copy($0){{.*}}(%self, %__result__)


# CHECK-LABEL: lit.fn @"testCopyMoveSynth
fn testCopyMoveSynth(var a: IntPair, var b: IntPairWrapper):
    # CHECK: lit.call {{.*}}IntPair::@"__copyinit__{{.*}}({{.*}}, %aCopy)
    var aCopy = a

    # CHECK: lit.call {{.*}}IntPair::@"__moveinit__{{.*}}({{.*}}, %aMove)
    var aMove = a^

    # CHECK: lit.call {{.*}}IntPair::@"copy{{.*}}({{.*}}, %aExCopy)
    var aExCopy = a.copy()

    # CHECK: lit.call {{.*}}IntPairWrapper::@"__copyinit__{{.*}}({{.*}}, %bCopy)
    var bCopy = b

    # CHECK: lit.call {{.*}}IntPairWrapper::@"__moveinit__{{.*}}({{.*}}, %bMove)
    var bMove = b^

    # CHECK: lit.call {{.*}}IntPairWrapper::@"copy{{.*}}({{.*}}, %bExCopy)
    var bExCopy = b.copy()


# ===----------------------------------------------------------------------=== #
# Fieldwise init tests
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct FieldwiseInitExample1[T: Movable]:
    var x: Int
    var y: Self.T

    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x
        self.y = other.y^


# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample1
# CHECK: lit.fn @"__init__
# CHECK-SAME: (%x: !Int, %y: !lit.ref<:!Movable T, mut *"y`"> owned_in_mem,
# CHECK-SAME: %self: !lit.ref<{{.*}}> byref_result)
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[x]
# CHECK-NEXT: lit.ref.store %x, [[TMP]]
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[y]
# CHECK-NEXT: lit.call{{.*}}"__moveinit__{{.*}}"{{.*}}(%y, [[TMP]])
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>


# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample2
@fieldwise_init("implicit")
struct FieldwiseInitExample2:
    var x: Int


# CHECK-LABEL: lit.fn @"testFieldwiseInitExample2
# CHECK: FieldwiseInitExample2::@"__init__{{.*}}(%a, %b)
fn testFieldwiseInitExample2(a: Int):
    var b: FieldwiseInitExample2 = a


# Register passable example.
# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample3
@fieldwise_init("implicit")
@register_passable
struct FieldwiseInitExample3:
    var x: Int


# ===----------------------------------------------------------------------=== #
# Shadow auto-parameterized parameters
# ===----------------------------------------------------------------------=== #


struct MyParam[p: Int]:
    pass


trait TraitWithPAlias:
    comptime p: Int = 42


# CHECK-LABEL: lit.struct.decl @MyStruct
# CHECK-SAME: <[{{.*}}]*"[[P2:.*]]": !Int, [{{.*}}]*"[[P1:.*]]": !Int, +, p: !Int,
# CHECK-SAME: m1: !lit.struct<#MyParam <:!Int *"[[P1]]">>, m2: !lit.struct<#MyParam <:!Int *"[[P2]]">>
struct MyStruct[p: Int, m1: MyParam[_], m2: MyParam[_]]:
    # CHECK: lit.fn @"__init__()"[
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWithPVar
struct MyStructWithPVar[m1: MyParam[_]]:
    fn __init__(out self):
        pass

    # COM: Ensure there's no conflict with this var.
    var p: Int


# CHECK-LABEL: lit.struct.decl @MyStructWithPAlias
struct MyStructWithPAlias[m1: MyParam[_]]:
    fn __init__(out self):
        pass

    # COM: Ensure there's no conflict with this alias.
    comptime p: Int = 2


# CHECK-LABEL: lit.struct.decl @MyStructWithTraitWithPAlias
struct MyStructWithTraitWithPAlias[m1: MyParam[_]](TraitWithPAlias):
    # COM: Ensure there's no conflict with the inherited alias.
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWithPFunc
struct MyStructWithPFunc[m1: MyParam[_]]:
    fn __init__(out self):
        pass

    # COM: Ensure there's no conflict with this method (single definition).
    fn p(self, x: Int):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWith2PFuncs
struct MyStructWith2PFuncs[m1: MyParam[_]]:
    fn __init__(out self):
        pass

    # COM: Ensure there's no conflict with this method (multiple definitions).
    fn p(self):
        pass

    fn p(self, x: Int):
        pass
