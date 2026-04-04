# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -verify-diagnostics | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Support types
# ===----------------------------------------------------------------------=== #

trait RPTTrait(TrivialRegisterPassable):
    pass

# ===----------------------------------------------------------------------=== #
# Destructor tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @DtorExample1
# Shouldn't have a registered destructor because it's trivial and not explicit.
# It does have a destructor though because of AnyType conformance.
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample1(TrivialRegisterPassable, AnyType):
    var a: Int


# CHECK-LABEL: lit.struct.decl @DtorExample2
# Shouldn't have a registered destructor because it's trivial and not explicit
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample2(TrivialRegisterPassable, AnyType):
    var a: Int


# CHECK-LABEL: lit.struct.decl @DtorExample3
# Should have a registered destructor because it's explicit.
# CHECK-NEXT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample3(AnyType, RegisterPassable):
    var a: Int

    def __del__(deinit self):
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
def testCopyMoveSynth(var a: IntPair, var b: IntPairWrapper):
    # CHECK: lit.memcpy %a, %aCopy
    var aCopy = a

    # CHECK: lit.call {{.*}}IntPair::@"__init__{{.*}}"{{.*}}({{.*}}, %aMove){{.*}}*, "take"
    var aMove = a^

    # CHECK: lit.call {{.*}}IntPair::@"copy{{.*}}({{.*}}, %aExCopy)
    var aExCopy = a.copy()

    # CHECK: lit.memcpy %b, %bCopy
    var bCopy = b

    # CHECK: lit.call {{.*}}IntPairWrapper::@"__init__{{.*}}"{{.*}}({{.*}}, %bMove){{.*}}*, "take"
    var bMove = b^

    # CHECK: lit.call {{.*}}IntPairWrapper::@"copy{{.*}}({{.*}}, %bExCopy)
    var bExCopy = b.copy()


struct IntPairNT(ImplicitlyCopyable):
    var x: Int
    var y: Int
    comptime __copy_ctor_is_trivial = False


struct IntPairWrapperNT(ImplicitlyCopyable):
    var value: IntPairNT


# CHECK-LABEL: lit.fn @"testCopyMoveSynthNonTrivial
def testCopyMoveSynthNonTrivial(var a: IntPairNT, var b: IntPairWrapperNT):
    # CHECK: lit.call {{.*}}IntPairNT::@"__init__{{.*}}"{{.*}}({{.*}}, %aCopy){{.*}}*, "copy"
    var aCopy = a

    # CHECK: lit.call {{.*}}IntPairNT::@"__init__{{.*}}"{{.*}}({{.*}}, %aMove){{.*}}*, "take"
    var aMove = a^

    # CHECK: lit.call {{.*}}IntPairNT::@"copy{{.*}}({{.*}}, %aExCopy)
    var aExCopy = a.copy()

    # CHECK: lit.call {{.*}}IntPairWrapperNT::@"__init__{{.*}}"{{.*}}({{.*}}, %bCopy){{.*}}*, "copy"
    var bCopy = b

    # CHECK: lit.call {{.*}}IntPairWrapperNT::@"__init__{{.*}}"{{.*}}({{.*}}, %bMove){{.*}}*, "take"
    var bMove = b^

    # CHECK: lit.call {{.*}}IntPairWrapperNT::@"copy{{.*}}({{.*}}, %bExCopy)
    var bExCopy = b.copy()


# ===----------------------------------------------------------------------=== #
# Fieldwise init tests
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct FieldwiseInitExample1[T: Movable]:
    var x: Int
    var y: Self.T


# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample1
# CHECK: lit.fn @"__init__
# CHECK-SAME: (%x: !Int, %y: !lit.ref<:!Movable T, mut *"y`"> owned_in_mem,
# CHECK-SAME: %self: !lit.ref<{{.*}}> byref_result)
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[x]
# CHECK-NEXT: lit.ref.store %x, [[TMP]]
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[y]
# CHECK-NEXT: lit.call{{.*}}"__init__(take:$0$)">{{.*}}(%y, [[TMP]])
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>


# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample2
@fieldwise_init("implicit")
struct FieldwiseInitExample2:
    var x: Int


# CHECK-LABEL: lit.fn @"testFieldwiseInitExample2
# CHECK: FieldwiseInitExample2::@"__init__{{.*}}(%a, %b)
def testFieldwiseInitExample2(a: Int):
    var b: FieldwiseInitExample2 = a


# Register passable example.
# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample3
@fieldwise_init("implicit")
struct FieldwiseInitExample3(RegisterPassable):
    var x: Int


# ===----------------------------------------------------------------------=== #
# Shadow auto-parameterized parameters
# ===----------------------------------------------------------------------=== #


struct MyParam[p: Int]:
    pass


trait TraitWithPAlias:
    comptime p: Int = 42


# CHECK-LABEL: lit.struct.decl @MyStruct
# CHECK-SAME: <[{{.*}}]*"[[P1:.*]]": !Int, [{{.*}}]*"[[P2:.*]]": !Int, +, p: !Int,
# CHECK-SAME: m1: !lit.struct<#MyParam <:!Int *"[[P1]]">>, m2: !lit.struct<#MyParam <:!Int *"[[P2]]">>
struct MyStruct[p: Int, m1: MyParam[_], m2: MyParam[_]]:
    # CHECK: lit.fn @"__init__()"[
    def __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWithPVar
struct MyStructWithPVar[m1: MyParam[_]]:
    def __init__(out self):
        pass

    # COM: Ensure there's no conflict with this var.
    var p: Int


# CHECK-LABEL: lit.struct.decl @MyStructWithPAlias
struct MyStructWithPAlias[m1: MyParam[_]]:
    def __init__(out self):
        pass

    # COM: Ensure there's no conflict with this alias.
    comptime p: Int = 2


# CHECK-LABEL: lit.struct.decl @MyStructWithTraitWithPAlias
struct MyStructWithTraitWithPAlias[m1: MyParam[_]](TraitWithPAlias):
    # COM: Ensure there's no conflict with the inherited alias.
    def __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWithPFunc
struct MyStructWithPFunc[m1: MyParam[_]]:
    def __init__(out self):
        pass

    # COM: Ensure there's no conflict with this method (single definition).
    def p(self, x: Int):
        pass


# CHECK-LABEL: lit.struct.decl @MyStructWith2PFuncs
struct MyStructWith2PFuncs[m1: MyParam[_]]:
    def __init__(out self):
        pass

    # COM: Ensure there's no conflict with this method (multiple definitions).
    def p(self):
        pass

    def p(self, x: Int):
        pass

# ===----------------------------------------------------------------------=== #
# Nonmaterializable
# ===----------------------------------------------------------------------=== #


@fieldwise_init("implicit")
struct NmTarget(TrivialRegisterPassable):
    var x: Bool

    @always_inline("builtin")
    @implicit
    def __init__(out self, nms: NmStruct):
        self.x = True if (nms.x == 77) else False

    def __bool__(self) -> Bool:
        return self.x


@__nonmaterializable(NmTarget)
struct NmStruct(TrivialRegisterPassable):
    var x: Int

    @always_inline("builtin")
    @implicit
    def __init__(out self, x: Int):
        self.x = x

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        return NmStruct(self.x + rhs.x)


# CHECK: lit.alias.decl{{.*}}notMaterializedAlias{{.*}}NmStruct{{.*}}77
comptime notMaterializedAlias = NmStruct(77)
# CHECK: lit.alias.decl{{.*}}notMaterializedButConverted{{.*}}: !NmTarget = {{.*}}{:i1 0}})>
comptime notMaterializedButConverted: NmTarget = NmStruct(76)


def tail_types[T: AnyType, *U: AnyType](a: T, *b: *U):
    pass


def nmTargetNoop(x: NmTarget):
    pass


def nmResult() -> NmStruct:
    pass


# CHECK-LABEL: lit.fn @"useNonmaterializable
def useNonmaterializable(p: Bool):
    # CHECK: lit.var.decl "gotConverted1" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}{:i1 1}
    var gotConverted1 = NmStruct(76) + NmStruct(1)
    # CHECK: lit.var.decl "gotConverted2" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}{:i1 0}
    var gotConverted2 = notMaterializedAlias + NmStruct(1)
    # CHECK: lit.alias.decl{{.*}}useIfAlias{{.*}}NmStruct{{.*}}2
    comptime useIfAlias = NmStruct(2) if True else NmStruct(3)
    # CHECK: lit.var.decl "useIfVar" var : !lit.ref<!NmTarget
    var useIfVar = NmStruct(2) if p else NmStruct(77)

    # CHECK: lit.var.decl "useIfVarLopsided" var : !lit.ref<!NmTarget
    var useIfVarLopsided = NmTarget(False) if not p else NmStruct(77)

    # CHECK: lit.var.decl "useOrVar1" var : !lit.ref<!NmTarget
    var useOrVar1 = NmStruct(2) or NmStruct(77)
    # CHECK: lit.var.decl "useOrVar2" var : !lit.ref<!NmTarget
    var useOrVar2 = NmStruct(2) or NmStruct(3)
    # CHECK: lit.var.decl "useAndVar1" var : !lit.ref<!NmTarget
    var useAndVar1 = NmStruct(2) and NmStruct(77)
    # CHECK: lit.var.decl "useAndVar2" var : !lit.ref<!NmTarget
    var useAndVar2 = NmStruct(77) and NmStruct(77)

    # Test that parameter inference using nonmaterializable gives the target,
    # not the nonmaterializable type.
    # CHECK: call {{.*}}tail_types{{.*}}<:!AnyType !NmTarget, :param_list<!AnyType> []
    tail_types(NmStruct(5))
    # CHECK: call {{.*}}tail_types{{.*}}<:!AnyType !NmTarget, :param_list<!AnyType> [!NmTarget]
    tail_types(NmStruct(5), NmStruct(6))

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}nmResult{{.*}}()
    # CHECK: %nmResult = lit.var.decl
    # CHECK-NEXT: [[TMP2:%.*]] = lit.call {{.*}}NmTarget::@"__init__{{.*}}([[TMP]])
    # CHECK-NEXT: lit.ref.store [[TMP2]], %nmResult
    var nmResult = nmResult()
