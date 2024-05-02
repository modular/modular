# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo -verify-diagnostics | FileCheck %s

##===----------------------------------------------------------------------===##
# Struct with Nonmaterializable
##===----------------------------------------------------------------------===##


@value
@register_passable("trivial")
struct NmTarget:
    var x: Bool

    fn __init__(x: Bool) -> Self:
        return Self{x: x}

    @always_inline("nodebug")
    fn __init__(nms: NmStruct) -> Self:
        return Self{x: True if (nms.x == 77) else False}

    fn __bool__(self: Self) -> Bool:
        return self.x


@value
@nonmaterializable(NmTarget)
@register_passable("trivial")
struct NmStruct:
    var x: Int

    @always_inline("nodebug")
    fn __add__(self: Self, rhs: Self) -> Self:
        return NmStruct(self.x + rhs.x)


# CHECK: lit.alias.decl{{.*}}notMaterializedAlias{{.*}}NmStruct{{.*}}77
alias notMaterializedAlias = NmStruct(77)
# CHECK: lit.alias.decl{{.*}}notMaterializedButConverted{{.*}}NmTarget{{.*}}false
alias notMaterializedButConverted: NmTarget = NmStruct(76)


fn tail_types[T: AnyType, *U: AnyType](a: T, *b: *U):
    pass


fn nmTargetNoop(x: NmTarget):
    pass

# CHECK-LABEL: lit.func @"useNonmaterializable
fn useNonmaterializable(p: Bool):
    # CHECK: lit.var.decl "gotConverted1" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}true
    var gotConverted1 = NmStruct(76) + NmStruct(1)
    # CHECK: lit.var.decl "gotConverted2" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}false
    var gotConverted2 = notMaterializedAlias + NmStruct(1)
    # CHECK: lit.alias.decl{{.*}}useIfAlias{{.*}}NmStruct{{.*}}2
    alias useIfAlias = NmStruct(2) if True else NmStruct(3)
    # CHECK: lit.var.decl "useIfVar" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}false
    var useIfVar = NmStruct(2) if p else NmStruct(77)
    # CHECK: lit.var.decl "useIfVarLopsided" var : !lit.ref<!NmTarget
    # CHECK: kgen.param.constant: !NmTarget {{.*}}true
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
    # CHECK: call {{.*}}tail_types{{.*}}<:!AnyType #NmTarget1, :variadic<!AnyType> []>
    tail_types(NmStruct(5))
    # CHECK: call {{.*}}tail_types{{.*}}<:!AnyType #NmTarget1, :variadic<!AnyType> [#NmTarget1]>
    tail_types(NmStruct(5), NmStruct(6))
