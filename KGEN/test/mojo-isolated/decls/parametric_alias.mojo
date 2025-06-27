# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

##===----------------------------------------------------------------------===##
# declarations
##===----------------------------------------------------------------------===##

# CHECK: ![[INT_META:.*]] = !lit.meta<!Int>

# CHECK: lit.alias.decl *"noParam{{.*}}": !Int = <{78}>
alias noParam: Int = 78

# CHECK: lit.alias.decl *"emptyParams{{.*}}": !Int = <{89}>
alias emptyParams[]: Int = 89

# CHECK: lit.alias.decl *"idInt{{.*}}": !lit.generator<<"x": !Int>!Int> = <#kgen.gen<*(0,0)>>
alias idInt[x: Int]: Int = x

# CHECK: lit.alias.decl *"myIntAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int>!Int> = <#kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>>
alias myIntAdd[x: Int, y: Int] = x + y

# CHECK: lit.alias.decl *"myDefaultAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int = {1}>!Int> = <#kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>>
alias myDefaultAdd[x: Int, y: Int = 1] = x + y

# CHECK: lit.alias.decl *"myDependentDefaultAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int = *(0,0)>!Int> = <#kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>>
alias myDependentDefaultAdd[x: Int, y: Int = x] = x + y

# CHECK: lit.alias.decl *"myIntFMA{{.*}}": !lit.generator<<"x": !Int, "y": !Int, "z": !Int>!Int> = <#kgen.gen<{value = add(mul(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">), #lit.struct.extract<:!Int *(0,2), "value">)}>>
alias myIntFMA[x: Int, y: Int, z: Int] = x * y + z


@fieldwise_init
struct PS[a: Int, b: Int, c: Int]:
    pass


# CHECK: lit.alias.decl *"PS_xy3{{.*}}": !lit.generator<<"x": !Int, "y": !Int>meta<!lit.struct<#PS <:!Int *(0,0), :!Int *(0,1), :!Int {3}>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int *(0,0), :!Int *(0,1), :!Int {3}>>>
alias PS_xy3[x: Int, y: Int] = PS[x, y, 3]

# CHECK: lit.alias.decl *"PS_21x{{.*}}": !lit.generator<<"x": !Int>meta<!lit.struct<#PS <:!Int {2}, :!Int {1}, :!Int *(0,0)>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int {2}, :!Int {1}, :!Int *(0,0)>>>
alias PS_21x[x: Int] = PS[2, 1, x]

# CHECK: lit.alias.decl *"PS_21xy{{.*}}": !lit.generator<<"x": !Int, "y": !Int>meta<!lit.struct<#PS <:!Int {2}, :!Int {1}, :!Int {value = mul(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int {2}, :!Int {1}, :!Int {value = mul(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>>>
alias PS_21xy[x: Int, y: Int] = PS[2, 1, x * y]

# CHECK: lit.trait.decl @MyTrait
trait MyTrait:
    # CHECK-NEXT: lit.alias.decl *"ParamType{{.*}}": !lit.generator<<"a": !Int>!AnyType>
    alias ParamType[a: Int]: AnyType

# CHECK: lit.struct.decl @MyStruct
struct MyStruct[a: Int, b: Int](MyTrait):
    # CHECK-NEXT: lit.alias.decl *"ParamType{{.*}}": !lit.generator<<"a1": !Int>![[INT_META]]> = <#kgen.gen<!Int>>
    alias ParamType[a1: Int] = Int
    # CHECK: kgen.conformance @"{{.*}}::MyTrait"
    # CHECK-NEXT: kgen.witness "ParamType" : !lit.generator<<"a": !Int>!AnyType> = rebind(:!lit.generator<<"a1": !Int>![[INT_META]]> #kgen.gen<!Int>)

##===----------------------------------------------------------------------===##
# usages
##===----------------------------------------------------------------------===##


# CHECK: lit.alias.decl *"myDouble{{.*}}": !lit.generator<<"x": !Int>!Int> = <#kgen.gen<{value = mul(#lit.struct.extract<:!Int *(0,0), "value">, 2)}>>
alias myDouble[x: Int] = myDependentDefaultAdd[x]


# CHECK-LABEL: fn @"expect_two_ints
# CHECK-SAME: <binop: !lit.generator<<"x": !Int, "y": !Int>!Int>>
fn expect_two_ints[binop: __type_of(myIntAdd)]():
    pass


# CHECK-LABEL: fn @"implicit_conversions()"
fn implicit_conversions():
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> #kgen.gen<
    expect_two_ints[myIntAdd]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> rebind(
    expect_two_ints[myDefaultAdd]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> bind_params(
    expect_two_ints[myIntFMA[z=2]]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> rebind(:!lit.generator<<"y": !Int, "z": !Int>!Int> bind_params(
    expect_two_ints[myIntFMA[x=2]]()


# CHECK-LABEL: lit.fn @"test_type_equality()"
fn test_type_equality():
    # CHECK-NEXT: %[[PS_345:.*]] = lit.var.decl "ps_345" {{.*}}@PS<:!Int {3}, :!Int {4}, :!Int {5}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {3}, :!Int {4}, :!Int {5}>(%[[PS_345]])
    var ps_345: PS[3, 4, 5] = PS[idInt[3], myIntAdd[2, 2], myDefaultAdd[4]]()

    # CHECK-NEXT: %[[PS_215:.*]] = lit.var.decl "ps_215" {{.*}}@PS<:!Int {2}, :!Int {1}, :!Int {5}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {2}, :!Int {1}, :!Int {5}>(%[[PS_215]])
    var ps_215: PS_21x[5] = PS[2, 1, 5]()

    # CHECK-NEXT: %[[PS_216:.*]] = lit.var.decl "ps_216" {{.*}}@PS<:!Int {2}, :!Int {1}, :!Int {6}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {2}, :!Int {1}, :!Int {6}>(%[[PS_216]])
    var ps_216: PS_21x[6] = PS_21xy[2, 3]()

    # CHECK-NEXT: %[[PS_213:.*]] = lit.var.decl "ps_213" {{.*}}@PS<:!Int {2}, :!Int {1}, :!Int {3}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {2}, :!Int {1}, :!Int {3}>(%[[PS_213]])
    var ps_213: PS_21x[myIntFMA[1, 3, 0]] = PS_xy3[2, 1]()


fn two_identical_inputs[T: AnyType](x: T, y: T):
    pass


# CHECK-LABEL: fn @"test_type_inference()"
fn test_type_inference():
    # CHECK: lit.call @parametric_alias::@"two_identical_inputs
    # CHECK-SAME: <:!AnyType [@parametric_alias::@PS<:!Int {2}, :!Int {1}, :!Int {5}>,
    # CHECK-SAME: "x": !lit.ref<@parametric_alias::@PS<:!Int {2}, :!Int {1}, :!Int {5}>
    # CHECK-SAME: "y": !lit.ref<@parametric_alias::@PS<:!Int {2}, :!Int {1}, :!Int {5}>
    two_identical_inputs(PS_21x[5](), PS[2, 1, 5]())


# CHECK-LABEL: fn @"partial_binding()"
fn partial_binding():
    # CHECK: lit.alias.decl *"myIntMulPlus3{{.*}}": !lit.generator<<"x": !Int, "y": !Int>!Int> = <bind_params(:!lit.generator<<"x": !Int, "y": !Int, "z": !Int>!Int> #kgen.gen<{value = add(mul(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">), #lit.struct.extract<:!Int *(0,2), "value">)}>, ?, ?, {3})>
    alias myIntMulPlus3 = myIntFMA[z=3]
    # CHECK: lit.alias.decl *"myIntMul2Plus3{{.*}}": !lit.generator<<"x": !Int>!Int> = <bind_params(:!lit.generator<<"x": !Int, "y": !Int, "z": !Int>!Int> #kgen.gen<{value = add(mul(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">), #lit.struct.extract<:!Int *(0,2), "value">)}>, ?, {2}, {3})>
    alias myIntMul2Plus3 = myIntMulPlus3[y=2]
    # CHECK: lit.alias.decl *"myEleven{{.*}}": !Int = <{11}>
    alias myEleven = myIntMul2Plus3[x=4]


# CHECK-LABEL: fn @"nested_generators()"
fn nested_generators():
    # CHECK-NEXT: lit.alias.decl *"myCurriedIntAdd{{.*}}": !lit.generator<<"x": !Int>!lit.generator<<"y": !Int>!Int>> = <#kgen.gen<bind_params(:!lit.generator<<"x": !Int, "y": !Int>!Int> #kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>, *(0,0), ?)>
    alias myCurriedIntAdd[x: Int] = myIntAdd[x]

    # CHECK-NEXT: lit.alias.decl *"myRenamedCurriedIntAdd{{.*}}": !lit.generator<<"a": !Int>!lit.generator<<"y": !Int>!Int>> = <#kgen.gen<bind_params(:!lit.generator<<"x": !Int, "y": !Int>!Int> #kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>, *(0,0), ?)>
    alias myRenamedCurriedIntAdd[a: Int] = myCurriedIntAdd[a]

    # CHECK-NEXT: lit.alias.decl *"myAdd2{{.*}}": !lit.generator<<"y": !Int>!Int> = <bind_params(:!lit.generator<<"x": !Int, "y": !Int>!Int> #kgen.gen<{value = add(#lit.struct.extract<:!Int *(0,0), "value">, #lit.struct.extract<:!Int *(0,1), "value">)}>, {2}, ?)>
    alias myAdd2 = myRenamedCurriedIntAdd[2]

    # CHECK-NEXT: lit.alias.decl *"myFive{{.*}}": !Int = <{5}>
    alias myFive = myAdd2[3]

    # CHECK-NEXT: lit.alias.decl *"mySix{{.*}}": !Int = <{6}>
    alias mySix = myRenamedCurriedIntAdd[2][4]
