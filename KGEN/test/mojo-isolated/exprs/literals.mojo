# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn var_let_decls():
    # CHECK: %xx = lit.var.decl "xx" var
    # CHECK: %[[V1:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: lit.ref.store %[[V1]], %xx
    var xx = 42

    # CHECK: lit.alias.decl {{.*}}il{{.*}}@IntLiteral<:!pop.int_literal 43>
    alias il = 43

    # CHECK: %yy = lit.var.decl "yy" var
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !FloatDyn = <{:scalar<f64> "1"}> 
    # CHECK: lit.ref.store [[TMP]], %yy
    var yy = 1.0

    # CHECK: lit.alias.decl {{.*}}fl1{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<2|1>> = <*?>
    alias fl1 = 2.0
    # CHECK: lit.alias.decl {{.*}}fl2{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<63|10>> = <*?>
    alias fl2 = 6.3
    # CHECK: lit.alias.decl {{.*}}fl3{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<41|2>> = <*?>
    alias fl3 = 20.5
    # CHECK: lit.alias.decl {{.*}}fl4{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<-41|2>> = <*?>
    alias fl4 = -20.5
    # CHECK: lit.alias.decl {{.*}}fl5{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<neg_zero>> = <*?>
    alias fl5 = -0.0

    # Smallest positive float (moco-1796)
    # CHECK: lit.alias.decl {{.*}}fl6{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<1|2{{(0)+}}>> = <*?>
    alias fl6 = 5e-324

    # TODO - Python raises an error when dividing by zero.  We need support for
    # parameter-time evaluation of `raise` to support that semantics, in which
    # case these will be static errors instead.
    # CHECK: lit.alias.decl {{.*}}flDivZero{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<nan>> = <*?>
    alias flDivZero = 5.0 / 0.0
    # CHECK: lit.alias.decl {{.*}}flDivNegZero{{.*}}@FloatLiteral<:!pop.float_literal #pop.float_literal<nan>> = <*?>
    alias flDivNegZero = 5.0 / -0.0

    # CHECK: %str = lit.var.decl {{.*}} : !lit.ref<{{.*}}StringLiteral<:string "hello">
    # CHECK: [[CONST:%.*]] = kgen.param.constant: {{.*}}@StringLiteral<:string "hello"> = <*?>
    # CHECK: lit.ref.store [[CONST]], %str
    var str = "hello"

# ===----------------------------------------------------------------------=== #
# List Literals
# ===----------------------------------------------------------------------=== #

@register_passable("trivial")
struct IntList:
   fn __init__(out self, *list_elements: Int, __list_literal__: ()): pass

@register_passable("trivial")
struct List[T: AnyType]:
   fn __init__(out self, *list_elements: T, __list_literal__: ()): pass


fn inspect(list: List[FloatDyn]):
    pass

# CHECK-LABEL: lit.fn @"test_list_literal
fn test_list_literal():
    #var a = [1, 2, 3]

    # CHECK: lit.call {{.*}}Tuple::@"__init__{{.*}}([[EMPTY_TUPLE:%.*]]) :
    # CHECK-NEXT: [[TMP1:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: [[TMP3:%.*]] = kgen.param.constant: !Int = <{3}>
    # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.create [[[TMP1]], [[TMP2]], [[TMP3]]]
    # CHECK-NEXT: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK-NEXT: lit.call {{.*}}IntList::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]])
    var b : IntList = [1, 2, 3]

    # CHECK: lit.call {{.*}}Tuple::@"__init__{{.*}}([[EMPTY_TUPLE:%.*]]) :
    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: variadic<!Int> = <[]>
    # CHECK-NEXT: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK-NEXT: lit.call {{.*}}IntList::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]])
    var c : IntList = []



    # CHECK: lit.call {{.*}}List::@"__init__{{.*}}<:!AnyType #FloatDyn1>
    inspect([1.0, 2])

