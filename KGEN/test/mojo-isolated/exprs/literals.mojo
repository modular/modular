# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

@register_passable("trivial")
struct SimpleIntRange:
    fn __init__(out self): pass
    fn __has_next__(self) -> Bool:
        pass
    fn __len__(self) -> Int:
        pass
    fn __next__(mut self) -> Int:
        pass
    fn __iter__(self) -> Self:
        pass


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
   fn __init__(out self, *list_elements: Int, __list_literal__: () = ()): pass

   fn append(mut self, value: Int): pass

fn inspect(list: List[_]):
    pass

# CHECK-LABEL: lit.fn @"test_list_literal
fn test_list_literal():
    # CHECK: [[VARIADIC:%.*]] = pop.variadic.create
    # CHECK: [[TUPVAL:%.*]] = kgen.param.materialize{{.*}}@Tuple::@"__init__()"<:variadic<!AnyType> []>)
    # CHECK-NEXT: lit.ref.store [[TUPVAL]], [[EMPTY_TUPLE:%.*]] :
    # CHECK: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]], %a)
    var a = [1, 2, 3]

    # CHECK: [[TMP1:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: [[TMP3:%.*]] = kgen.param.constant: !Int = <{3}>
    # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.create [[[TMP1]], [[TMP2]], [[TMP3]]]
    # CHECK: [[TUPVAL:%.*]] = kgen.param.materialize{{.*}}@Tuple::@"__init__()"<:variadic<!AnyType> []>)
    # CHECK-NEXT: lit.ref.store [[TUPVAL]], [[EMPTY_TUPLE:%.*]] :
    # CHECK-NEXT: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK-NEXT: lit.call {{.*}}@IntList::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]])
    var b : IntList = [1, 2, 3]

    # CHECK: [[VARIADIC:%.*]] = kgen.param.constant: variadic<!Int> = <[]>
    # CHECK: [[TUPVAL:%.*]] = kgen.param.materialize{{.*}}@Tuple::@"__init__()"<:variadic<!AnyType> []>)
    # CHECK-NEXT: lit.ref.store [[TUPVAL]], [[EMPTY_TUPLE:%.*]] :
    # CHECK-NEXT: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK-NEXT: lit.call {{.*}}@IntList::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]])
    var c : IntList = []

    # CHECK: lit.call {{.*}}List::@"__init__{{.*}}<:!AnyType #FloatDyn1>
    inspect([1.0, 2])

# CHECK-LABEL: lit.fn @"test_list_comprehension
fn test_list_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}@List<:!AnyType #Int1>
    # CHECK: lit.loop cond {
    # CHECK:   SimpleIntRange::@"__has_next__
    # CHECK: } body {
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK-NEXT: lit.ref.store [[TMP]], %i1
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %i1
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@Int::@"__mul__{{.*}}([[TMP]], [[TMP2]]
    # CHECK:      lit.call {{.*}}@List::@"append
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = [i1*2 for i1 in SimpleIntRange()]

    # CHECK: %b_collection = lit.var.decl{{.*}}@List<:!AnyType #Int1>
    # CHECK: } body {
    # CHECK:   } body {
    # CHECK-NEXT:  [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK-NEXT: lit.ref.store [[TMP]], %i3
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %i2
    # CHECK-NEXT: [[TMP2:%.*]] = lit.ref.load %i3
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@Int::@"__mul__{{.*}}([[TMP]], [[TMP2]]
    # CHECK:      lit.call {{.*}}@List::@"append
    var b_collection = [i2*i3 for i2 in SimpleIntRange() for i3 in SimpleIntRange()]

    # Inferred to type IntList and using an "if" clause.
    # CHECK: %c_collection = lit.var.decl{{.*}}!lit.ref<!IntList,
    # CHECK: lit.loop cond {
    # CHECK:   SimpleIntRange::@"__has_next__
    # CHECK: } body {
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK-NEXT: lit.ref.store [[TMP]], %i4
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT:    [[TMP:%.*]] = lit.ref.load %i4
    # CHECK-NEXT:    @Int::@"__bool__
    # CHECK-NEXT:    @Bool::@"__mlir_i1__
    # CHECK-NEXT:    hlcf.elif.yield
    # CHECK-NEXT: } then {
    # CHECK-NEXT: [[RES:%.*]] = lit.ref.load %i4
    # CHECK-NEXT  lit.call {{.*}}@IntList::@"append{{.*}}(%c_collection, [[RES]])
    var c_collection : IntList = [i4 for i4 in SimpleIntRange() if i4]


# ===----------------------------------------------------------------------=== #
# Dictionary Literals
# ===----------------------------------------------------------------------=== #

struct MyDict[K: Movable, V: AnyType]:
    fn __init__(out self, owned keys: List[K], owned values: List[V], __dict_literal__: ()):
        pass

struct IntDict:
    fn __init__(out self, keys: IntList, values: IntList, __dict_literal__: () = ()):
        pass


# CHECK-LABEL: lit.fn @"test_dict_literal
fn test_dict_literal(aBool: Bool):
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[KEYS_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[VALUES_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@Dict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %a) :
    var a = {1: aBool, 2: aBool}

    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[KEYS_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@List::@"__init__{{.*}}({{.*}}, [[VALUES_LIST:%.*]]) :
    # CHECK: lit.call {{.*}}@MyDict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %b) :
    var b : MyDict[Int, Bool] = {1: aBool, 2: aBool}

    # CHECK: [[KEYS_LIST:%.*]] = lit.call {{.*}}@IntList::@"__init__
    # CHECK: [[VALUES_LIST:%.*]] = lit.call {{.*}}@IntList::@"__init__
    # CHECK: lit.call {{.*}}@IntDict::@"__init__{{.*}}([[KEYS_LIST]], [[VALUES_LIST]], {{.*}}, %c) :
    var c : IntDict = {1: 7, 2: 8}


# CHECK-LABEL: lit.fn @"test_dict_comprehension
fn test_dict_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}@Dict<:!AnyType #Int1, :!Copyable_Movable #String1>
    # CHECK: lit.loop cond {
    # CHECK:   SimpleIntRange::@"__has_next__
    # CHECK: } body {
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK-NEXT: lit.ref.store [[TMP]], %i
    # CHECK:       lit.call {{.*}}String::@"__init__()
    # CHECK:      [[TMP:%.*]] = lit.ref.immut %i
    # CHECK:      lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%a_collection, [[TMP]],
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = {i: String() for i in SimpleIntRange()}

# ===----------------------------------------------------------------------=== #
# Set Literals
# ===----------------------------------------------------------------------=== #

struct MySet[T: AnyType]:
    fn __init__(out self, owned *values: T, __set_literal__: ()):
        pass

fn param_infer_equal[T: AnyType](a: T, b: T): pass

# CHECK-LABEL: lit.fn @"test_set_literal
fn test_set_literal():
    # CHECK: [[VARIADIC:%.*]] = pop.variadic.create
    # CHECK: [[TUPVAL:%.*]] = kgen.param.materialize{{.*}}@Tuple::@"__init__()"<:variadic<!AnyType> []>)
    # CHECK-NEXT: lit.ref.store [[TUPVAL]], [[EMPTY_TUPLE:%.*]] :
    # CHECK: [[TUP_TMP:%.*]] = lit.ref.immut [[EMPTY_TUPLE]]
    # CHECK: lit.call {{.*}}@Set::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]], %a)
    var a = {1, 2, 3}

    # MOCO-1974 - Param inference isn't substituting full type
    param_infer_equal(a, {})

    # CHECK: [[VARIADIC:%.*]] = pop.variadic.create
    # CHECK: [[TUPVAL:%.*]] = kgen.param.materialize{{.*}}@Tuple::@"__init__()"<:variadic<!AnyType> []>)
    # CHECK-NEXT: lit.ref.store [[TUPVAL]], [[EMPTY_TUPLE:%.*]] :
    # CHECK: [[TUP_TMP:%.*]] = lit.ref.immut
    # CHECK: lit.call {{.*}}@MySet::@"__init__{{.*}}([[VARIADIC]], [[TUP_TMP]], %b)
    var b : MySet[Int] = {1, 2}


# CHECK-LABEL: lit.fn @"test_set_comprehension
fn test_set_comprehension():
    # CHECK-NEXT: %a_collection = lit.var.decl{{.*}}@Set<:!AnyType #Int1>
    # CHECK: lit.loop cond {
    # CHECK:   SimpleIntRange::@"__has_next__
    # CHECK: } body {
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}SimpleIntRange::@"__next__
    # CHECK-NEXT: lit.ref.store [[TMP]], %i1
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %i1
    # CHECK-NEXT: [[TMP2:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}@Int::@"__mul__{{.*}}([[TMP]], [[TMP2]]
    # CHECK:      lit.call {{.*}}@Set::@"add
    # CHECK-NEXT: lit.loop.continue
    # CHECK: }
    var a_collection = {i1*2 for i1 in SimpleIntRange()}


# ===----------------------------------------------------------------------=== #
# Initializer Lists
# ===----------------------------------------------------------------------=== #

struct InitType[T: AnyType]:
    fn __init__(out self, value: T):
        pass
    fn __init__(out self, value: T, value2: Int):
        pass

# CHECK-LABEL: lit.fn @"test_initializer_list
fn test_initializer_list():
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], %a)
    var a : InitType[Int] = {1}
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: [[TWO:%.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], [[TWO]], %b)
    var b : InitType[Int] = {1, 2}
    # CHECK: [[TMP:%.*]] = lit.ref.immut
    # CHECK: [[INT:%.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: lit.call {{.*}}@InitType::@"__init__{{.*}}([[TMP]], [[INT]], %c)
    var c : InitType[String] = {"foo", 42}

# ===----------------------------------------------------------------------=== #
# Ambiguity for e.g. PythonObject
# ===----------------------------------------------------------------------=== #

# This can be formed with any collection and has its own initializer list too.
struct AnyCollection:
    fn __init__(out self):
        pass
    fn __init__(out self, value: AnyType):
        pass
    fn __init__(out self, owned *values: Int, __list_literal__: ()):
        pass
    fn __init__(out self, owned *values: Int, __set_literal__: ()):
        pass
    fn __init__(out self, keys: IntList, values: IntList, __dict_literal__: ()):
        pass

# CHECK-LABEL: lit.fn @"test_any_collection
fn test_any_collection():
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %a){{.*}}__dict_literal__
    var a : AnyCollection = {}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %b){{.*}}__set_literal__
    var b : AnyCollection = {1}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %c){{.*}}__set_literal__
    var c : AnyCollection = {1, 2}
    # CHECK: lit.call {{.*}}@AnyCollection::@"__init__{{.*}}({{.*}}, %d){{.*}}__dict_literal__
    var d : AnyCollection = {1: 2}
