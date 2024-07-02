# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# ===----------------------------------------------------------------------=== #
# elif
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.func @"test_elif_chain
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[TEST_A:%.*]] = lit.call @{{.*}}::@Bool::@"__mlir_i1__{{.*}}"(%a)
# CHECK-NEXT:      hlcf.elif.yield [[TEST_A]] : i1
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_a = lit.var.decl "inside_a"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[B_EQ:%.*]] = lit.call @{{.*}}::@"__eq__{{.*}}"(%b, %d) : !lit.signature<("lhs": !Int, "rhs": !Int) -> !Bool>
# CHECK-NEXT:      [[TEST_B:%.*]] = lit.call @{{.*}}::@"__mlir_i1__{{.*}}"([[B_EQ]]) : !lit.signature<("self": !Bool) -> i1>
# CHECK-NEXT:      hlcf.elif.yield [[TEST_B]] : i1
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_b = lit.var.decl "inside_b"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[TEST_C:%.*]] = lit.call @{{.*}}::@"__mlir_i1__{{.*}}"(%c)
# CHECK-NEXT:      hlcf.elif.yield [[TEST_C]] : i1
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_c = lit.var.decl "inside_c"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      %inside_else = lit.var.decl "inside_else"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
fn test_elif_chain(a: Bool, b: Int, c: Bool, d: Int) -> Bool:
    if a:
        var inside_a: Int
    elif b == d:
        var inside_b: Int
    elif c:
        var inside_c: Int
    else:
        var inside_else: Int
    return a


# CHECK-LABEL: lit.func @"test_constant
fn test_constant(a: Bool) -> Bool:
    # CHECK: [[FOUR:%.*]] = kgen.param.constant{{.*}}4
    # CHECK-NEXT: store [[FOUR]], %z
    var z: Int = 4

    # Walrus operator in if's.
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: store [[FIVE]], %z
    # CHECK-NEXT: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}([[FIVE]])
    # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
    # CHECK-NEXT: hlcf.elif.yield [[I1]] : i1
    if z := 5:
        return a

    return a


# CHECK-LABEL: lit.func @"test_if_nested
fn test_if_nested(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT:   %0 = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%a)
    # CHECK-NEXT:   hlcf.elif.yield %0 : i1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   %inside_a = lit.var.decl "inside_a"
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.elif {
    # CHECK-NEXT:     [[TEST_B:%.*]] = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%b)
    # CHECK-NEXT:     hlcf.elif.yield [[TEST_B]] : i1
    # CHECK-NEXT:   } then {
    # CHECK-NEXT:     %inside_b = lit.var.decl "inside_b"  var : !lit.ref<!Int, mut *"inside_b`1">
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.elif {
    # CHECK-NEXT:       [[TEST_C:%.*]] = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%c)
    # CHECK-NEXT:       hlcf.elif.yield [[TEST_C]] : i1
    # CHECK-NEXT:     } then {
    # CHECK-NEXT:       %inside_c = lit.var.decl "inside_c"
    # CHECK-NEXT:       hlcf.yield
    # CHECK-NEXT:     } else {
    # CHECK-NEXT:       %inside_else = lit.var.decl "inside_else"
    # CHECK-NEXT:       hlcf.yield
    # CHECK-NEXT:     }
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: }
    if a:
        var inside_a: Int
    else:
        if b:
            var inside_b: Int
        else:
            if c:
                var inside_c: Int
            else:
                var inside_else: Int
    return a


# CHECK-LABEL: lit.func @"param_if{{.*}}"<a: i1, b: !Bool>()
fn param_if[a: __mlir_type.i1, b: Bool]():
    # CHECK: kgen.param.if <a> {
    @parameter
    if a:
        # CHECK: lit.var.decl "inside_1" var
        var inside_1: Int
    # CHECK: } else {
    # CHECK:     kgen.param.if <apply{{.*}}{{.*}}Bool::@"__mlir_i1__{{.*}}b)> {
    elif b:
        # CHECK:     lit.var.decl "inside_2" var
        var inside_2: Int
    # CHECK:     kgen.param.yield
    # CHECK:   }
    # CHECK:   kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.func @"param_if_andor_i1{{.*}}"<a: i1, b: i1>()
fn param_if_andor_i1[a: __mlir_type.i1, b: __mlir_type.i1]():
    # CHECK: kgen.param.if <cond(a, b, a)>
    @parameter
    if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: } else {
    # CHECK: kgen.param.if <cond(a, a, b)>
    elif a or b:
        # CHECK:   lit.var.decl "w" var
        var w: Int


# CHECK-LABEL: lit.func @"param_if_and{{.*}}"<a: !Bool, b: !Bool>()
fn param_if_and[a: Bool, b: Bool]():
    # CHECK: kgen.param.if <apply(
    # CHECK-SAME: !lit.signature<("self": !Bool) -> i1> {{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)", cond(
    # CHECK-SAME: apply({{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)", a), b, a))> {
    @parameter
    if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: }


# [Mojo] Can't have try inside else branch
# https://github.com/modularml/modular/issues/25305
# CHECK-LABEL: lit.func @"if_try
fn if_try(p: Bool):
    # CHECK:      hlcf.elif {
    # CHECK-NEXT:   [[TEST_P:%*.]] = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%p)
    # CHECK-NEXT:   hlcf.elif.yield [[TEST_P]] : i1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   %e = lit.var.decl {{.*}} !lit.ref<!Error,
    # CHECK-NEXT:   lit.try %e : {{.*}} {
    # CHECK-NEXT:     %b = lit.var.decl "b"  var
    # CHECK-NEXT:     kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT:     lit.ref.store
    # CHECK-NEXT:     lit.try.yield
    # CHECK-NEXT:   } except {
    # CHECK-NEXT:     %c = lit.var.decl "c"
    # CHECK-NEXT:     kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT:     lit.ref.store
    # CHECK-NEXT:     lit.try.yield
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     lit.try.yield
    # CHECK-NEXT:   } finally {
    # CHECK-NEXT:     lit.try.yield
    # CHECK-NEXT    }
    # CHECK:        hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %d = lit.var.decl "d"
    # CHECK-NEXT:   kgen.param.constant: !Int = <{3}>
    # CHECK-NEXT:   lit.ref.store
    # CHECK-NEXT:   hlcf.yield
    if p:
        try:
            var b = 1
        except e:
            var c = 2
    else:
        var d = 3


# CHECK-LABEL: lit.func @"testCondAsArg
fn testCondAsArg(exit_early: __mlir_type.i1):
    # CHECK: hlcf.elif
    if exit_early:
        return


# CHECK-LABEL: lit.func @"constantTrue
fn constantTrue(cond: Bool, x: Int, y: Int) -> Int:
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT:  %0 = kgen.param.constant: i1 = <1>
    # CHECK-NEXT:  hlcf.elif.yield %0 : i1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    if `True`:
        return x
    return y


# CHECK-LABEL: lit.func @"constantFalse
fn constantFalse(cond: Bool, x: Int, y: Int) -> Int:
    # CHECK:      hlcf.elif {
    # CHECK-NEXT:   %0 = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%cond)
    # CHECK-NEXT:   hlcf.elif.yield %0 : i1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %0 = kgen.param.constant: i1 = <0>
    # CHECK-NEXT:   hlcf.elif.yield %0 : i1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    if cond:
        return x
    elif `False`:
        return y
    else:
        return x


##===----------------------------------------------------------------------===##
# While
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"test_while
# CHECK:       %inside_a = lit.var.decl "inside_a" var
# CHECK:       %inside_b = lit.var.decl "inside_b" var
# CHECK:       %inside_else = lit.var.decl "inside_else" var
# CHECK:       lit.loop cond {
# CHECK:         [[V0:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)"(%a)
# CHECK:         lit.loop.condition [[V0]] : i1
# CHECK:       } body {
# CHECK-NEXT:    kgen.param.constant: {{.*}} = <{0}>
# CHECK-NEXT:    lit.ref.store {{.+}}, %inside_a
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[TEST_B:%*.]] = lit.call @{{.*}}@"__mlir_i1__{{.*}}"(%b)
# CHECK-NEXT:      hlcf.elif.yield [[TEST_B]] : i1
# CHECK-NEXT:    } then {
# CHECK-NEXT:      kgen.param.constant: !Int = <{1}>
# CHECK-NEXT:      lit.ref.store
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
# CHECK-NEXT:    lit.loop.continue
# CHECK-NEXT:  } else {
# CHECK-NEXT:    kgen.param.constant: !Int = <{2}>
# CHECK-NEXT:    lit.ref.store
# CHECK-NEXT:    lit.loop.yield
fn test_while(a: Bool, b: Bool) -> Bool:
    var inside_a: Int
    var inside_b: Int
    var inside_else: Int
    while a:
        inside_a = 0
        if b:
            inside_b = 1
    else:
        inside_else = 2
    return a


# CHECK-LABEL: lit.func @"test_else_outside_while
def test_else_outside_while(a: Bool, b: Bool) -> Bool:
    # CHECK: hlcf.elif {
    # CHECK:   hlcf.elif.yield
    # CHECK: } then {
    if b:
        # CHECK: lit.loop cond {
        # CHECK:   [[V1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)"(%a)
        # CHECK:   lit.loop.condition [[V1]] : i1
        # CHECK: } body {
        while a:
            # CHECK: lit.ref.store {{.+}}, %inside_a
            inside_a = 0
            # CHECK: lit.loop.continue
            # CHECK: } else {
            # CHECK:   lit.loop.yield
            # CHECK: }
    # CHECK: } else {
    else:
        # CHECK: lit.ref.store {{.+}}, %inside_else
        inside_else = 2
    # CHECK: }
    # CHECK: lit.return
    return a


# CHECK-LABEL: lit.func @"test_break_continue_inside_while
def test_break_continue_inside_while(a: Bool) -> Bool:
    # CHECK: lit.loop cond {
    # CHECK:   [[V1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)"(%a)
    # CHECK:   lit.loop.condition [[V1]] : i1
    # CHECK: } body {
    while a:
        # CHECK:      hlcf.elif {
        # CHECK-NEXT:   lit.call {{.*}}__mlir_i1__
        # CHECK-NEXT:   hlcf.elif.yield
        # CHECK-NEXT: } then {
        if a:
            # CHECK-NEXT:   lit.break
            break
            # CHECK:   lit.ref.store
            # CHECK-NEXT:   hlcf.yield
            c = 1
        else:
            # CHECK-NEXT: } else {
            # CHECK-NEXT:   lit.continue
            continue
            # CHECK-NEXT:   hlcf.yield
        # CHECK: lit.loop.continue
    return a


# CHECK-LABEL: lit.func @"test_early_return
def test_early_return():
    # CHECK:  hlcf.elif {
    # CHECK:    hlcf.elif.yield
    # CHECK:  } then {
    var a: Bool
    if a:
        # CHECK: lit.return
        return
        # CHECK: lit.ref.store
        b = 2
        # CHECK-NEXT: hlcf.yield
    # CHECK: else
    # CHECK-NEXT: yield
    # CHECK: lit.return
    return
    # CHECK: lit.ref.store
    c = 3
    # CHECK: lit.return
    return
    # CHECK: lit.end_func


##===----------------------------------------------------------------------===##
# pass
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"pass
def `pass`():
    pass
    # CHECK: lit.end_func


##===----------------------------------------------------------------------===##
# return
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"return_impl_convert
fn return_impl_convert() -> Int:
    # CHECK: %0 = kgen{{.*}}{4}
    # CHECK: lit.return %0
    return 4  # Implicit conversion from literal to Int


# CHECK-LABEL: lit.func @"return_new_line
fn return_new_line() -> Int:
    # CHECK: %0 = kgen{{.*}}{17}
    return
        17  # Weird indentation should be fine


# CHECK-LABEL: lit.func @"return_impl_convert_raises
fn return_impl_convert_raises() raises -> Int:
    # CHECK: %0 = kgen{{.*}}{4}
    # CHECK-NEXT: lit.ref.store %0, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: return [[FALSE]]
    return 4  # Implicit conversion from literal to Int


##===----------------------------------------------------------------------===##
# while
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_simple
# CHECK:       lit.loop cond {
# CHECK:         [[V0:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__({{.*}}Bool)"(%a)
# CHECK:         lit.loop.condition [[V0]] : i1
# CHECK:       } body {
# CHECK-NEXT:     lit.loop.continue
# CHECK-NEXT:  } else {
# CHECK-NEXT:    lit.loop.yield
# CHECK-NEXT:  }
# CHECK-NEXT:   %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:   lit.return %none :  !kgen.none
# CHECK-NEXT:   lit.end_func
# CHECK-NEXT: }
fn test_simple(a: Bool):
    while a:
        pass


##===----------------------------------------------------------------------===##
# for
##===----------------------------------------------------------------------===##

# This iterator returns elements by value.
struct ValueIter:
    fn __init__(inout self): pass
    fn __next__(inout self) -> Int: return 0
    fn __len__(self) -> Int: return 0

struct ListValueIter:
    fn __init__(inout self): pass
    fn __iter__(self) -> ValueIter: return ValueIter()

fn use(value: Int): pass


# CHECK-LABEL: lit.func @"for_range_loop
fn for_range_loop():
    var value_iter_list = ListValueIter()

    # CHECK: %$RANGE = lit.var.decl "$RANGE" synth
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %value_iter_list
    # CHECK-NEXT: [[ITER:%.*]] = lit.call @{{.*}}__iter__{{.*}}([[IMMREF]], %$RANGE)
    for item in value_iter_list:
        # CHECK: lit.loop cond {
        # CHECK:   [[IMMREF:%.*]] = lit.ref.immut %$RANGE
        # CHECK:   [[LENGTH:%.*]] = lit.call {{.*}}__len__{{.*}}([[IMMREF]])
        # CHECK:   [[INDEX:%.*]] = lit.call {{.*}}__index__{{.*}}([[LENGTH]])
        # CHECK:   [[MLIR_INDEX:%.*]] = lit.call {{.*}}__mlir_index__{{.*}}([[INDEX]])
        # CHECK:   [[COND:%.*]] = index.cmp sgt([[MLIR_INDEX]], %idx0)
        # CHECK:   lit.loop.condition [[COND]]
        # CHECK: } body {
        # CHECK:   lit.loop.continue
        # CHECK: } else {
        # CHECK-NEXT: lit.loop.yield
        # CHECK: }
        use(item)


# This iterator returns elements by reference, using the mutability and lifetime
# of the list.
struct RefIter[list_mutability: Bool, //,
               list_lifetime: AnyLifetime[list_mutability].type]:
    fn __init__(inout self): pass
    fn __next__(inout self) -> ref [list_lifetime] Int: pass
    fn __len__(self) -> Int: return 0

struct ListWithRefIter:
    fn __init__(inout self): pass
    fn __iter__(ref [_] self: Self) -> RefIter[__lifetime_of(self)]:
        return RefIter[__lifetime_of(self)]()

# CHECK-LABEL: lit.func @"for_range_ref_loop
fn for_range_ref_loop(imm_list_ref_iter: ListWithRefIter,
                      inout mut_list_ref_iter: ListWithRefIter):

    # CHECK: [[ITEM:%.*]] = lit.var.decl "item"
    # CHECK-NEXT: %$RANGE = lit.var.decl "$RANGE" synth
    # CHECK-NEXT: [[ITER:%.*]] = lit.call @{{.*}}__iter__{{.*}}(%mut_list_ref_iter, %$RANGE)
    for item in mut_list_ref_iter:
        # CHECK: lit.loop cond {
        # CHECK:   [[IMMREF:%.*]] = lit.ref.immut %$RANGE
        # CHECK:   [[LENGTH:%.*]] = lit.call {{.*}}__len__{{.*}}([[IMMREF]])
        # CHECK:   [[INDEX:%.*]] = lit.call {{.*}}__index__{{.*}}([[LENGTH]])
        # CHECK:   [[MLIR_INDEX:%.*]] = lit.call {{.*}}__mlir_index__{{.*}}([[INDEX]])
        # CHECK:   [[COND:%.*]] = index.cmp sgt([[MLIR_INDEX]], %idx0)
        # CHECK:   lit.loop.condition [[COND]]
        # CHECK: } body {
        # CHECK:   [[ELTREF:%.*]] = lit.call {{.*}}RefIter::@"__next__{{.*}}(%$RANGE)

        # The int value from this element is captured into item, not the reference.
        # CHECK: [[ELTVAL:%.*]] = lit.ref.load [[ELTREF]]
        # CHECK: lit.ref.store [[ELTVAL]], %item

        # CHECK: [[ELTVAL:%.*]] = lit.ref.load %item
        # CHECK: lit.call {{.*}}use{{.*}}([[ELTVAL]])
        use(item)

        # The iterator is a mutable var, so this changes the value of the var
        # not the list element.
        item = 4
        # CHECK: lit.ref.store {{.*}}, %item

        # CHECK:   lit.loop.continue
        # CHECK: } else {
        # CHECK-NEXT:   lit.loop.yield
        # CHECK: }


@value
struct IterRange:
    var value: Int

    fn __iter__(self) -> Self:
        return self

    fn __len__(self) -> Int:
        return self.value

    fn __next__(inout self) -> Int:
        return self.value


# CHECK-LABEL: @"induction_var_scope()"
fn induction_var_scope():
    # CHECK: "item"
    # CHECK: lit.loop
    for item in IterRange(0):
        # CHECK: lit.ref.load %item
        # CHECK: lit.ref.store %{{.*}}, %g
        var g = item
    for item in IterRange(0):
        # CHECK: lit.ref.load %item
        var g = item

struct MyType:
    pass

fn use(value: MyType):
    pass

# CHECK-LABEL: lit.func @"parameter_for
# CHECK-SAME: [mut [[LT:.*]]]<a: !Int>(%value: !lit.ref<!MyType, mut [[LT]]>
fn parameter_for[a: Int](owned value: MyType):
    # CHECK-NEXT: kgen.param.for [[i:.*]]: !Int in :!IterRange apply
    # CHECK-SAME: iter :{{.*}}parameter_for_generator{{.*}}<:!IntIterable #IterRange
    @parameter
    for i in IterRange(a):
        # CHECK: [[IMM:%.*]] = lit.ref.immut %value
        # CHECK: use{{.*}}[muttoimm [[LT]]]([[IMM]])
        use(value)
        # CHECK: kgen.param.for.continue

##===----------------------------------------------------------------------===##

# TODO(Issue #6139)

# struct Iterable:

# fn test_for(iterable: Iterable):
#  var result = 0
#  for i in iterable:
#    result += i
