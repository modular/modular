# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# elif
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"test_elif_chain
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[TEST_A_SB:%.*]] = lit.call {{.*}}::@Bool::@"__mlir_bool__{{.*}}"(%a)
# CHECK-NEXT:      [[TEST_A:%.*]] = pop.cast_to_builtin [[TEST_A_SB]]
# CHECK-NEXT:      hlcf.elif.yield [[TEST_A]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_a = lit.var.decl "inside_a"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[B_EQ:%.*]] = lit.call {{.*}}::@"__eq__{{.*}}"(%b, %d) : !lit.generator<("lhs": !Int, "rhs": !Int) -> !Bool>
# CHECK-NEXT:      [[TEST_B_SB:%.*]] = lit.call {{.*}}::@"__mlir_bool__{{.*}}"([[B_EQ]]) : !lit.generator<("self": !Bool) -> !kgen.scalar<bool>>
# CHECK-NEXT:      [[TEST_B:%.*]] = pop.cast_to_builtin [[TEST_B_SB]]
# CHECK-NEXT:      hlcf.elif.yield [[TEST_B]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_b = lit.var.decl "inside_b"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[TEST_C_SB:%.*]] = lit.call {{.*}}::@"__mlir_bool__{{.*}}"(%c)
# CHECK-NEXT:      [[TEST_C:%.*]] = pop.cast_to_builtin [[TEST_C_SB]]
# CHECK-NEXT:      hlcf.elif.yield [[TEST_C]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      %inside_c = lit.var.decl "inside_c"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      %inside_else = lit.var.decl "inside_else"
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def test_elif_chain(a: Bool, b: Int, c: Bool, d: Int) -> Bool:
    if a:
        var inside_a: Int
    elif b == d:
        var inside_b: Int
    elif c:
        var inside_c: Int
    else:
        var inside_else: Int
    return a


# CHECK-LABEL: lit.fn @"test_constant
def test_constant(a: Bool) -> Bool:
    # CHECK: [[FOUR:%.*]] = kgen.param.constant{{.*}}4
    # CHECK-NEXT: store [[FOUR]], %z
    var z: Int = 4

    # Walrus operator in if's.
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: store [[FIVE]], %z
    # CHECK-NEXT: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}([[FIVE]])
    # CHECK-NEXT: [[SB:%.*]] = lit.call {{.*}}__mlir_bool__{{.*}}([[BOOL]])
    # CHECK-NEXT: [[I1:%.*]] = pop.cast_to_builtin [[SB]]
    # CHECK-NEXT: hlcf.elif.yield [[I1]]
    if z := 5:
        return a

    return a


# CHECK-LABEL: lit.fn @"test_if_nested
def test_if_nested(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT:   %0 = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%a)
    # CHECK-NEXT:   %1 = pop.cast_to_builtin %0
    # CHECK-NEXT:   hlcf.elif.yield %1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   %inside_a = lit.var.decl "inside_a"
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.elif {
    # CHECK-NEXT:     [[TEST_B_SB:%.*]] = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%b)
    # CHECK-NEXT:     [[TEST_B:%.*]] = pop.cast_to_builtin [[TEST_B_SB]]
    # CHECK-NEXT:     hlcf.elif.yield [[TEST_B]]
    # CHECK-NEXT:   } then {
    # CHECK-NEXT:     %inside_b = lit.var.decl "inside_b"  var : !lit.ref<!Int, mut *"inside_b`1">
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.elif {
    # CHECK-NEXT:       [[TEST_C_SB:%.*]] = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%c)
    # CHECK-NEXT:       [[TEST_C:%.*]] = pop.cast_to_builtin [[TEST_C_SB]]
    # CHECK-NEXT:       hlcf.elif.yield [[TEST_C]]
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


# [Mojo] Can't have try inside else branch
# https://github.com/modularml/modular/issues/25305
# CHECK-LABEL: lit.fn @"if_try
def if_try(p: Bool):
    # CHECK:      hlcf.elif {
    # CHECK-NEXT:   [[TEST_P_SB:%*.]] = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%p)
    # CHECK-NEXT:   [[TEST_P:%.*]] = pop.cast_to_builtin [[TEST_P_SB]]
    # CHECK-NEXT:   hlcf.elif.yield [[TEST_P]]
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


# CHECK-LABEL: lit.fn @"testCondAsArg
def testCondAsArg(exit_early: __mlir_type.i1):
    # CHECK: hlcf.elif
    if exit_early:
        return


# CHECK-LABEL: lit.fn @"constantTrue
def constantTrue(cond: Bool, x: Int, y: Int) -> Int:
    # CHECK-NEXT: hlcf.elif {
    # CHECK-NEXT:  %0 = kgen.param.constant: i1 = <{{.*}}1{{.*}}>
    # CHECK-NEXT:  hlcf.elif.yield %0
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.unreachable
    if True:
        return x
    return y


# CHECK-LABEL: lit.fn @"constantFalse
def constantFalse(cond: Bool, x: Int, y: Int) -> Int:
    # CHECK:      hlcf.elif {
    # CHECK-NEXT:   %0 = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%cond)
    # CHECK-NEXT:   %1 = pop.cast_to_builtin %0
    # CHECK-NEXT:   hlcf.elif.yield %1
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %0 = kgen.param.constant: i1 = <{{.*}}0{{.*}}>
    # CHECK-NEXT:   hlcf.elif.yield %0
    # CHECK-NEXT: } then {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.return %x : !Int
    # CHECK-NEXT:   hlcf.yield
    if cond:
        return x
    elif False:
        return y
    else:
        return x


##===----------------------------------------------------------------------===##
# While
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"test_while
# CHECK:       %inside_a = lit.var.decl "inside_a" var
# CHECK:       %inside_b = lit.var.decl "inside_b" var
# CHECK:       %inside_else = lit.var.decl "inside_else" var
# CHECK:       lit.loop {
# CHECK:         [[V0_SB:%.*]] = lit.call {{.*}}@Bool::@"__mlir_bool__({{.*}}Bool)"(%a)
# CHECK:         [[V0:%.*]] = pop.cast_to_builtin [[V0_SB]]
# CHECK:         hlcf.if [[V0]] {
# CHECK-NEXT:        hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:        lit.loop.break.else
# CHECK-NEXT:    }
# CHECK-NEXT:    kgen.param.constant: {{.*}} = <{0}>
# CHECK-NEXT:    lit.ref.store {{.+}}, %inside_a
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[TEST_B_SB:%*.]] = lit.call {{.*}}@"__mlir_bool__{{.*}}"(%b)
# CHECK-NEXT:      [[TEST_B:%.*]] = pop.cast_to_builtin [[TEST_B_SB]]
# CHECK-NEXT:      hlcf.elif.yield [[TEST_B]]
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
def test_while(a: Bool, b: Bool) -> Bool:
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


# CHECK-LABEL: lit.fn @"test_else_outside_while
def test_else_outside_while(a: Bool, b: Bool) raises -> Bool:
    # CHECK: hlcf.elif {
    # CHECK:   hlcf.elif.yield
    # CHECK: } then {
    if b:
        # CHECK: lit.loop {
        # CHECK:   [[V1_SB:%.*]] = lit.call {{.*}}@Bool::@"__mlir_bool__({{.*}}Bool)"(%a)
        # CHECK:   [[V1:%.*]] = pop.cast_to_builtin [[V1_SB]]
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


# CHECK-LABEL: lit.fn @"test_break_continue_inside_while
def test_break_continue_inside_while(a: Bool) raises -> Bool:
    # CHECK: lit.loop {
    # CHECK:   [[V1_SB:%.*]] = lit.call {{.*}}@Bool::@"__mlir_bool__({{.*}}Bool)"(%a)
    # CHECK:   [[V1:%.*]] = pop.cast_to_builtin [[V1_SB]]
    while a:
        # CHECK:      hlcf.elif {
        # CHECK-NEXT:   lit.call {{.*}}__mlir_bool__
        # CHECK-NEXT:   pop.cast_to_builtin
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


# CHECK-LABEL: lit.fn @"test_early_return
def test_early_return() raises:
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
    # CHECK: lit.end_fn


##===----------------------------------------------------------------------===##
# pass
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"pass
def `pass`() raises:
    pass
    # CHECK: lit.end_fn


##===----------------------------------------------------------------------===##
# return
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"return_impl_convert
def return_impl_convert() -> Int:
    # CHECK: %0 = kgen{{.*}}{4}
    # CHECK: lit.return %0
    return 4  # Implicit conversion from literal to Int


# CHECK-LABEL: lit.fn @"return_new_line
def return_new_line() -> Int:
    # CHECK: %0 = kgen{{.*}}{17}
    return
        17  # Weird indentation should be fine


# CHECK-LABEL: lit.fn @"return_impl_convert_raises
def return_impl_convert_raises() raises -> Int:
    # CHECK: %0 = kgen{{.*}}{4}
    # CHECK-NEXT: lit.ref.store %0, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: return [[FALSE]]
    return 4  # Implicit conversion from literal to Int


##===----------------------------------------------------------------------===##
# while
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"test_simple
# CHECK:       lit.loop {
# CHECK:         [[V0_SB:%.*]] = lit.call {{.*}}@Bool::@"__mlir_bool__({{.*}}Bool)"(%a)
# CHECK:         [[V0:%.*]] = pop.cast_to_builtin [[V0_SB]]
# CHECK:         lit.loop.continue
# CHECK-NEXT:  } else {
# CHECK-NEXT:    lit.loop.yield
# CHECK-NEXT:  }
# CHECK-NEXT:   %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:   lit.return %none :  !kgen.none
# CHECK-NEXT:   lit.end_fn
# CHECK-NEXT: }
def test_simple(a: Bool):
    while a:
        pass


##===----------------------------------------------------------------------===##
# for
##===----------------------------------------------------------------------===##

# This iterator returns elements by value.
struct ValueIter:
    def __init__(out self): pass
    def __next__(mut self) raises StopIteration -> Int: return 0

struct ListValueIter:
    def __init__(out self): pass
    def __iter__(self) -> ValueIter: return ValueIter()

def use(value: Int): pass


# CHECK-LABEL: lit.fn @"for_range_loop
def for_range_loop():
    var value_iter_list = ListValueIter()

    # CHECK: %$ITER = lit.var.decl "$ITER" synth
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %value_iter_list
    # CHECK-NEXT: [[ITER:%.*]] = lit.call {{.*}}__iter__{{.*}}([[IMMREF]], %$ITER)
    for item in value_iter_list:
        # CHECK: lit.loop {
        # CHECK-NEXT:  [[ANON:%.*]] = lit.var.decl "anonymous*"
        # CHECK-NEXT:   %__call_error_tmp__ = lit.var.decl
        # CHECK-NEXT:   lit.try %__call_error_tmp__
        # CHECK-NEXT:    lit.call {{.*}}__next__{{.*}}(%$ITER, %__call_error_tmp__, [[ANON]])
        # CHECK-NEXT:    lit.try.yield
        # CHECK-NEXT:        } except {
        # CHECK-NEXT:  lit.loop.break.else
        # CHECK: } {suppressWarnings = true}
        # CHECK-NEXT:  %item = lit.var.decl
        # CHECK: [[TMP:%.*]] = lit.ref.load %item
        # CHECK-NEXT: [[TMPREF:%.*]] = lit.ref.load [[TMP]]
        # CHECK-NEXT: lit.call{{.*}}use{{.*}}([[TMPREF]])
        # CHECK-NEXT: lit.loop.continue
        use(item)
    else:
        pass
        # CHECK: } else {
        # CHECK:   lit.loop.yield
        # CHECK: }


# This iterator returns elements by reference, using the mutability and origin
# of the list.
struct RefIter[list_mutability: Bool, //,
               list_origin: Origin[mut=list_mutability]]:
    def __init__(out self): pass
    def __next__(mut self) raises StopIteration -> ref [Self.list_origin] Int: pass

struct ListWithRefIter:
    def __init__(out self): pass
    def __iter__(ref self) -> RefIter[origin_of(self)]:
        return RefIter[origin_of(self)]()

# CHECK-LABEL: lit.fn @"for_range_ref_loop
def for_range_ref_loop(imm_list_ref_iter: ListWithRefIter,
                      mut mut_list_ref_iter: ListWithRefIter):

    # CHECK-NEXT: %$ITER = lit.var.decl "$ITER" synth
    # CHECK-NEXT: [[ITER:%.*]] = lit.call {{.*}}__iter__{{.*}}(%mut_list_ref_iter, %$ITER)
    for ref item in mut_list_ref_iter:
        # CHECK: lit.loop {
        # CHECK-NEXT: %__ref_result_tmp__ = lit.var.decl
        # CHECK-NEXT: %__call_error_tmp__ = lit.var.decl
        # CHECK:   lit.call {{.*}}RefIter::@"__next__{{.*}}(%$ITER, %__call_error_tmp__, %__ref_result_tmp__)

        # CHECK:   %item = lit.var.decl "item" ref

        # The Index value from this element is captured into item, not the reference.
        # CHECK: [[ELTREF:%.*]] = lit.ref.load %item
        # CHECK: [[ELTVAL:%.*]] = lit.ref.load [[ELTREF]]
        # CHECK: lit.call {{.*}}use{{.*}}([[ELTVAL]])
        use(item)

        # The iterator is a mutable var, so this changes the value of the var
        # not the list element.
        item = 4
        # CHECK: [[ELTREF:%.*]] = lit.ref.load %item
        # CHECK: lit.ref.store {{.*}}, [[ELTREF]]

        # CHECK:   lit.loop.continue
        # CHECK: } else {
        # CHECK-NEXT:   lit.loop.yield
        # CHECK: }


@fieldwise_init
struct IterRange(Iterator, ImplicitlyCopyable):
    comptime Element = Int

    var value: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.value <= 0:
            raise StopIteration()
        return self.value


# CHECK-LABEL: @"induction_var_scope()"
def induction_var_scope():
    # CHECK: lit.loop {
    for item in IterRange(0):
        # CHECK: __next__
        # CHECK: %item = lit.var.decl "item"
        # CHECK: lit.ref.load %item
        # CHECK: lit.ref.store %{{.*}}, %g
        var g = item

    # CHECK: lit.loop {
    for (var item) in IterRange(0):
        # CHECK: __next__
        # CHECK: %item = lit.var.decl "item"
        # CHECK: lit.ref.load %item
        var g = item

# CHECK-LABEL: @"induction_var_scope_def()"
def induction_var_scope_def() raises:
    # CHECK: lit.loop {
    for item in IterRange(0):
        # CHECK: __next__
        # CHECK: %item = lit.var.decl "item"
        # CHECK: lit.ref.load %item
        # CHECK: lit.ref.store {{.*}}, %g
        var g = item

    # CHECK: lit.loop {
    for item in IterRange(0):
        # CHECK: __next__
        # CHECK: %item = lit.var.decl "item"
        # CHECK: lit.ref.load %item
        var g = item


# CHECK-LABEL: lit.fn @"weird_llvm_dialect_op
# https://github.com/modular/mojo/issues/3805
def weird_llvm_dialect_op():
    # CHECK: %0 = llvm.mlir.zero : !llvm.ptr
   x = __mlir_op.`llvm.mlir.zero`[_type = __mlir_type.`!llvm.ptr<0>`]()


##===----------------------------------------------------------------------===##

# TODO(Issue #6139)

# struct Iterable:

# def test_for(iterable: Iterable):
#  var result = 0
#  for i in iterable:
#    result += i

