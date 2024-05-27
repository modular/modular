# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo -verify-diagnostics | FileCheck %s

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
# While
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
# For
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
        # CHECK-NEXT:   lit.loop.yield
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
    fn __iter__(self: Reference[Self, _, _]) -> RefIter[self.lifetime]:
        return RefIter[self.lifetime]()

# CHECK-LABEL: lit.func @"for_range_ref_loop
fn for_range_ref_loop(imm_list_ref_iter: ListWithRefIter,
                      inout mut_list_ref_iter: ListWithRefIter):

    # CHECK: [[ITEM:%.*]] = lit.var.decl "item"
    # CHECK-NEXT: %$RANGE = lit.var.decl "$RANGE" synth
    # CHECK-NEXT: %anonymous2A = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}Reference::@"__init__{{.*}}(%anonymous2A, %mut_list_ref_iter)
    # CHECK-NEXT: [[MUTREF:%.*]] = lit.ref.load %anonymous2A
    # CHECK-NEXT: [[ITER:%.*]] = lit.call @{{.*}}__iter__{{.*}}([[MUTREF]], %$RANGE)
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


# CHECK-LABEL: @"induction_var_scope()"
fn induction_var_scope():
    # CHECK: "item"
    # CHECK: lit.loop
    for item in range(0):
        # CHECK: lit.ref.load %item
        # CHECK: lit.ref.store %{{.*}}, %g
        var g = item
    for item in range(0):
        # CHECK: lit.ref.load %item
        var g = item

struct MyType:
    pass

fn use(value: MyType):
    pass

# CHECK-LABEL: lit.func @"parameter_for
# CHECK-SAME: [mut [[LT:.*]]]<a: !Int>(%value: !lit.ref<!MyType, mut [[LT]]>
fn parameter_for[a: Int](owned value: MyType):
    # CHECK-NEXT: kgen.param.for [[i:.*]]: !Int in :!ZeroStartingRange apply
    # CHECK-SAME: iter :{{.*}}parameter_for_generator{{.*}}<:!IntIterable #ZeroStartingRange
    @parameter
    for i in range(a):
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
