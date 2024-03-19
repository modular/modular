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
    # CHECK: %1 = kgen.variant.create %0
    # CHECK: lit.return %1 : !kgen.variant<!Error, !Int>
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


struct my_iter:
    fn __init__(inout self): pass
    fn __next__(inout self: my_iter) -> Int: return 0
    fn __len__(self: my_iter) -> Int: return 0


struct MyList:
    fn __init__(inout self): pass
    fn __iter__(self) -> my_iter: return my_iter()


# CHECK-LABEL: lit.func @"for_range_loop()"
fn for_range_loop():
    var my_list = MyList()

    # CHECK: %$RANGE = lit.var.decl "$RANGE" synth
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %my_list
    # CHECK-NEXT: [[ITER:%.*]] = lit.call @{{.*}}__iter__{{.*}}([[IMMREF]], %$RANGE)
    for item in my_list:
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
        pass


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

# CHECK-LABEL: lit.func @"unroll_for()"
fn unroll_for():
    @unroll
    for i in range(1, 9, 2):
        _ = i
        @unroll
        for j in range (1, 4):
            _ = i + j
    # CHECK: } {unrollLevel = #hlcf<unroll_level full>}
    # CHECK: } {unrollLevel = #hlcf<unroll_level full>}

    @unroll(2)
    for j in range (1, 4):
        _ = j
    # CHECK: } {unrollLevel = #hlcf<unroll_level 2>}

# CHECK-LABEL: lit.func @"unroll_while()"
fn unroll_while():
  var i = 1
  @unroll
  while i < 4:
      _ = i
  # CHECK: } {unrollLevel = #hlcf<unroll_level full>}

fn unroll_factor_parameter():
  alias a = 1
  alias b = 1
  var i = 1
  @unroll(a+b)
  while i < 4:
      _ = i
  # CHECK: } {unrollLevel = #kgen.param.expr<apply, #kgen.symbol.constant

##===----------------------------------------------------------------------===##

# TODO(Issue #6139)

# struct Iterable:

# fn test_for(iterable: Iterable):
#  var result = 0
#  for i in iterable:
#    result += i
