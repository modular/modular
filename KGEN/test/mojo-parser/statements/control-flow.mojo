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
    # CHECK: %0 = kgen{{.*}}= 4}
    # CHECK: lit.return %0
    return 4  # Implicit conversion from literal to Int


# CHECK-LABEL: lit.func @"return_new_line
fn return_new_line() -> Int:
    # CHECK: %0 = kgen{{.*}}= 17}
    return
        17  # Weird indentation should be fine


# CHECK-LABEL: lit.func @"return_impl_convert_raises
fn return_impl_convert_raises() raises -> Int:
    # CHECK: %0 = kgen{{.*}}= 4}
    # CHECK: %1 = kgen.variant.create %0
    # CHECK: lit.return %1 : !kgen.variant<!Error, !Int>
    return 4  # Implicit conversion from literal to Int


##===----------------------------------------------------------------------===##
# If
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_if
fn test_if(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK:          hlcf.if
    if a:
        # CHECK-NEXT: %inside_a = lit.varlet.decl "inside_a" var
        var inside_a: Int
    # CHECK:          } else {
    # CHECK:            hlcf.if
    elif b:
        # CHECK-NEXT: %inside_b = lit.varlet.decl "inside_b" var
        var inside_b: Int
    # CHECK:            } else {
    # CHECK:              hlcf.if
    elif c:
        # CHECK-NEXT: %inside_c = lit.varlet.decl "inside_c" var
        var inside_c: Int
    # CHECK:              } else {
    else:
        # CHECK-NEXT: %inside_else = lit.varlet.decl "inside_else" var
        var inside_else: Int
    # CHECK:                hlcf.yield
    # CHECK-NEXT:         }
    # CHECK-NEXT:         hlcf.yield
    # CHECK-NEXT:       }
    # CHECK-NEXT:       hlcf.yield
    # CHECK-NEXT:     }
    # CHECK-NEXT: %z = lit.varlet.decl "z" var
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant{{.*}}4
    # CHECK-NEXT: store [[FOUR]], %z
    var z: Int = 4

    # Walrus operator in if's.
    # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: store [[FIVE]], %z
    # CHECK-NEXT: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}([[FIVE]])
    # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
    # CHECK-NEXT: hlcf.if [[I1]] {
    if z := 5:
        return a

    return a


# CHECK-LABEL: lit.func @"test_if_nested
fn test_if_nested(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK-NEXT:   [[I1:%.*]] = lit.call {{.*}}Bool::@"__mlir_i1__($builtin::$bool::Bool)"(%a)
    # CHECK-NEXT:              hlcf.if [[I1]]
    if a:
        # CHECK-NEXT: %inside_a = lit.varlet.decl "inside_a" var
        var inside_a: Int
    # CHECK:                   } else {
    # CHECK:                     hlcf.if
    else:
        if b:
            # CHECK-NEXT: %inside_b = lit.varlet.decl "inside_b" var
            var inside_b: Int
        # CHECK:                     } else {
        # CHECK:                       hlcf.if
        else:
            if c:
                # CHECK-NEXT: %inside_c = lit.varlet.decl "inside_c" var
                var inside_c: Int
            # CHECK:                       } else {
            else:
                # CHECK-NEXT: %inside_else = lit.varlet.decl "inside_else" var
                var inside_else: Int
    # CHECK:                         hlcf.yield
    # CHECK:                       }
    # CHECK:                       hlcf.yield
    # CHECK-NEXT:               }
    # CHECK:                    hlcf.yield
    # CHECK-NEXT:             }
    var z: Int = 4
    return a

# CHECK-LABEL: lit.func @"param_if{{.*}})"<
# CHECK-SAME: [[A:.*_a]][a]: i1, [[B:.*_b]][b]: !Bool>()
fn param_if[a: __mlir_type.i1, b: Bool]():
  # CHECK: kgen.param.if <[[A]]> {
  @parameter
  if a:
    # CHECK: lit.varlet.decl "inside_1" var
    var inside_1: Int
  # CHECK: } else {
  # CHECK:     kgen.param.if <apply{{.*}}{{.*}}Bool::@"__mlir_i1__{{.*}}[[B]])> {
  elif b:
  # CHECK:     lit.varlet.decl "inside_2" var
    var inside_2: Int
  # CHECK:     kgen.param.yield
  # CHECK:   }
  # CHECK:   kgen.param.yield
  # CHECK: }

# CHECK-LABEL: lit.func @"param_if_andor_i1[__mlir_type.i1,__mlir_type.i1]()"<
# CHECK-SAME: [[A:.*_a]][a]: i1, [[B:.*_b]][b]: i1>()
fn param_if_andor_i1[a: __mlir_type.i1, b: __mlir_type.i1]():
  # CHECK: kgen.param.if <cond([[A]], [[B]], [[A]])>
  @parameter
  if a and b:
  # CHECK:   lit.varlet.decl "v" var
    var v: Int
  # CHECK:   kgen.param.yield
  # CHECK: } else {
  # CHECK: kgen.param.if <cond([[A]], [[A]], [[B]])>
  elif a or b:
  # CHECK:   lit.varlet.decl "w" var
    var w: Int


# CHECK-LABEL: lit.func @"param_if_and[$builtin::$bool::Bool,$builtin::$bool::Bool]()"<
# CHECK-SAME: [[A:.*_a]][a]: !Bool, [[B:.*_b]][b]: !Bool>()
fn param_if_and[a: Bool, b: Bool]():
  # CHECK: kgen.param.if <apply(
  # CHECK-SAME: !lit.signature<("self": !Bool borrow) -> i1> {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)", cond(
  # CHECK-SAME: apply({{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)", [[A]]), [[B]], [[A]]))> {
  @parameter
  if a and b:
  # CHECK:   lit.varlet.decl "v" var
    var v: Int
  # CHECK:   kgen.param.yield
  # CHECK: }

# [Mojo] Can't have try inside else branch
# https://github.com/modularml/modular/issues/25305
# CHECK-LABEL: lit.func @"if_try
fn if_try(p: Bool):
    # CHECK: hlcf.if %0 {
    if p:
        # CHECK: lit.try {
        try:
            # CHECK: lit.letreg.decl "b"
            let b = 1
            # CHECK: lit.try.yield
        # CHECK: } except (%arg0: !Error)
        except e:
            # CHECK: lit.letreg.decl "c"
            let c = 2
            # CHECK: lit.try.yield
        # CHECK-NEXT: } else {
        # CHECK-NEXT:  lit.try.yield
        # CHECK-NEXT:} finally {
        # CHECK-NEXT:  lit.try.yield
        # CHECK-NEXT:}
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: } else {
    else:
        # CHECK: lit.letreg.decl "d"
        let d = 3
        # CHECK-NEXT: hlcf.yield
    # CHECK-NEXT: }


##===----------------------------------------------------------------------===##
# While
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_while
# CHECK:       %inside_a = lit.varlet.decl "inside_a" var
# CHECK:       %inside_b = lit.varlet.decl "inside_b" var
# CHECK:       %inside_else = lit.varlet.decl "inside_else" var
# CHECK:       lit.loop cond {
# CHECK:         [[V0:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)"(%a)
# CHECK:         lit.loop.condition [[V0]] : i1
# CHECK:       } body {
# CHECK-NEXT:    kgen.param.constant: {{.*}} = <#lit.struct<{value = 0}>>
# CHECK-NEXT:    lit.ref.store {{.+}}, %inside_a
# CHECK:         hlcf.if
# CHECK-NEXT:      kgen.param.constant: {{.*}} = <#lit.struct<{value = 1}>>
# CHECK-NEXT:      lit.ref.store {{.+}}, %inside_b
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
# CHECK-NEXT:    lit.loop.continue
# CHECK-NEXT:  } else {
# CHECK-NEXT:     kgen.param.constant: {{.*}} = <#lit.struct<{value = 2}>>
# CHECK-NEXT:     lit.ref.store {{.+}}, %inside_else
# CHECK-NEXT:    lit.loop.yield
# CHECK-NEXT:  }
# CHECK-NEXT:  lit.return
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }
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


# CHECK-LABEL: lit.func @"test_simple
# CHECK:       lit.loop cond {
# CHECK:         [[V0:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)"(%a)
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


# CHECK-LABEL: lit.func @"test_else_outside_while
def test_else_outside_while(a: Bool, b: Bool) -> Bool:
    # CHECK: %a_0 = lit.varlet.decl "a" imp
    # CHECK: lit.ref.store %a, %a_0
    # CHECK: hlcf.if {{.+}} {
    if b:
        # CHECK: lit.loop cond {
        # CHECK:   [[V0:%.*]] = lit.ref.load %a_0
        # CHECK:   [[V1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)"([[V0]])
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
    # CHECK:   [[V0:%.*]] = lit.ref.load %a_0
    # CHECK:   [[V1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)"([[V0]])
    # CHECK:   lit.loop.condition [[V1]] : i1
    # CHECK: } body {
    while a:
        # CHECK:      hlcf.if
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
    # CHECK: hlcf.if
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
    let my_list = MyList()

    # CHECK: %$RANGE = lit.varlet.decl "$RANGE" synth
    # CHECK-NEXT: [[ITER:.*]] = lit.call @{{.*}}__iter__{{.*}}(%$RANGE, %my_list)
    for item in my_list:
        # CHECK: lit.loop cond {
        # CHECK:   [[LENGTH:%.*]] = lit.call {{.*}}__len__{{.*}}(%$RANGE)
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
        # CHECK: "g" = %{{.*}}
        let g = item
    for item in range(0):
        # CHECK: lit.ref.load %item
        let g = item

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
  let i = 1
  @unroll
  while i < 4:
      _ = i
  # CHECK: } {unrollLevel = #hlcf<unroll_level full>}

fn unroll_factor_parameter():
  alias a = 1
  alias b = 1
  let i = 1
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

