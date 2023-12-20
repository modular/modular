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
fn if_try():
    # CHECK: hlcf.if %0 {
    if True:
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

    # CHECK: %$RANGE = lit.varlet.decl "$RANGE" {{.*}}{isSynthetic}
    # CHECK: [[RANGEPTR:%.*]] = lit.ref.to_pointer %$RANGE
    # CHECK: [[LISTPTR:%.*]] = lit.ref.to_pointer %my_list
    # CHECK: [[ITER:.*]] = lit.call @{{.*}}__iter__{{.*}}([[RANGEPTR]], [[LISTPTR]])
    for item in my_list:
        # CHECK: lit.loop cond {
        # CHECK:   [[RANGEPTR:%.*]] = lit.ref.to_pointer %$RANGE
        # CHECK:   [[LENGTH:%.*]] = lit.call {{.*}}__len__{{.*}}([[RANGEPTR]])
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

##===----------------------------------------------------------------------===##
# Raise and Try
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"simpleTryExcept
fn simpleTryExcept():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: except (%{{.*}}: !Error)
    except:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: else
    # CHECK-NEXT: lit.try.yield
    # CHECK: lit.end_func


# CHECK-LABEL: lit.func @"tryExceptElse
fn tryExceptElse():
    var a: Int
    # CHECK: lit.try
    try:
        pass
    except:
        pass
    # CHECK: else
    else:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield


fn eatError(err: Error):
    pass


# CHECK-LABEL: lit.func @"tryExceptArg
fn tryExceptArg():
    try:
        pass
    # CHECK: except (%arg0: !Error)
    except err:
        # CHECK-NEXT: lit.call @"{{.*}}::@"eatError{{.*}}(%arg0)
        eatError(err)


# CHECK-LABEL: lit.func @"tryExceptArgDef
def tryExceptArgDef():
    try:
        pass
    # CHECK: except (%arg0: !Error)
    except err:
        # CHECK-NEXT: lit.varlet.decl "err" imp
        # CHECK: [[ERRVAL:%.*]] = lit.ref.load %err
        # CHECK: eatError{{.*}}([[ERRVAL]])
        eatError(err)


# CHECK-LABEL: lit.func @"tryFinally
fn tryFinally():
    # CHECK-NEXT: lit.try
    try:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: except
    # CHECK-NEXT: lit.try.yield
    # CHECK: finally
    finally:
        # CHECK: lit.return
        return
    # CHECK: lit.try {
    try:
        # CHECK-NEXT: lit.try
        try:
            # CHECK-NEXT: lit.try.yield
            pass
        # CHECK-NEXT: except (%arg0:
        # CHECK-NEXT: lit.raise %arg0
        finally:
            pass
    except:
        pass


def maybeRaises() -> Int:
    return 0


# CHECK-LABEL: lit.func @"propagateErrorInDef
def propagateErrorInDef():
    # CHECK: %[[VALUE:.*]] = lit.call @"{{.*}}"::@"maybeRaises
    # CHECK: %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
    # CHECK: {
    # CHECK:    [[VAR:%.*]] = kgen.variant.get %0, 1 : <!Error, !Int>
    # CHECK:    lit.yield [[VAR]] : !Int
    # CHECK: } else {
    # CHECK:    [[ERR:%.*]] = kgen.variant.get %0, 0 : <!Error, !Int>
    # CHECK:    lit.raise [[ERR]] : !Error
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK: %a = lit.varlet.decl "a"
    # CHECK-NEXT: lit.ref.store %1, %a
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInRaisingFn
fn propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.varlet.decl {{.*}} : !lit.ref<mut !Int,
    var a: Int
    # CHECK:  %0 = lit.call @"$statements"::@"maybeRaises()"() : !lit.signature<() throws -> !kgen.variant<!Error, !Int>>
    # CHECK:  %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
    # CHECK:  {
    # CHECK:    [[ERR:%.*]] = kgen.variant.get %0
    # CHECK:    lit.yield [[ERR]] : !Int
    # CHECK:  } else {
    # CHECK:    [[ERR:%.*]] = kgen.variant.get %0
    # CHECK:    lit.raise [[ERR]] : !Error
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK:  lit.ref.store %1, %a
    a = maybeRaises()

# CHECK-LABEL: lit.func @"propagateErrorInTry
fn propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: %0 = lit.call @"$statements"::@"maybeRaises()"()
        # CHECK: %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
        # CHECK: {
        # CHECK: } else {
        # CHECK:   [[ERR:%.*]] = kgen.variant.get %0
        # CHECK:   lit.raise [[ERR]] : !Error
        # CHECK: }

        # CHECK-NEXT: lit.ref.store %1, %a
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.func @"raiseError
def raiseErrorInDef(err: Error):
    # CHECK: %err_0 = lit.varlet.decl "err"
    # CHECK: lit.ref.store %err, %err_0
    # CHECK: %[[ERRVAL:.*]] = lit.ref.load %err_0
    # CHECK: %[[ERRVALCOPY:.*]] = lit.call {{.*}}@Error::@"__copyinit__
    # CHECK: lit.raise %[[ERRVALCOPY]] : !Error
    raise err


# CHECK-LABEL: lit.func @"raiseErrorInIf
def raiseErrorInIf(cond: Bool, err: Error):
    # CHECK: hlcf.if
    if cond:
        # CHECK: lit.raise {{.*}} : !Error
        raise err


# CHECK-LABEL: lit.func @"raiseErrorInTry
fn raiseErrorInTry(err: Error):
    # CHECK: lit.try {
    try:
        # CHECK-NEXT: = lit.call {{.*}}@Error::@"__copyinit__
        # CHECK-NEXT: lit.raise {{.*}} : !Error
        raise err
    except:
        pass


# CHECK-LABEL: lit.func @"rethrowsToRethrow
fn rethrowsToRethrow():
    # CHECK: lit.try {
    try:
        # CHECK: lit.try {
        try:
            # CHECK:  lit.call @"$statements"::@"maybeRaises()"()
            maybeRaises() # expected-warning {{'Int' value is unused}}
        # CHECK: } except (%arg0:
        except:
            # CHECK: lit.raise %arg0
            raise
        # CHECK: }
    # CHECK: } except (%arg0: !Error)
    except:
        # CHECK: lit.return %none
        return

# Issue #12358
# CHECK-LABEL: lit.func @"raise_string
fn raise_string() raises:
   # CHECK: %0 = kgen.param.constant: !StringLiteral = <#lit.struct<{value: string = "thing"}>>
   # CHECK: %1 = lit.call @"$builtin"::@"$error"::@Error::@"__init__{{.*}}"(%0) : !lit.signature<("value": !StringLiteral borrow) ownedresult -> !Error>
   # CHECK: lit.raise %1 : !Error
   raise "thing"

struct S:
  var v: Int

  fn __init__(inout self, x: Int):
    self.v = x

  fn __init__(inout self) raises:
    self.v = 1

  fn __copyinit__(inout self, existing: Self):
    self.v = existing.v


fn fail(str: StringRef) raises -> S:
  return 0


# CHECK-LABEL: lit.func @"call_raising
fn call_raising():
  # CHECK: [[XPTR:%.*]] = lit.ref.to_pointer %x
  try:
    # CHECK: [[ERR:%.*]] =  lit.call @"$statements"::@"fail
    # CHECK: [[VAR0:%.*]] = lit.handle_variant [[ERR]], [[XPTR]]
    # CHECK:   [[VAR1:%.*]] = kgen.variant.get [[ERR]]
    # CHECK:   lit.yield [[VAR1]] : !kgen.none
    # CHECK: } else {
    # CHECK:   [[VAR2:%.*]] = kgen.variant.get [[ERR]]
    # CHECK:   lit.raise [[VAR2]]
    # CHECK:   kgen.unreachable
    # CHECK: }
    let x = fail("hello world")
    # CHECK: %y = lit.varlet.decl "y"
    # CHECK: [[YPTR:%.*]] = lit.ref.to_pointer %y
    # CHECK: [[VAR1:%.*]] = lit.handle_variant [[ERR:.*]], [[YPTR]]
    # CHECK:   [[VAR2:%.*]] = kgen.variant.get [[ERR]]
    # CHECK:   lit.yield [[VAR2]] : !kgen.none
    # CHECK: } else {
    # CHECK:   [[VAR2:%.*]] = kgen.variant.get [[ERR]]
    # CHECK:   lit.raise [[VAR2]]
    # CHECK:   kgen.unreachable
    # CHECK: }
    let y = S()
  except e:
    pass


fn fail_raises(str: StringRef) raises -> S:
  return fail(str)


fn fail_register(str: StringRef) raises -> Int:
  return 0


fn fail_register_raises(str: StringRef) raises -> Int:
  # CHECK: %[[VAR0:.*]] = lit.handle_variant %0
  # CHECK:   %[[VAR1:.*]] = kgen.variant.get %0
  # CHECK:   lit.yield %[[VAR1]]
  # CHECK: } else {
  # CHECK:   %[[VAR2:.*]] = kgen.variant.get %0
  # CHECK:   lit.raise %[[VAR2]]
  # CHECK:   kgen.unreachable
  # CHECK: }
  return fail_register(str)

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM:
  fn __copyinit__(inout self, existing: Self): pass

  fn __enter__(self) -> Int:
    return 42
  fn __exit__(self):
    pass # normal
  fn __exit__(self, err: Error) -> Bool:
    return True # Raise

# Cannot use mutating __enter__
# https://github.com/modularml/modular/issues/27371
struct MutatingCM:
  fn __init__(inout self): pass
  fn __enter__(inout self) -> Int:
    return 42
  fn __exit__(inout self):
    pass # normal

fn noop(a: Int): pass


# CHECK-LABEL: lit.func @"testWithNonRaising
fn testWithNonRaising(a: ExampleCM):
  # CHECK-NEXT: $CONTEXTMGR = lit.varlet.decl "$CONTEXTMGR"
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[CTXPTR]], %a)
  # CHECK-NEXT: %val = lit.varlet.decl {{.*}} imp
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer
  # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[CTXPTR]])
  # CHECK-NEXT: lit.ref.store [[TARGET]], %val
  # CHECK-NEXT: lit.try
  with a as val:
    # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load %val
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
    noop(val)
  # CHECK: finally
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer
  # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[CTXPTR]])

  # Test a with with no target.

  # CHECK: %$CONTEXTMGR_0 = lit.varlet.decl "$CONTEXTMGR"
  # CHECK: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_0
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[CTXPTR]], %a)
  # CHECK: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_0
  # CHECK: lit.call {{.*}}__enter__{{.*}}([[CTXPTR]])
  # CHECK-NEXT: lit.try
  with a:
    # CHECK-NEXT: kgen.param.constant: {{.*}}42
    # CHECK-NEXT: lit.call {{.*}}noop
    noop(42)
  # CHECK: finally
  # CHECK: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_0
  # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[CTXPTR]])

  # CHECK: %$CONTEXTMGR_1 = lit.varlet.decl "$CONTEXTMGR"{{.*}}!MutatingCM
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_1
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[CTXPTR]])
  # CHECK: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_1
  # CHECK-NEXT: lit.call {{.*}}__enter__{{.*}}([[CTXPTR]])
  with MutatingCM() as val:
    # CHECK: lit.call {{.*}}noop
    noop(val)
  # CHECK: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_1
  # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[CTXPTR]])

# CHECK-LABEL: lit.func @"testWithRaising
fn testWithRaising(a: ExampleCM) raises:
  # CHECK: %$CONTEXTMGR = lit.varlet.decl
  # CHECK: %val = lit.varlet.decl {{.*}} imp
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[CTXPTR]])
  # CHECK-NEXT: lit.ref.store [[TARGET]], %val
  # CHECK: lit.ref.store %true, %__with_exc__
  # CHECK-NEXT: lit.try
  # CHECK-NEXT: lit.try
  with a as val:
    # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load %val
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
    noop(val)

    # CHECK-NEXT: [[RESULT:%.*]] = lit.call {{.*}}raise_string()
    # CHECK-NEXT: lit.handle_variant [[RESULT]]
    # CHECK-NEXT:   [[OK:%.*]] = kgen.variant.get [[RESULT]]
    # CHECK-NEXT:   lit.yield [[OK]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   kgen.variant.get
    # CHECK-NEXT:   lit.raise
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    raise_string()
    # CHECK-NEXT: lit.try.yield
  # CHECK-NEXT: } except (%arg0: !Error) {
  # CHECK:        lit.ref.store %false, %__with_exc__
  # CHECK-NEXT:   [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT:   [[EXIT_RESULT:%.*]] = lit.call {{.*}}__exit__{{.*}}([[CTXPTR]], %arg0)
  # CHECK-NEXT:   [[SUCCESS:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[EXIT_RESULT]])
  # CHECK-NEXT:   hlcf.if [[SUCCESS]] {
  # CHECK-NEXT:     hlcf.yield
  # CHECK-NEXT:   } else {
  # CHECK-NEXT:     lit.raise %arg0
  # CHECK-NEXT:     hlcf.yield
  # CHECK-NEXT:   }
  # CHECK-NEXT:   lit.try.yield
  # CHECK:      } finally {
  # CHECK:    } except
  # CHECK-NEXT:  lit.raise %arg0
  # CHECK:    } finally {
  # CHECK-NEXT: %[[EXC:.*]] = lit.ref.load %__with_exc__
  # CHECK-NEXT: hlcf.if %[[EXC]]
  # CHECK-NEXT:   [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT:   call {{.*}}__exit__{{.*}}([[CTXPTR]])

# CHECK-LABEL: lit.func @"testWithInTry
fn testWithInTry(a: ExampleCM):
  # CHECK: lit.try {
  try:
     # CHECK: %$CONTEXTMGR = lit.varlet.decl
     # CHECK: %cm = lit.varlet.decl "cm"
     # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
     # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[CTXPTR]])
     # CHECK-NEXT: lit.ref.store [[TARGET]], %cm
     # CHECK: lit.ref.store %true, %__with_exc__
     # CHECK: lit.try {
     with a as cm:
        # CHECK: lit.try {
        # CHECK-NEXT: [[RESULT:%.*]] = lit.call {{.*}}raise_string()
        # CHECK-NEXT: lit.handle_variant [[RESULT]]
        # CHECK-NEXT:   [[OK:%.*]] = kgen.variant.get [[RESULT]]
        # CHECK-NEXT:   lit.yield [[OK]]
        raise_string()
  except e:
    _ = e


# CHECK-LABEL: lit.func @"testWithScoping
fn testWithScoping(a: ExampleCM):
  # This is a test that issue #18811 is fixed, in which a `with`
  # statement inside a `fn` does not respect lexical scope and binds
  # its variable in its parent scope.
  with a as withDecl:
    # CHECK: %withDecl = lit.varlet.decl "withDecl" imp
    noop(withDecl)
  with a as withDecl:
    # CHECK: = lit.varlet.decl "withDecl" imp
    noop(withDecl)

# CHECK-LABEL: lit.func @"testWithInDef
def testWithInDef(a: ExampleCM):
  # This is a test that issue #20141 is fixed.
  # https://github.com/modularml/modular/issues/20141
  # IE that when used inside a `def`, the `with` statement uses
  # mutable function scope variables.
  # CHECK: [[VAL1:%.*]] = lit.ref.load %val1
  val1 = 77
  # CHECK: lit.call {{.*}}noop{{.*}}([[VAL1]])
  noop(val1)
  with a as val1:
    # CHECK: [[VAL1:%.*]] = lit.ref.load %val1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL1]])
    noop(val1)
  noop(val1)
  with a as val2:
    # CHECK: [[VAL2:%.*]] = lit.ref.load %val2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL2]])
    noop(val2)
  # CHECK: [[VAL2:%.*]] = lit.ref.load %val2
  val2 = 78
  # CHECK: lit.call {{.*}}noop{{.*}}([[VAL2]])
  noop(val2)


# Issue #21990: [Mojo-lang] Support context managers in with statements that
# don't implement the __exit__ method.
# https://github.com/modularml/modular/issues/21990

struct CMWithoutExit:
  fn __init__(inout self): pass
  fn __moveinit__(inout self, owned existing: Self): pass

  # This context manager consumes itself and returns it as the value.
  fn __enter__(owned self) -> Self:
    return self^
  fn method(self): pass

# CHECK-LABEL: lit.func @"testCMWithoutExit
fn testCMWithoutExit():
  # CHECK: [[APTR:%.*]] = lit.ref.to_pointer %a
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}([[APTR]], [[CTXPTR]])
  # CHECK-NEXT: lit.try {
  # CHECK-NEXT:   [[APTR1:%.*]] = lit.ref.to_pointer %a
  # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[APTR1]])
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } except (%arg0: i1) {
  # CHECK-NEXT:   kgen.unreachable
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } finally {
  # CHECK-NEXT:   lit.ownership.use [[APTR]]
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: }
  with CMWithoutExit() as a:
    a.method()

  # CHECK: %$CONTEXTMGR_0 = lit.varlet.decl "$CONTEXTMGR"
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_0
  # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}([[CTXPTR]])
  # CHECK: %a_1 = lit.varlet.decl "a"
  # CHECK-NEXT: [[APTR:%.*]] = lit.ref.to_pointer %a_1
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR_0
  # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}([[APTR]], [[CTXPTR]])
  # CHECK-NEXT: lit.try {
  # CHECK-NEXT:   [[APTR1:%.*]] = lit.ref.to_pointer %a_1
  # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[APTR1]])
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } except (%arg0: i1) {
  # CHECK-NEXT:   kgen.unreachable
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } finally {
  # CHECK-NEXT:   lit.ownership.use [[APTR]]
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: }

  # Test that we don't have a name collision between two 'a's.
  with CMWithoutExit() as a:
    a.method()

  # Test that we can nest these.
  with CMWithoutExit() as a:
    with CMWithoutExit() as b:
      b.method()

# CHECK-LABEL: lit.func @"testCMWithoutExitEarlyReturn
# https://github.com/modularml/modular/issues/23693
fn testCMWithoutExitEarlyReturn():
  # CHECK: %$CONTEXTMGR = lit.varlet.decl "$CONTEXTMGR"
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}([[CTXPTR]])
  # CHECK: %a = lit.varlet.decl "a"
  # CHECK-NEXT: [[APTR:%.*]] = lit.ref.to_pointer %a
  # CHECK-NEXT: [[CTXPTR:%.*]] = lit.ref.to_pointer %$CONTEXTMGR
  # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}([[APTR]], [[CTXPTR]])
  # CHECK-NEXT: lit.try {
  # CHECK-NEXT:   [[APTR1:%.*]] = lit.ref.to_pointer %a
  # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[APTR1]])
  # CHECK-NEXT:   %none_0 = kgen.param.constant: none = <#kgen.none>
  # CHECK-NEXT:   lit.return %none_0 : !kgen.none
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } except (%arg0: i1) {
  # CHECK-NEXT:   kgen.unreachable
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: } finally {
  # CHECK-NEXT:   lit.ownership.use [[APTR]]
  # CHECK-NEXT:   lit.try.yield
  # CHECK-NEXT: }
  with CMWithoutExit() as a:
    a.method()
    return

##===----------------------------------------------------------------------===##

# TODO(Issue #6139)

# struct Iterable:

# fn test_for(iterable: Iterable):
#  var result = 0
#  for i in iterable:
#    result += i

##===----------------------------------------------------------------------===##
# Struct with Nonmaterializable
##===----------------------------------------------------------------------===##

@value
@register_passable("trivial")
struct NmTarget:
  var x: Bool
  fn __init__(x: Bool) -> Self:
    return Self {x: x}
  @always_inline("nodebug")
  fn __init__(nms: NmStruct) -> Self:
    return Self {x: True if (nms.x == 77) else False}
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

fn tail_types[T: AnyRegType, *U: AnyRegType](a: T, *b: *U):
    pass
fn nmTargetNoop(x: NmTarget): pass
fn useNonmaterializable():
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "gotConverted1" var : !lit.ref<mut !NmTarget
  # CHECK-NEXT: kgen.param.constant: !NmTarget {{.*}}true
  var gotConverted1 = NmStruct(76) + NmStruct(1)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "gotConverted2" var : !lit.ref<mut !NmTarget
  # CHECK-NEXT: kgen.param.constant: !NmTarget {{.*}}false
  var gotConverted2 = notMaterializedAlias + NmStruct(1)
  # CHECK: lit.alias.decl{{.*}}useIfAlias{{.*}}NmStruct{{.*}}2
  alias useIfAlias = NmStruct(2) if True else NmStruct(3)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useIfVar" var : !lit.ref<mut !NmTarget
  # CHECK: kgen.param.constant: !NmTarget {{.*}}false
  var useIfVar = NmStruct(2) if True else NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useIfVarLopsided" var : !lit.ref<mut !NmTarget
  # CHECK: kgen.param.constant: !NmTarget {{.*}}true
  var useIfVarLopsided = NmTarget(False) if False else NmStruct(77)

  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useOrVar1" var : !lit.ref<mut !NmTarget
  var useOrVar1 = NmStruct(2) or NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useOrVar2" var : !lit.ref<mut !NmTarget
  var useOrVar2 = NmStruct(2) or NmStruct(3)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useAndVar1" var : !lit.ref<mut !NmTarget
  var useAndVar1 = NmStruct(2) and NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useAndVar2" var : !lit.ref<mut !NmTarget
  var useAndVar2 = NmStruct(77) and NmStruct(77)

  # Test that parameter inference using nonmaterializable gives the target,
  # not the nonmaterializable type.
  # CHECK: call {{.*}}tail_types{{.*}}<:regtype !NmTarget, :variadic<regtype> []>
  tail_types(NmStruct(5))
  # CHECK: call {{.*}}tail_types{{.*}}<:regtype !NmTarget, :variadic<regtype> [{{.*}}NmTarget]>
  tail_types(NmStruct(5), NmStruct(6))
