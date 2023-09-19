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
    # CHECK: %1 = pop.variant.create %0
    # CHECK: lit.return %1 : !pop.variant<!Error, !Int>
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
    # CHECK-NEXT: %z = lit.varlet.decl "z"
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant{{.*}}4
    # CHECK-NEXT: store [[FOUR]], %z
    var z: Int = 4

    # Walrus operator in if's.
    # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: store [[FIVE]], %z
    # CHECK-NEXT: [[BOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}([[FIVE]])
    # CHECK-NEXT: [[I1:%.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
    # CHECK-NEXT: hlcf.if [[I1]] {
    if z := 5:
        return a

    return a


# CHECK-LABEL: lit.func @"test_if_nested
fn test_if_nested(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK-NEXT:   [[I1:%.*]] = kgen.call {{.*}}Bool::@"__mlir_i1__($builtin::$bool::Bool)"(%a)
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
# CHECK-SAME: [[A:.*]]: i1, [[B:.*]]: !Bool>()
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
# CHECK-SAME: [[A:.*]]: i1, [[B:.*]]: i1>()
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
# CHECK-SAME: [[A:.*]]: !Bool, [[B:.*]]: !Bool>()
fn param_if_and[a: Bool, b: Bool]():
  # CHECK: kgen.param.if <apply(
  # CHECK-SAME: :("self": !Bool borrow) -> i1 {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)", cond(
  # CHECK-SAME: apply(:("self": !Bool borrow) -> i1 {{.*}}@Bool::@"__mlir_i1__($builtin::$bool::Bool)", [[A]]), [[B]], [[A]]))> {
  @parameter
  if a and b:
  # CHECK:   lit.varlet.decl "v" var
    var v: Int
  # CHECK:   kgen.param.yield
  # CHECK: }

##===----------------------------------------------------------------------===##
# While
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_while
# CHECK:       %inside_a = lit.varlet.decl "inside_a" var
# CHECK:       %inside_b = lit.varlet.decl "inside_b" var
# CHECK:       %inside_else = lit.varlet.decl "inside_else" var
# CHECK:       hlcf.loop {
# CHECK:         hlcf.if
# CHECK-NEXT:     hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:     kgen.param.constant: {{.*}} = <#lit.struct<{value = 2}>>
# CHECK-NEXT:     pop.store {{.+}}, %inside_else
# CHECK-NEXT:     hlcf.break
# CHECK-NEXT:    }
# CHECK-NEXT:    kgen.param.constant: {{.*}} = <#lit.struct<{value = 0}>>
# CHECK-NEXT:    pop.store {{.+}}, %inside_a
# CHECK:         hlcf.if
# CHECK-NEXT:      kgen.param.constant: {{.*}} = <#lit.struct<{value = 1}>>
# CHECK-NEXT:      pop.store {{.+}}, %inside_b
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
# CHECK-NEXT:    hlcf.continue
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
# CHECK:       hlcf.loop {
# CHECK:       hlcf.if
# CHECK-NEXT:       hlcf.yield
# CHECK-NEXT:     } else {
# CHECK-NEXT:       hlcf.break
# CHECK-NEXT:     }
# CHECK-NEXT:     hlcf.continue
# CHECK-NEXT:   }
# CHECK-NEXT:   %0 = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:   lit.return %0 :  !lit.none
# CHECK-NEXT:   lit.end_func
# CHECK-NEXT: }
fn test_simple(a: Bool):
    while a:
        pass


# CHECK-LABEL: lit.func @"test_else_outside_while
def test_else_outside_while(a: Bool, b: Bool) -> Bool:
    # CHECK: %a_0 = lit.varlet.decl2 "a"
    # CHECK: [[APTR:%.*]] = lit.ref_to_pointer %a_0
    # CHECK: pop.store %a, [[APTR]]
    # CHECK: [[APTR:%.*]] = lit.ref_to_pointer %a_0
    # CHECK: hlcf.if {{.+}} {
    if b:
        # CHECK: hlcf.loop
        # CHECK: {{.+}} = pop.load [[APTR]]
        while a:
            # CHECK: pop.store {{.+}}, %inside_a
            inside_a = 0
            # CHECK: hlcf.continue
    # CHECK: } else {
    else:
        # CHECK: pop.store {{.+}}, %inside_else
        inside_else = 2
    # CHECK: }
    # CHECK: lit.return
    return a


# CHECK-LABEL: lit.func @"test_break_continue_inside_while
def test_break_continue_inside_while(a: Bool) -> Bool:
    # CHECK: hlcf.loop
    # CHECK: hlcf.if
    while a:
        # CHECK:      hlcf.if
        if a:
            # CHECK-NEXT:   lit.break
            break
            # CHECK:   pop.store
            # CHECK-NEXT:   hlcf.yield
            c = 1
        else:
            # CHECK-NEXT: } else {
            # CHECK-NEXT:   lit.continue
            continue
            # CHECK-NEXT:   hlcf.yield
        # CHECK: hlcf.continue
    return a


# CHECK-LABEL: lit.func @"test_early_return
def test_early_return():
    # CHECK: hlcf.if
    var a: Bool
    if a:
        # CHECK: lit.return
        return
        # CHECK: pop.store
        b = 2
        # CHECK-NEXT: hlcf.yield
    # CHECK: else
    # CHECK-NEXT: yield
    # CHECK: lit.return
    return
    # CHECK: pop.store
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

    # CHECK: %$RANGE = lit.varlet.decl "$RANGE"
    # CHECK: [[ITER:.*]] = kgen.call @{{.*}}__iter__{{.*}}(%$RANGE, %my_list)
    for item in my_list:
        # CHECK: [[LENGTH:%.*]] = kgen.call {{.*}}__len__{{.*}}(%$RANGE)
        # CHECK: [[INDEX:%.*]] = kgen.call {{.*}}__index__{{.*}}([[LENGTH]])
        # CHECK: [[MLIR_INDEX:%.*]] = kgen.call {{.*}}__mlir_index__{{.*}}([[INDEX]])
        # CHECK: [[COND:%.*]] = index.cmp sgt([[MLIR_INDEX]], %idx0)
        # CHECK: if [[COND]]
        # CHECK-NEXT: yield
        pass


# CHECK-LABEL: @"induction_var_scope()"
fn induction_var_scope():
    # CHECK: "item"
    # CHECK: hlcf.loop
    for item in range(0):
        # CHECK: pop.load %item
        # CHECK: "g" = %{{.*}}
        let g = item
    for item in range(0):
        # CHECK: pop.load %item
        let g = item

# CHECK-LABEL: lit.func @"unroll_for()"
fn unroll_for():
    @unroll
    for i in range(1, 9, 2):
        print(i)
        @unroll
        for j in range (1, 4):
            print (i + j)
    # CHECK: } {unrollLevel = #hlcf<unroll_level full>}
    # CHECK: } {unrollLevel = #hlcf<unroll_level full>}

    @unroll(2)
    for j in range (1, 4):
        print (j)
    # CHECK: } {unrollLevel = #hlcf<unroll_level 2>}

# CHECK-LABEL: lit.func @"unroll_while()"
fn unroll_while():
  let i = 1
  @unroll
  while i < 4:
      print(i)
  # CHECK: } {unrollLevel = #hlcf<unroll_level full>}

##===----------------------------------------------------------------------===##
# Raise and Try
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"simpleTryExcept
fn simpleTryExcept():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: pop.store
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
        # CHECK: pop.store
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
        # CHECK-NEXT: kgen.call @"{{.*}}::@"eatError{{.*}}(%arg0)
        eatError(err)


# CHECK-LABEL: lit.func @"tryExceptArgDef
def tryExceptArgDef():
    try:
        pass
    # CHECK: except (%arg0: !Error)
    except err:
        # CHECK-NEXT: lit.varlet.decl "err" var
        # CHECK: [[ERRVAL:%.*]] = pop.load %err
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
    # CHECK: %[[VALUE:.*]] = kgen.call @"{{.*}}"::@"maybeRaises
    # CHECK: %1 = lit.handle_variant %0 : (!pop.variant<!Error, !Int>) -> !Int
    # CHECK: {
    # CHECK:    [[VAR:%.*]] = pop.variant.get %0 : !pop.variant<!Error, !Int> as !Int
    # CHECK:    lit.yield [[VAR]] : !Int
    # CHECK: } else {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0 : !pop.variant<!Error, !Int> as !Error
    # CHECK:    lit.raise [[ERR]] : !Error
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK-NEXT: pop.store %1, %a
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInRaisingFn
fn propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.varlet.decl {{.*}} : <!Int>
    var a: Int
    # CHECK:  %0 = kgen.call @"$statements"::@"maybeRaises()"() : () throws -> !pop.variant<!Error, !Int>
    # CHECK:  %1 = lit.handle_variant %0 : (!pop.variant<!Error, !Int>) -> !Int
    # CHECK:  {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0
    # CHECK:    lit.yield [[ERR]] : !Int
    # CHECK:  } else {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0
    # CHECK:    lit.raise [[ERR]] : !Error
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK:  pop.store %1, %a
    a = maybeRaises()

# CHECK-LABEL: lit.func @"propagateErrorInTry
fn propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: %1 = kgen.call @"$statements"::@"maybeRaises()"() : () throws -> !pop.variant<!Error, !Int>
        # CHECK: %2 = lit.handle_variant %1 : (!pop.variant<!Error, !Int>) -> !Int
        # CHECK: {
        # CHECK: } else {
        # CHECK:   [[ERR:%.*]] = pop.variant.get %1
        # CHECK:   lit.raise [[ERR]] : !Error
        # CHECK: }

        # CHECK-NEXT: pop.store %2, %a
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.func @"raiseError
def raiseErrorInDef(err: Error):
    # CHECK: %err_0 = lit.varlet.decl2 "err"
    # CHECK: %0 = lit.ref_to_pointer %err_0
    # CHECK: pop.store %err, %0
    # CHECK: %1 = lit.ref_to_pointer %err_0
    # CHECK: %[[ERRVAL:.*]] = pop.load %1
    # CHECK: %[[ERRVALCOPY:.*]] = kgen.call {{.*}}@Error::@"__copyinit__
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
        # CHECK-NEXT: = kgen.call {{.*}}@Error::@"__copyinit__
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
            # CHECK:  kgen.call @"$statements"::@"maybeRaises()"()
            maybeRaises() # expected-warning {{'Int' value is unused}}
        # CHECK: } except (%arg0:
        except:
            # CHECK: lit.raise %arg0
            raise
        # CHECK: }
    # CHECK: } except (%arg0: !Error)
    except:
        # CHECK: lit.return %0
        return

# Issue #12358
# CHECK-LABEL: lit.func @"raise_string
fn raise_string() raises:
   # CHECK-NEXT: %0 = kgen.param.materialize: {{.*}}Error = <{{.*}}>
   # CHECK-NEXT: lit.raise %0 : !Error
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


fn call_raising():
  try:
    # CHECK: %[[VAR0:.*]] = lit.handle_variant %[[ERR:.*]], %x
    # CHECK:   %[[VAR1:.*]] = pop.variant.get %[[ERR]]
    # CHECK:   lit.yield %[[VAR1]] : !lit.none
    # CHECK: } else {
    # CHECK:   %[[VAR2:.*]] = pop.variant.get %[[ERR]]
    # CHECK:   lit.raise %[[VAR2]]
    # CHECK:   kgen.unreachable
    # CHECK: }
    let x = fail("hello world")
    # CHECK: %[[VAR1:.*]] = lit.handle_variant %[[ERR:.*]], %y
    # CHECK:   %[[VAR2:.*]] = pop.variant.get %[[ERR]]
    # CHECK:   lit.yield %[[VAR2]] : !lit.none
    # CHECK: } else {
    # CHECK:   %[[VAR2:.*]] = pop.variant.get %[[ERR]]
    # CHECK:   lit.raise %[[VAR2]]
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
  # CHECK:   %[[VAR1:.*]] = pop.variant.get %0
  # CHECK:   lit.yield %[[VAR1]]
  # CHECK: } else {
  # CHECK:   %[[VAR2:.*]] = pop.variant.get %0
  # CHECK:   lit.raise %[[VAR2]]
  # CHECK:   kgen.unreachable
  # CHECK: }
  return fail_register(str)

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM:
  fn __enter__(self) -> Int:
    return 42
  fn __exit__(self):
    pass # normal
  fn __exit__(self, err: Error) -> Bool:
    return True # Raise

fn noop(a: Int): pass


# CHECK-LABEL: lit.func @"testWithNonRaising
fn testWithNonRaising(a: ExampleCM):
  # CHECK-NEXT: %val = lit.varlet.decl
  # CHECK-NEXT: [[TARGET:%.*]] = kgen.call {{.*}}__enter__{{.*}}(%a)
  # CHECK-NEXT: pop.store [[TARGET]], %val
  # CHECK-NEXT: lit.try
  with a as val:
    # CHECK-NEXT: [[VAL:%.*]] = pop.load %val
    # CHECK-NEXT: kgen.call {{.*}}noop{{.*}}([[VAL]])
    noop(val)
  # CHECK: finally
  # CHECK-NEXT: kgen.call {{.*}}__exit__{{.*}}(%a)

  # Test a with with no target.

  # CHECK: kgen.call {{.*}}__enter__{{.*}}(%a)
  # CHECK-NEXT: lit.try
  with a:
    # CHECK-NEXT: kgen.param.constant: {{.*}}42
    # CHECK-NEXT: kgen.call {{.*}}noop
    noop(42)
  # CHECK: finally
  # CHECK-NEXT: kgen.call {{.*}}__exit__{{.*}}(%a)

# CHECK-LABEL: lit.func @"testWithRaising
fn testWithRaising(a: ExampleCM) raises:
  # CHECK-NEXT: %val = lit.varlet.decl
  # CHECK-NEXT: [[TARGET:%.*]] = kgen.call {{.*}}__enter__{{.*}}(%a)
  # CHECK-NEXT: pop.store [[TARGET]], %val
  # CHECK: pop.store %true, %__with_exc__ : !kgen.pointer<i1>
  # CHECK-NEXT: lit.try
  # CHECK-NEXT: lit.try
  with a as val:
    # CHECK-NEXT: [[VAL:%.*]] = pop.load %val
    # CHECK-NEXT: kgen.call {{.*}}noop{{.*}}([[VAL]])
    noop(val)

    # CHECK-NEXT: %5 = kgen.call {{.*}}raise_string()
    # CHECK-NEXT: %6 = lit.handle_variant %5
    # CHECK-NEXT:   %7 = pop.variant.get %5
    # CHECK-NEXT:   lit.yield %7 : !lit.none
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   pop.variant.get
    # CHECK-NEXT:   lit.raise
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    raise_string()
    # CHECK-NEXT: lit.try.yield
  # CHECK-NEXT: } except (%arg0: !Error) {
  # CHECK:        pop.store %false, %__with_exc__
  # CHECK-NEXT:   %3 = kgen.call {{.*}}__exit__{{.*}}(%a, %arg0)
  # CHECK-NEXT:   %4 = kgen.call {{.*}}__mlir_i1__{{.*}}(%3)
  # CHECK-NEXT:   hlcf.if %4 {
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
  # CHECK-NEXT: %[[EXC:.*]] = pop.load %__with_exc__
  # CHECK-NEXT: hlcf.if %[[EXC]]
  # CHECK-NEXT:   call {{.*}}__exit__{{.*}}(%a)

fn testWithScoping(a: ExampleCM):
  # This is a test that issue #18811 is fixed, in which a `with`
  # statement inside a `fn` does not respect lexical scope and binds
  # its variable in its parent scope.
  with a as withDecl:
    # CHECK: %withDecl = lit.varlet.decl "withDecl"{{.*}}
    noop(withDecl)
  with a as withDecl:
    # CHECK: %withDecl_0 = lit.varlet.decl "withDecl"{{.*}}
    noop(withDecl)

def testWithInDef(a: ExampleCM):
  # This is a test that issue #20141 is fixed.
  # https://github.com/modularml/modular/issues/20141
  # IE that when used inside a `def`, the `with` statement uses
  # mutable function scope variables.
  # CHECK: [[VAL1:%.*]] = pop.load %val1
  val1 = 77
  # CHECK: kgen.call {{.*}}noop{{.*}}([[VAL1]])
  noop(val1)
  with a as val1:
    # CHECK: [[VAL1:%.*]] = pop.load %val1
    # CHECK-NEXT: kgen.call {{.*}}noop{{.*}}([[VAL1]])
    noop(val1)
  noop(val1)
  with a as val2:
    # CHECK: [[VAL2:%.*]] = pop.load %val2
    # CHECK-NEXT: kgen.call {{.*}}noop{{.*}}([[VAL2]])
    noop(val2)
  # CHECK: [[VAL2:%.*]] = pop.load %val2
  val2 = 78
  # CHECK: kgen.call {{.*}}noop{{.*}}([[VAL2]])
  noop(val2)



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

fn tail_types[T: AnyType, *U: AnyType](a: T, *b: *U):
    pass
fn nmTargetNoop(x: NmTarget): pass
fn useNonmaterializable():
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "gotConverted1" var : <!NmTarget>
  # CHECK-NEXT: kgen.param.constant: !NmTarget {{.*}}true
  var gotConverted1 = NmStruct(76) + NmStruct(1)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "gotConverted2" var : <!NmTarget>
  # CHECK-NEXT: kgen.param.constant: !NmTarget {{.*}}false
  var gotConverted2 = notMaterializedAlias + NmStruct(1)
  # CHECK: lit.alias.decl{{.*}}useIfAlias{{.*}}NmStruct{{.*}}2
  alias useIfAlias = NmStruct(2) if True else NmStruct(3)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useIfVar" var : <!NmTarget>
  # CHECK: kgen.param.constant: !NmTarget {{.*}}false
  var useIfVar = NmStruct(2) if True else NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useIfVarLopsided" var : <!NmTarget>
  # CHECK: kgen.param.constant: !NmTarget {{.*}}true
  var useIfVarLopsided = NmTarget(False) if False else NmStruct(77)

  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useOrVar1" var : <!NmTarget>
  var useOrVar1 = NmStruct(2) or NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useOrVar2" var : <!NmTarget>
  var useOrVar2 = NmStruct(2) or NmStruct(3)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useAndVar1" var : <!NmTarget>
  var useAndVar1 = NmStruct(2) and NmStruct(77)
  # CHECK: [[NMDECL:%.*]] lit.varlet.decl "useAndVar2" var : <!NmTarget>
  var useAndVar2 = NmStruct(77) and NmStruct(77)

  # Test that parameter inference using nonmaterializable gives the target,
  # not the nonmaterializable type.
  # CHECK: call {{.*}}tail_types{{.*}}<:type !NmTarget, :variadic<type> []>
  tail_types(NmStruct(5))
  # CHECK: call {{.*}}tail_types{{.*}}<:type !NmTarget, :variadic<type> [{{.*}}NmTarget]>
  tail_types(NmStruct(5), NmStruct(6))
