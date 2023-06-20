# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo -verify-diagnostics -I %S/../mojo-examples/ | FileCheck %s


from prolog import object, range

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
    # CHECK: lit.return %1 : !pop.variant<@{{.*}}::@Error, @"$Int"::@Int>
    return 4  # Implicit conversion from literal to Int


##===----------------------------------------------------------------------===##
# If
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_if
fn test_if(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK:          hlcf.if
    if a:
        # CHECK-NEXT: %inside_a = lit.varlet.decl "inside_a", var = true
        var inside_a: Int
    # CHECK:          } else {
    # CHECK:            hlcf.if
    elif b:
        # CHECK-NEXT: %inside_b = lit.varlet.decl "inside_b", var = true
        var inside_b: Int
    # CHECK:            } else {
    # CHECK:              hlcf.if
    elif c:
        # CHECK-NEXT: %inside_c = lit.varlet.decl "inside_c", var = true
        var inside_c: Int
    # CHECK:              } else {
    else:
        # CHECK-NEXT: %inside_else = lit.varlet.decl "inside_else", var = true
        var inside_else: Int
    # CHECK:                hlcf.yield
    # CHECK-NEXT:         }
    # CHECK-NEXT:         hlcf.yield
    # CHECK-NEXT:       }
    # CHECK-NEXT:       hlcf.yield
    # CHECK-NEXT:     }
    var z: Int = 4
    return a


# CHECK-LABEL: lit.func @"test_if_nested
fn test_if_nested(a: Bool, b: Bool, c: Bool) -> Bool:
    # CHECK-NEXT:   %[[I1:.*]] = kgen.call @"$Bool"::@Bool::@"__mlir_i1__($Bool::Bool)"(%a)
    # CHECK-NEXT:              hlcf.if %[[I1]]
    if a:
        # CHECK-NEXT: %inside_a = lit.varlet.decl "inside_a", var = true
        var inside_a: Int
    # CHECK:                   } else {
    # CHECK:                     hlcf.if
    else:
        if b:
            # CHECK-NEXT: %inside_b = lit.varlet.decl "inside_b", var = true
            var inside_b: Int
        # CHECK:                     } else {
        # CHECK:                       hlcf.if
        else:
            if c:
                # CHECK-NEXT: %inside_c = lit.varlet.decl "inside_c", var = true
                var inside_c: Int
            # CHECK:                       } else {
            else:
                # CHECK-NEXT: %inside_else = lit.varlet.decl "inside_else", var = true
                var inside_else: Int
    # CHECK:                         hlcf.yield
    # CHECK:                       }
    # CHECK:                       hlcf.yield
    # CHECK-NEXT:               }
    # CHECK:                    hlcf.yield
    # CHECK-NEXT:             }
    var z: Int = 4
    return a

# CHECK-LABEL: lit.func @"param_if{{.*}}<a: i1, b: @"$Bool"::@Bool>()
fn param_if[a: __mlir_type.i1, b: Bool]():
  # CHECK: kgen.param.if <a> {
  @parameter
  if a:
    # CHECK: lit.varlet.decl "inside_1", var = true
    var inside_1: Int
  # CHECK: } else {
  # CHECK:     kgen.param.if <apply{{.*}}@"$Bool"::@Bool::@"__mlir_i1__{{.*}}b)> {
  elif b:
  # CHECK:     lit.varlet.decl "inside_2", var = true
    var inside_2: Int
  # CHECK:     kgen.param.yield
  # CHECK:   }
  # CHECK:   kgen.param.yield
  # CHECK: }

#CHECK-LABEL: lit.func @"param_if_andor_i1[__mlir_type.i1,__mlir_type.i1]()"<a: i1, b: i1>()
fn param_if_andor_i1[a: __mlir_type.i1, b: __mlir_type.i1]():
  # CHECK: kgen.param.if <cond(a, b, a)>
  @parameter
  if a and b:
  # CHECK:   lit.varlet.decl "v", var = true
    var v: Int
  # CHECK:   kgen.param.yield
  # CHECK: } else {
  # CHECK: kgen.param.if <cond(a, a, b)>
  elif a or b:
  # CHECK:   lit.varlet.decl "w", var = true
    var w: Int


#CHECK-LABEL: lit.func @"param_if_and[$Bool::Bool,$Bool::Bool]()"<a: @"$Bool"::@Bool, b: @"$Bool"::@Bool>()
fn param_if_and[a: Bool, b: Bool]():
  # CHECK: kgen.param.if <apply(
  # CHECK-SAME:   :<>(!kgen.declref<@"$Bool"::@Bool> borrow) -> i1 @"$Bool"::@Bool::@"__mlir_i1__($Bool::Bool)",
  # CHECK-SAME:   cond(apply(:<>(!kgen.declref<@"$Bool"::@Bool> borrow) -> i1 @"$Bool"::@Bool::@"__mlir_i1__($Bool::Bool)", a), b, a))> {
  @parameter
  if a and b:
  # CHECK:   lit.varlet.decl "v", var = true
    var v: Int
  # CHECK:   kgen.param.yield
  # CHECK: }

##===----------------------------------------------------------------------===##
# While
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_while
# CHECK:       %inside_a = lit.varlet.decl "inside_a", var = true
# CHECK:       %inside_b = lit.varlet.decl "inside_b", var = true
# CHECK:       %inside_else = lit.varlet.decl "inside_else", var = true
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
    # CHECK: hlcf.if {{.+}} {
    if b:
        # CHECK: hlcf.loop
        # CHECK: {{.+}} = pop.load %a_0
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


# CHECK-LABEL: lit.func @"main()"
fn main():
    let my_list = MyList()

    # CHECK: %$RANGE = lit.varlet.decl "$RANGE"
    # CHECK: %[[ITER:.*]] = kgen.call @{{.*}}__iter__{{.*}}(%$RANGE, %my_list)
    for item in my_list:
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
    # CHECK-NEXT: except (%{{.*}}: !kgen.declref<@{{.*}}::@Error>)
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
    # CHECK: except (%arg0: !kgen.declref<@{{.*}}::@Error>)
    except err:
        # CHECK-NEXT: kgen.call @"{{.*}}::@"eatError{{.*}}(%arg0)
        eatError(err)


# CHECK-LABEL: lit.func @"tryExceptArgDef
def tryExceptArgDef():
    try:
        pass
    # CHECK: except (%arg0: !kgen.declref<@{{.*}}::@Error>)
    except err:
        # CHECK-NEXT: lit.varlet.decl "err", var = true
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
    # CHECK: %1 = lit.handle_variant %0 : (!pop.variant<@"$Error"::@Error, @"$Int"::@Int>) -> !kgen.declref<@"$Int"::@Int>
    # CHECK: {
    # CHECK:    [[VAR:%.*]] = pop.variant.get %0 : !pop.variant<@{{.*}}::@Error, @"$Int"::@Int> as !kgen.declref<@"$Int"::@Int>
    # CHECK:    lit.yield [[VAR]] : !kgen.declref<@"$Int"::@Int>
    # CHECK: } else {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0 : !pop.variant<@{{.*}}::@Error, @"$Int"::@Int> as !kgen.declref<@{{.*}}::@Error>
    # CHECK:    lit.raise [[ERR]] : <@{{.*}}::@Error>
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK-NEXT: pop.store %1, %a
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInRaisingFn
fn propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.varlet.decl {{.*}} : <@"$Int"::@Int>
    var a: Int
    # CHECK:  %0 = kgen.call @"$statements"::@"maybeRaises()"() : () throws -> !pop.variant<@{{.*}}::@Error, @"$Int"::@Int>
    # CHECK:  %1 = lit.handle_variant %0 : (!pop.variant<@"$Error"::@Error, @"$Int"::@Int>) -> !kgen.declref<@"$Int"::@Int>
    # CHECK:  {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0
    # CHECK:    lit.yield [[ERR]] : !kgen.declref<@"$Int"::@Int>
    # CHECK:  } else {
    # CHECK:    [[ERR:%.*]] = pop.variant.get %0
    # CHECK:    lit.raise [[ERR]] : <@"$Error"::@Error>
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK:  pop.store %1, %a
    a = maybeRaises()

# CHECK-LABEL: lit.func @"propagateErrorInTry
fn propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: %1 = kgen.call @"$statements"::@"maybeRaises()"() : () throws -> !pop.variant<@{{.*}}::@Error, @"$Int"::@Int>
        # CHECK: %2 = lit.handle_variant %1 : (!pop.variant<@"$Error"::@Error, @"$Int"::@Int>) -> !kgen.declref<@"$Int"::@Int>
        # CHECK: {
        # CHECK: } else {
        # CHECK:   [[ERR:%.*]] = pop.variant.get %1
        # CHECK:   lit.raise [[ERR]] : <@{{.*}}::@Error>
        # CHECK: }

        # CHECK-NEXT: pop.store %2, %a
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.func @"raiseError
def raiseErrorInDef(err: Error):
    # CHECK: %[[ERRVAL:.*]] = pop.load %err_0
    # CHECK: %[[ERRVALCOPY:.*]] = kgen.call {{.*}}@Error::@"__copyinit__
    # CHECK: lit.raise %[[ERRVALCOPY]] : <@{{.*}}@Error>
    raise err


# CHECK-LABEL: lit.func @"raiseErrorInIf
def raiseErrorInIf(cond: Bool, err: Error):
    # CHECK: hlcf.if
    if cond:
        # CHECK: lit.raise {{.*}} : <@{{.*}}::@Error>
        raise err


# CHECK-LABEL: lit.func @"raiseErrorInTry
fn raiseErrorInTry(err: Error):
    # CHECK: lit.try {
    try:
        # CHECK-NEXT: = kgen.call {{.*}}@Error::@"__copyinit__
        # CHECK-NEXT: lit.raise {{.*}} : <@"$Error"::@Error>
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
    # CHECK: } except (%arg0: !k
    except:
        # CHECK: lit.return %0
        return

# Issue #12358
# CHECK-LABEL: lit.func @"raise_string
fn raise_string() raises:
   # CHECK-NEXT: %0 = kgen.param.constant: @"$StringLiteral"::@StringLiteral = <#lit.struct<{value: string = "thing"}>>
   # CHECK-NEXT: %1 = kgen.call @"$Error"::@Error::@"__init__($StringLiteral::StringLiteral)"(%0) : (!kgen.declref<@"$StringLiteral"::@StringLiteral> borrow) ownedresult -> !kgen.declref<@"$Error"::@Error>
   # CHECK-NEXT: lit.raise %1 : <@"$Error"::@Error>
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
    # CHECK: %[[VAR0:.*]] = lit.handle_variant %3, %x : (!pop.variant<@"$Error"::@Error, !lit.none>, !pop.pointer<@"$statements"::@S>) -> !lit.none {
    # CHECK:       %[[VAR1:.*]] = pop.variant.get %3 : !pop.variant<@"$Error"::@Error, !lit.none> as !lit.none
    # CHECK:       lit.yield %[[VAR1]] : !lit.none
    # CHECK:     } else {
    # CHECK:      %[[VAR2:.*]] = pop.variant.get %3 : !pop.variant<@"$Error"::@Error, !lit.none> as !kgen.declref<@"$Error"::@Error>
    # CHECK:       lit.raise %[[VAR2]] : <@"$Error"::@Error>
    # CHECK:       kgen.unreachable
    # CHECK:     }
    let x = fail("hello world")
    # CHECK:  %[[VAR1:.*]] = lit.handle_variant %5, %y : (!pop.variant<@"$Error"::@Error, !lit.none>, !pop.pointer<@"$statements"::@S>) -> !lit.none {
    # CHECK:    %[[VAR2:.*]] = pop.variant.get %5 : !pop.variant<@"$Error"::@Error, !lit.none> as !lit.none
    # CHECK:    lit.yield %[[VAR2]] : !lit.none
    # CHECK:  } else {
    # CHECK:    %[[VAR2:.*]] = pop.variant.get %5 : !pop.variant<@"$Error"::@Error, !lit.none> as !kgen.declref<@"$Error"::@Error>
    # CHECK:    lit.raise %[[VAR2]] : <@"$Error"::@Error>
    # CHECK:    kgen.unreachable
    # CHECK:  }
    let y = S()
  except e:
    pass


fn fail_raises(str: StringRef) raises -> S:
  return fail(str)


fn fail_register(str: StringRef) raises -> Int:
  return 0


fn fail_register_raises(str: StringRef) raises -> Int:
  # CHECK: %[[VAR0:.*]] = lit.handle_variant %0 : (!pop.variant<@"$Error"::@Error, @"$Int"::@Int>) -> !kgen.declref<@"$Int"::@Int> {
  # CHECK:     %[[VAR1:.*]] = pop.variant.get %0 : !pop.variant<@"$Error"::@Error, @"$Int"::@Int> as !kgen.declref<@"$Int"::@Int>
  # CHECK:     lit.yield %[[VAR1]] : !kgen.declref<@"$Int"::@Int>
  # CHECK:   } else {
  # CHECK:     %[[VAR2:.*]] = pop.variant.get %0 : !pop.variant<@"$Error"::@Error, @"$Int"::@Int> as !kgen.declref<@"$Error"::@Error>
  # CHECK:     lit.raise %[[VAR2]] : <@"$Error"::@Error>
  # CHECK:     kgen.unreachable
  # CHECK:   }
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
  # CHECK: pop.store %true, %__with_exc__ : !pop.pointer<i1>
  # CHECK-NEXT: lit.try
  # CHECK-NEXT: lit.try
  with a as val:
    # CHECK-NEXT: [[VAL:%.*]] = pop.load %val
    # CHECK-NEXT: kgen.call {{.*}}noop{{.*}}([[VAL]])
    noop(val)

    # CHECK-NEXT: %5 = kgen.call {{.*}}raise_string()
    # CHECK-NEXT: %6 = lit.handle_variant %5 : (!pop.variant<@"$Error"::@Error, !lit.none>) -> !lit.none {
    # CHECK-NEXT: %7 = pop.variant.get %5 : !pop.variant<@"$Error"::@Error, !lit.none> as !lit.none
    # CHECK-NEXT: lit.yield %7 : !lit.none
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   pop.variant.get
    # CHECK-NEXT:   lit.raise
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: }
    raise_string()
    # CHECK-NEXT: lit.try.yield
  # CHECK-NEXT: } except (%arg0: !kgen.declref<@"$Error"::@Error>) {
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


##===----------------------------------------------------------------------===##

# TODO(Issue #6139)

# struct Iterable:

# fn test_for(iterable: Iterable):
#  var result = 0
#  for i in iterable:
#    result += i
