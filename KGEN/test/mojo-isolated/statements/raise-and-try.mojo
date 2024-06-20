# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo -verify-diagnostics | FileCheck %s

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
    # CHECK-NEXT: except
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


# CHECK-LABEL: lit.func @"tryExceptArgDef
fn tryExceptArgDef():
    try:
        pass
    # CHECK: except
    except err:
        # CHECK-NEXT: [[ERR:%.*]] = lit.ref.load %err
        # CHECK-NEXT: eatError{{.*}}([[ERR]])
        eatError(err)


# CHECK-LABEL: lit.func @"tryFinally
fn tryFinally():
    # CHECK: lit.try
    try:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: except
    # CHECK-NEXT: lit.try.yield
    # CHECK: finally
    finally:
        # CHECK: lit.return
        return
    # CHECK: lit.try %__try_error___0
    try:
        # CHECK: lit.try
        try:
            # CHECK-NEXT: lit.try.yield
            pass
        # CHECK-NEXT: except
        # CHECK-NEXT: [[ERR:%.*]] = lit.load.consume %__try_error__
        # CHECK-NEXT: lit.ref.store [[ERR]], %__try_error__
        # CHECK-NEXT: lit.raise
        finally:
            pass
    except:
        pass


def maybeRaises() -> Int:
    return 0


# CHECK-LABEL: lit.func @"propagateErrorInDef
def propagateErrorInDef():
    # CHECK: %a = lit.var.decl "a"
    # CHECK: lit.call {{.*}}maybeRaises{{.*}}(%__error__, %a)
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInRaisingFn
fn propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.var.decl {{.*}} : !lit.ref<!Int,
    var a: Int
    # CHECK:  lit.call {{.*}}maybeRaises{{.*}}(%__error__, %a)
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInTry
fn propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: %0 = lit.call {{.*}}maybeRaises{{.*}}(%__try_error__, %a)
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.func @"raiseError
def raiseErrorInDef(err: Error):
    # CHECK: lit.call {{.*}}@Error::@"__copyinit__{{.*}}(%__error__, %err)
    # CHECK-NEXT: lit.raise
    raise err


# CHECK-LABEL: lit.func @"raiseErrorInIf
def raiseErrorInIf(cond: Bool, err: Error):
    # CHECK: hlcf.elif
    if cond:
        # CHECK: lit.call {{.*}}@Error::@"__copyinit__{{.*}}(%__error__, %err)
        # CHECK-NEXT: lit.raise
        raise err


# CHECK-LABEL: lit.func @"raiseErrorInTry
fn raiseErrorInTry(err: Error):
    # CHECK: lit.try %__try_error__
    try:
        # CHECK-NEXT: lit.call {{.*}}@Error::@"__copyinit__{{.*}}(%__try_error__, %err)
        # CHECK-NEXT: lit.raise
        raise err
    except:
        pass


# CHECK-LABEL: lit.func @"rethrowsToRethrow
fn rethrowsToRethrow():
    # CHECK: lit.try [[TRY_ERROR1:%.*]] :
    try:
        # CHECK: lit.try [[TRY_ERROR2:%.*]] :
        try:
            # CHECK: lit.call {{.*}}maybeRaises{{.*}}([[TRY_ERROR2]], %anonymous
            maybeRaises()  # expected-warning {{'Int' value is unused}}
        # CHECK: } except {
        except:
            # CHECK: [[ERR:%.*]] = lit.load.consume [[TRY_ERROR2]]
            # CHECK-NEXT: lit.ref.store [[ERR]], [[TRY_ERROR1]]
            # CHECK-NEXT: lit.raise
            raise
        # CHECK: }
    # CHECK: } except {
    except:
        # CHECK: lit.return %none
        return


struct S:
    var v: Int

    fn __init__(inout self, x: Int):
        self.v = x

    fn __init__(inout self) raises:
        self.v = 1

    fn __copyinit__(inout self, existing: Self):
        self.v = existing.v


fn fail() raises -> S:
    return 0


# CHECK-LABEL: lit.func @"call_raising
fn call_raising():
    # CHECK: lit.try %e
    try:
        # CHECK: [[ERR:%.*]] =  lit.call {{.*}}::@"fail{{.*}}(%e, %x)
        var x = fail()
        # CHECK: %y = lit.var.decl "y"
        # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%y, %e)
        var y = S()
        # CHECK-NEXT: lit.try.yield
    except e:
        pass


fn fail_raises() raises -> S:
    return fail()


fn fail_register() raises -> Int:
    return 0


# CHECK-LABEL: lit.func @"fail_register_raises
fn fail_register_raises() raises -> Int:
    # CHECK-NEXT: call {{.*}}fail_register{{.*}}(%__error__, %__result__)
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return fail_register()
