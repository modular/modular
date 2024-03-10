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
        # CHECK-NEXT: lit.var.decl "err" imp
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
    # CHECK: %a = lit.var.decl "a"
    # CHECK: %[[VALUE:.*]] = lit.call @"{{.*}}"::@"maybeRaises
    # CHECK: %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
    # CHECK: {
    # CHECK:    [[VAR:%.*]] = kgen.variant.take %0, 1 : <!Error, !Int>
    # CHECK:    lit.yield [[VAR]] : !Int
    # CHECK: } else {
    # CHECK:    [[ERR:%.*]] = kgen.variant.take %0, 0 : <!Error, !Int>
    # CHECK:    lit.raise [[ERR]] : !Error
    # CHECK:    kgen.unreachable
    # CHECK:  }
    # CHECK-NEXT: lit.ref.store %1, %a
    a = maybeRaises()


# CHECK-LABEL: lit.func @"propagateErrorInRaisingFn
fn propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.var.decl {{.*}} : !lit.ref<!Int,
    var a: Int
    # CHECK:  %0 = lit.call {{.*}}::@"maybeRaises()"()
    # CHECK:  %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
    # CHECK:  {
    # CHECK:    [[ERR:%.*]] = kgen.variant.take %0
    # CHECK:    lit.yield [[ERR]] : !Int
    # CHECK:  } else {
    # CHECK:    [[ERR:%.*]] = kgen.variant.take %0
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
        # CHECK: %0 = lit.call {{.*}}::@"maybeRaises()"()
        # CHECK: %1 = lit.handle_variant %0 : (!kgen.variant<!Error, !Int>) -> !Int
        # CHECK: {
        # CHECK: } else {
        # CHECK:   [[ERR:%.*]] = kgen.variant.take %0
        # CHECK:   lit.raise [[ERR]] : !Error
        # CHECK: }

        # CHECK-NEXT: lit.ref.store %1, %a
        a = maybeRaises()
        # CHECK-NEXT: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.func @"raiseError
def raiseErrorInDef(err: Error):
    # CHECK: %err_0 = lit.var.decl "err"
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
            # CHECK:  lit.call {{.*}}::@"maybeRaises()"()
            maybeRaises()  # expected-warning {{'Int' value is unused}}
        # CHECK: } except (%arg0:
        except:
            # CHECK: lit.raise %arg0
            raise
        # CHECK: }
    # CHECK: } except (%arg0: !Error)
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


fn fail(str: StringRef) raises -> S:
    return 0


# CHECK-LABEL: lit.func @"call_raising
fn call_raising():
    # CHECK-NEXT: lit.try {
    try:
        # CHECK: [[ERR:%.*]] =  lit.call {{.*}}::@"fail
        # CHECK: [[VAR0:%.*]] = lit.handle_variant [[ERR]], %x
        # CHECK:   [[VAR1:%.*]] = kgen.variant.take [[ERR]]
        # CHECK:   lit.yield [[VAR1]] : !kgen.none
        # CHECK: } else {
        # CHECK:   [[VAR2:%.*]] = kgen.variant.take [[ERR]]
        # CHECK:   lit.raise [[VAR2]]
        # CHECK:   kgen.unreachable
        # CHECK: }
        var x = fail("hello world")
        # CHECK: %y = lit.var.decl "y"
        # CHECK: lit.call @{{.*}}__init__{{.*}}(%y)
        # CHECK: [[VAR1:%.*]] = lit.handle_variant [[ERR:.*]], %y
        # CHECK:   [[VAR2:%.*]] = kgen.variant.take [[ERR]]
        # CHECK:   lit.yield [[VAR2]] : !kgen.none
        # CHECK: } else {
        # CHECK:   [[VAR2:%.*]] = kgen.variant.take [[ERR]]
        # CHECK:   lit.raise [[VAR2]]
        # CHECK:   kgen.unreachable
        # CHECK: }
        var y = S()
    except e:
        pass


fn fail_raises(str: StringRef) raises -> S:
    return fail(str)


fn fail_register(str: StringRef) raises -> Int:
    return 0


fn fail_register_raises(str: StringRef) raises -> Int:
    # CHECK: %[[VAR0:.*]] = lit.handle_variant %0
    # CHECK:   %[[VAR1:.*]] = kgen.variant.take %0
    # CHECK:   lit.yield %[[VAR1]]
    # CHECK: } else {
    # CHECK:   %[[VAR2:.*]] = kgen.variant.take %0
    # CHECK:   lit.raise %[[VAR2]]
    # CHECK:   kgen.unreachable
    # CHECK: }
    return fail_register(str)
