# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo -verify-diagnostics | FileCheck %s

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##


# Issue #12358
# CHECK-LABEL: lit.func @"raise_string
fn raise_string() raises:
    # CHECK: %0 = kgen.param.constant: !StringLiteral = <{:string "thing"}>
    # CHECK: [[ERR:%.*]] = lit.call {{.*}}@error::@Error::@"__init__{{.*}}"(%0) : !lit.signature<("value": !StringLiteral borrow) -> !Error>
    # CHECK: lit.ref.store [[ERR]], %__error__
    # CHECK-NEXT: lit.raise
    raise "thing"


struct ExampleCM:
    fn __copyinit__(inout self, existing: Self):
        pass

    fn __enter__(self) -> Int:
        return 42

    fn __exit__(self):
        pass  # normal

    fn __exit__(self, err: Error) -> Bool:
        return True  # Raise


# Cannot use mutating __enter__
# https://github.com/modularml/modular/issues/27371
struct MutatingCM:
    fn __init__(inout self):
        pass

    fn __enter__(inout self) -> Int:
        return 42

    fn __exit__(inout self):
        pass  # normal


fn noop(a: Int):
    pass


# CHECK-LABEL: lit.func @"testWithNonRaising
fn testWithNonRaising(a: ExampleCM):
    # CHECK-NEXT: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %val = lit.var.decl {{.*}} imp
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
    # CHECK-NEXT: lit.ref.store [[TARGET]], %val
    # CHECK-NEXT: %__with_error__
    # CHECK-NEXT: lit.try %__with_error__
    with a as val:
        # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load %val
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
        noop(val)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # Test a with with no target.

    # CHECK: %$CONTEXTMGR_0 = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%$CONTEXTMGR_0, %a)
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK: lit.call {{.*}}__enter__{{.*}}([[IMMREF]]
    # CHECK: lit.try
    with a:
        # CHECK-NEXT: kgen.param.constant: {{.*}}42
        # CHECK-NEXT: lit.call {{.*}}noop
        noop(42)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # CHECK: [[MGR:%.*]] = lit.var.decl "$CONTEXTMGR"{{.*}}!MutatingCM
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[MGR]])
    # CHECK-NEXT: %val{{.*}} = lit.var.decl "val"
    # CHECK-NEXT: lit.call {{.*}}__enter__{{.*}}([[MGR]])
    with MutatingCM() as val:
        # CHECK: lit.call {{.*}}noop
        noop(val)
    # CHECK: lit.call {{.*}}__exit__{{.*}}([[MGR]])


# CHECK-LABEL: lit.func @"testWithRaising
fn testWithRaising(a: ExampleCM) raises:
    # CHECK: %$CONTEXTMGR = lit.var.decl
    # CHECK: %val = lit.var.decl {{.*}} imp
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
    # CHECK-NEXT: lit.ref.store [[TARGET]], %val
    # CHECK: lit.ref.store %true, %__with_exc__
    # CHECK: lit.try %__with_error__
    # CHECK: lit.try %__inner_error__
    with a as val:
        # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load %val
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
        noop(val)

        # CHECK: [[RESULT:%.*]] = lit.call {{.*}}raise_string{{.*}}(%__inner_error__, %anonymous
        raise_string()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK:        lit.ref.store %false, %__with_exc__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   [[ERROR:%.*]] = lit.ref.load %__inner_error__
    # CHECK-NEXT:   [[EXIT_RESULT:%.*]] = lit.call {{.*}}__exit__{{.*}}([[IMMREF]], [[ERROR]])
    # CHECK-NEXT:   [[SUCCESS:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[EXIT_RESULT]])
    # CHECK-NEXT:   hlcf.if [[SUCCESS]] {
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     [[ERROR:%.*]] = lit.load.consume %__inner_error__
    # CHECK-NEXT:     lit.ref.store [[ERROR]], %__with_error__
    # CHECK-NEXT:     lit.raise
    # CHECK-NEXT:     hlcf.yield
    # CHECK-NEXT:   }
    # CHECK-NEXT:   lit.try.yield
    # CHECK:      } finally {
    # CHECK:    } except
    # CHECK-NEXT: [[ERROR:%.*]] = lit.load.consume %__with_error__
    # CHECK-NEXT: lit.ref.store [[ERROR]], %__error__
    # CHECK:    } finally {
    # CHECK-NEXT: %[[EXC:.*]] = lit.ref.load %__with_exc__
    # CHECK-NEXT: hlcf.if %[[EXC]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   call {{.*}}__exit__{{.*}}([[IMMREF]])


# CHECK-LABEL: lit.func @"testWithInTry
fn testWithInTry(a: ExampleCM):
    # CHECK: %e = lit.var.decl "e" var
    # CHECK-NEXT: lit.try %e
    try:
        # CHECK: %$CONTEXTMGR = lit.var.decl
        # CHECK: %cm = lit.var.decl "cm"
        # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
        # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
        # CHECK-NEXT: lit.ref.store [[TARGET]], %cm
        # CHECK: lit.ref.store %true, %__with_exc__
        # CHECK-NEXT: lit.try %__with_error__
        with a as cm:
            # CHECK: %__inner_error__ = lit.var.decl
            # CHECK: lit.try %__inner_error__
            # CHECK: [[RESULT:%.*]] = lit.call {{.*}}raise_string{{.*}}(%__inner_error__, %anonymous
            raise_string()
    except e:
        _ = e


# CHECK-LABEL: lit.func @"testWithScoping
fn testWithScoping(a: ExampleCM):
    # This is a test that issue #18811 is fixed, in which a `with`
    # statement inside a `fn` does not respect lexical scope and binds
    # its variable in its parent scope.
    with a as withDecl:
        # CHECK: %withDecl = lit.var.decl "withDecl" imp
        noop(withDecl)
    with a as withDecl:
        # CHECK: = lit.var.decl "withDecl" imp
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
    fn __init__(inout self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    # This context manager consumes itself and returns it as the value.
    fn __enter__(owned self) -> Self:
        return self^

    fn method(self):
        pass


# CHECK-LABEL: lit.func @"testCMWithoutExit
fn testCMWithoutExit():
    # CHECK: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK: %a = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %__with_error__ = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    with CMWithoutExit() as a:
        a.method()

    # CHECK: %$CONTEXTMGR_0 = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR_0)
    # CHECK: %a_1 = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR_0, %a_1)
    # CHECK-NEXT: %__with_error__{{.*}} = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a_1
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a_1
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
    # CHECK: %$CONTEXTMGR = lit.var.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR)
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %__with_error__ = lit.var.decl "__with_error__" synth
    # CHECK-NEXT: lit.try %__with_error__
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   kgen.param.constant: none
    # CHECK-NEXT:   lit.return
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    with CMWithoutExit() as a:
        a.method()
        return
