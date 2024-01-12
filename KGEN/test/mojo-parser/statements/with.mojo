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
    # CHECK: %0 = kgen.param.constant: !StringLiteral = <#lit.struct<{value: string = "thing"}>>
    # CHECK: %1 = lit.call {{.*}}@"$builtin"::@"$error"::@Error::@"__init__{{.*}}"(%0) : !lit.signature<("value": !StringLiteral borrow) ownedresult -> !Error>
    # CHECK: lit.raise %1 : !Error
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
    # CHECK-NEXT: %$CONTEXTMGR = lit.varlet.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%$CONTEXTMGR, %a)
    # CHECK-NEXT: %val = lit.varlet.decl {{.*}} imp
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
    # CHECK-NEXT: lit.ref.store [[TARGET]], %val
    # CHECK-NEXT: lit.try
    with a as val:
        # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load %val
        # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[VAL]])
        noop(val)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # Test a with with no target.

    # CHECK: %$CONTEXTMGR_0 = lit.varlet.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%$CONTEXTMGR_0, %a)
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK: lit.call {{.*}}__enter__{{.*}}([[IMMREF]]
    # CHECK-NEXT: lit.try
    with a:
        # CHECK-NEXT: kgen.param.constant: {{.*}}42
        # CHECK-NEXT: lit.call {{.*}}noop
        noop(42)
    # CHECK: finally
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR_0
    # CHECK-NEXT: lit.call {{.*}}__exit__{{.*}}([[IMMREF]])

    # CHECK: %$CONTEXTMGR_1 = lit.varlet.decl "$CONTEXTMGR"{{.*}}!MutatingCM
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%$CONTEXTMGR_1)
    # CHECK-NEXT: %val_2 = lit.varlet.decl "val"
    # CHECK-NEXT: lit.call {{.*}}__enter__{{.*}}(%$CONTEXTMGR_1)
    with MutatingCM() as val:
        # CHECK: lit.call {{.*}}noop
        noop(val)
    # CHECK: lit.call {{.*}}__exit__{{.*}}(%$CONTEXTMGR_1)


# CHECK-LABEL: lit.func @"testWithRaising
fn testWithRaising(a: ExampleCM) raises:
    # CHECK: %$CONTEXTMGR = lit.varlet.decl
    # CHECK: %val = lit.varlet.decl {{.*}} imp
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
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
        # CHECK-NEXT:   [[OK:%.*]] = kgen.variant.take [[RESULT]]
        # CHECK-NEXT:   lit.yield [[OK]]
        # CHECK-NEXT: } else {
        # CHECK-NEXT:   kgen.variant.take
        # CHECK-NEXT:   lit.raise
        # CHECK-NEXT:   kgen.unreachable
        # CHECK-NEXT: }
        raise_string()
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } except (%arg0: !Error) {
    # CHECK:        lit.ref.store %false, %__with_exc__
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   [[EXIT_RESULT:%.*]] = lit.call {{.*}}__exit__{{.*}}([[IMMREF]], %arg0)
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
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
    # CHECK-NEXT:   call {{.*}}__exit__{{.*}}([[IMMREF]])


# CHECK-LABEL: lit.func @"testWithInTry
fn testWithInTry(a: ExampleCM):
    # CHECK: lit.try {
    try:
        # CHECK: %$CONTEXTMGR = lit.varlet.decl
        # CHECK: %cm = lit.varlet.decl "cm"
        # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %$CONTEXTMGR
        # CHECK-NEXT: [[TARGET:%.*]] = lit.call {{.*}}__enter__{{.*}}([[IMMREF]])
        # CHECK-NEXT: lit.ref.store [[TARGET]], %cm
        # CHECK: lit.ref.store %true, %__with_exc__
        # CHECK: lit.try {
        with a as cm:
            # CHECK: lit.try {
            # CHECK-NEXT: [[RESULT:%.*]] = lit.call {{.*}}raise_string()
            # CHECK-NEXT: lit.handle_variant [[RESULT]]
            # CHECK-NEXT:   [[OK:%.*]] = kgen.variant.take [[RESULT]]
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
    fn __init__(inout self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    # This context manager consumes itself and returns it as the value.
    fn __enter__(owned self) -> Self:
        return self ^

    fn method(self):
        pass


# CHECK-LABEL: lit.func @"testCMWithoutExit
fn testCMWithoutExit():
    # CHECK: %$CONTEXTMGR = lit.varlet.decl "$CONTEXTMGR"
    # CHECK: %a = lit.varlet.decl
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%a, %$CONTEXTMGR)
    # CHECK-NEXT: lit.try {
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except (%arg0: i1) {
    # CHECK-NEXT:   kgen.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT:   lit.ownership.use %a
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: }
    with CMWithoutExit() as a:
        a.method()

    # CHECK: %$CONTEXTMGR_0 = lit.varlet.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR_0)
    # CHECK: %a_1 = lit.varlet.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%a_1, %$CONTEXTMGR_0)
    # CHECK-NEXT: lit.try {
    # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut %a_1
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except (%arg0: i1) {
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
    # CHECK: %$CONTEXTMGR = lit.varlet.decl "$CONTEXTMGR"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__init__{{.*}}(%$CONTEXTMGR)
    # CHECK: %a = lit.varlet.decl "a"
    # CHECK-NEXT: lit.call {{.*}}@CMWithoutExit::@"__enter__{{.*}}(%a, %$CONTEXTMGR)
    # CHECK-NEXT: lit.try {
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
    # CHECK-NEXT:   lit.call {{.*}}@CMWithoutExit::@"method{{.*}}([[IMMREF]])
    # CHECK-NEXT:   kgen.param.constant: none
    # CHECK-NEXT:   lit.return
    # CHECK-NEXT:   lit.try.yield
    # CHECK-NEXT: } except (%arg0: i1) {
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
