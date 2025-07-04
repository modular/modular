# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | FileCheck %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

from builtin.coroutine import Coroutine, RaisingCoroutine, AnyCoroutine


@explicit_destroy("Must use consume!")
struct EmptyExplicit:
    fn __init__(out self):
        pass
    fn consume(var self):
        __disable_del self

fn errorExample():
    # expected-error @below {{Must use consume!}}
    _ = EmptyExplicit()


# expected-error @below {{Must use consume!}}
struct ImplicitlyDestructibleContainerOfExplicitWithAutoDel:
    var m: EmptyExplicit
    fn __init__(out self):
        self.m = EmptyExplicit()

struct ImplicitlyDestructibleContainerOfExplicitWithIncompleteDel:
    var m: EmptyExplicit
    fn __init__(out self):
        self.m = EmptyExplicit()
    # expected-error @below {{Must use consume!}}
    fn __del__(owned self):
        pass

# CHECK-LABEL: @"foo
# expected-error @below {{Unhandled explicit_destroy type UnknownDestructibility}}
fn foo[T: UnknownDestructibility](var x: T):
    pass


# TODO(MOCO-1468): Require error message for @explicit_destroy
@explicit_destroy
trait LinearCopyable:
    fn __copyinit__(out self, existing: Self, /):
        pass

# CHECK-LABEL: @"receiveLinearCopyable
# expected-error @below {{Unhandled explicit_destroy type LinearCopyable}}
fn receiveLinearCopyable[T: LinearCopyable](var x: T):
    pass

@explicit_destroy
struct LinearCopyableStruct(LinearCopyable):
    fn __copyinit__(out self, existing: Self, /):
        pass

# CHECK-LABEL: @"upcastLinearCopyable
fn upcastLinearCopyable(var x: LinearCopyableStruct):
    receiveLinearCopyable(x)


# CHECK-LABEL: lit.fn @"callsWith
fn callsWith():
  # expected-error @below {{Unhandled explicit_destroy type Coroutine}}
  _ = testAsyncVoid()
  # CHECK-NOT: lit.call {{.*}}__del__


# CHECK-LABEL: lit.struct.decl @Coroutine
# CHECK-NOT: destructor :!lit.generator
async fn testAsyncVoid(): pass
