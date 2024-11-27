# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

@explicit_destroy("Must use consume!")
struct EmptyExplicit:
    fn __init__(inout self):
        pass
    fn consume(owned self):
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(self))

fn errorExample():
    # expected-error @below {{Must use consume!}}
    var l = EmptyExplicit()


# expected-error @below {{Must use consume!}}
struct ImplicitlyDestructibleContainerOfExplicitWithAutoDel:
    var m: EmptyExplicit
    fn __init__(inout self):
        self.m = EmptyExplicit()

struct ImplicitlyDestructibleContainerOfExplicitWithIncompleteDel:
    var m: EmptyExplicit
    fn __init__(inout self):
        self.m = EmptyExplicit()
    # expected-error @below {{Must use consume!}}
    fn __del__(owned self):
        pass

# CHECK-LABEL: @"foo
# expected-error @below {{Unhandled explicit_destroy type UnknownDestructibility}}
fn foo[T: UnknownDestructibility](owned x: T):
    pass

# TODO(MOCO-1468): Require error message for @explicit_destroy
@explicit_destroy
trait LinearCopyable:
    fn __copyinit__(out self, existing: Self, /):
        pass

# CHECK-LABEL: @"receiveLinearCopyable
# expected-error @below {{Unhandled explicit_destroy type LinearCopyable}}
fn receiveLinearCopyable[T: LinearCopyable](owned x: T):
    pass

@explicit_destroy
struct LinearCopyableStruct(LinearCopyable):
    fn __copyinit__(out self, existing: Self, /):
        pass

# CHECK-LABEL: @"upcastLinearCopyable
fn upcastLinearCopyable(owned x: LinearCopyableStruct):
    receiveLinearCopyable(x)
