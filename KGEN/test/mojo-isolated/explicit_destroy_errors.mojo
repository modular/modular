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

    fn consume(deinit self):
        pass


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
    fn __del__(deinit self):
        pass


# CHECK-LABEL: @"foo
# expected-error @below {{Unhandled explicit_destroy type UnknownDestructibility}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
fn foo[T: UnknownDestructibility](var x: T):
    pass


# TODO(MOCO-2363): re-enable the test below
#
# # TODO(MOCO-1468): Require error message for @explicit_destroy
# @explicit_destroy
# trait LinearCopyable(ImplicitlyCopyable):
#     pass


# # C_HECK-LABEL: @"receiveLinearCopyable
# # e_xpected-error @below {{Unhandled explicit_destroy type LinearCopyable}}
# fn receiveLinearCopyable[T: LinearCopyable](var x: T):
#     pass


# @explicit_destroy
# struct LinearCopyableStruct(LinearCopyable):
#     fn __copyinit__(out self, existing: Self, /):
#         pass


# # C_HECK-LABEL: @"upcastLinearCopyable
# fn upcastLinearCopyable(var x: LinearCopyableStruct):
#     receiveLinearCopyable(x)


# CHECK-LABEL: lit.fn @"callsWith
fn callsWith():
    # expected-error @below {{Unhandled explicit_destroy type Coroutine}}
    _ = testAsyncVoid()
    # CHECK-NOT: lit.call {{.*}}__del__

# CHECK-LABEL: lit.fn @"testAsyncVoid
async fn testAsyncVoid():
    pass


# CHECK-LABEL: lit.struct.decl @ExplicitWithDel

# MOCO-2787 - Linear types do not error if they contain an explicit del
@explicit_destroy("must use __del__() explicitly")
struct ExplicitWithDel:
    fn __init__(out self):
        pass

    # Presence of a del shouldn't override @explicit_destroy.
    fn __del__(deinit self):
        pass

    fn method(self): pass

fn testExplicitWithDel():
    a = ExplicitWithDel()
    a.method()
    a^.__del__() # ok

    b = ExplicitWithDel()
    b.method() # expected-error {{'b' abandoned without being explicitly destroyed: must use __del__() explicitly}}


# This comes from stubs library.
# CHECK-LABEL: lit.struct.decl @Coroutine
# CHECK-NOT: destructor :!lit.generator
