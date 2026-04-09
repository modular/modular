# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | FileCheck %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

from std.builtin.coroutine import Coroutine, RaisingCoroutine, AnyCoroutine


@explicit_destroy("Must use consume!")
struct EmptyExplicit:
    def __init__(out self):
        pass

    def consume(deinit self):
        pass


def errorExample():
    # expected-error @below {{Must use consume!}}
    _ = EmptyExplicit()


# expected-error @below {{Must use consume!}}
struct ImplicitlyDestructibleContainerOfExplicitWithAutoDel:
    var m: EmptyExplicit

    def __init__(out self):
        self.m = EmptyExplicit()


struct ImplicitlyDestructibleContainerOfExplicitWithIncompleteDel:
    var m: EmptyExplicit

    def __init__(out self):
        self.m = EmptyExplicit()

    # expected-error @below {{Must use consume!}}
    def __del__(deinit self):
        pass


# CHECK-LABEL: @"test_any_type_error
# expected-error @below {{unhandled explicitly destroyed type 'AnyType'}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def test_any_type_error[T: AnyType](var x: T):
    pass


# TODO(MOCO-2363): re-enable the test below
#
# # TODO(MOCO-1468): Require error message for @explicit_destroy
# @explicit_destroy
# trait LinearCopyable(ImplicitlyCopyable):
#     pass


# # C_HECK-LABEL: @"receiveLinearCopyable
# # e_xpected-error @below {{Unhandled explicit_destroy type LinearCopyable}}
# def receiveLinearCopyable[T: LinearCopyable](var x: T):
#     pass


# @explicit_destroy
# struct LinearCopyableStruct(LinearCopyable):
#     def __init__(out self, *, copy: Self):
#         pass


# # C_HECK-LABEL: @"upcastLinearCopyable
# def upcastLinearCopyable(var x: LinearCopyableStruct):
#     receiveLinearCopyable(x)


# CHECK-LABEL: lit.fn @"callsWith
def callsWith():
    # expected-error @below {{Unhandled explicit_destroy type Coroutine}}
    _ = testAsyncVoid()
    # CHECK-NOT: lit.call {{.*}}__del__


# CHECK-LABEL: lit.fn @"testAsyncVoid
async def testAsyncVoid():
    pass


# CHECK-LABEL: lit.struct.decl @ExplicitWithDel


# MOCO-2787 - Linear types do not error if they contain an explicit del
@explicit_destroy("must use __del__() explicitly")
struct ExplicitWithDel:
    def __init__(out self):
        pass

    # Presence of a del shouldn't override @explicit_destroy.
    def __del__(deinit self):
        pass

    def method(self):
        pass


def testExplicitWithDel():
    a = ExplicitWithDel()
    a.method()
    a^.__del__()  # ok

    b = ExplicitWithDel()
    b.method()  # expected-error {{'b' abandoned without being explicitly destroyed: must use __del__() explicitly}}


# This comes from stubs library.
# CHECK-LABEL: lit.struct.decl @Coroutine
# CHECK-NOT: destructor :!lit.generator


# ===----------------------------------------------------------------------=== #
# Trait with custom @explicit_destroy error message
# ===----------------------------------------------------------------------=== #


# Trait without @explicit_destroy and without ImplicitlyDestructible
trait PlainTrait:
    def do_something(self):
        ...


@explicit_destroy
trait ExplicitDestroyNoMessage:
    def destroy_no_msg(deinit self):
        ...


@explicit_destroy("Use `destroy()` method.")
trait ExplicitDestroyWithMessage:
    def destroy(deinit self):
        ...


# Test: Plain trait without @explicit_destroy
# expected-error @below {{unhandled explicitly destroyed type 'PlainTrait'}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def take_plain_trait[T: PlainTrait](var value: T):
    pass


# Test: Trait with @explicit_destroy but no custom message
# expected-error @below {{Unhandled explicit_destroy type ExplicitDestroyNoMessage}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def take_generic_linear_no_message[T: ExplicitDestroyNoMessage](var value: T):
    pass


# Test: Trait with @explicit_destroy("...") custom message
def take_generic_linear_with_message[
    T: ExplicitDestroyWithMessage
    # expected-error @below {{Use `destroy()` method.}}
    # expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
](var value: T):
    pass


# ===----------------------------------------------------------------------=== #
# Trait composition with @explicit_destroy
# ===----------------------------------------------------------------------=== #


@explicit_destroy("Use custom_foo_destroy().")
trait LinearFoo:
    def custom_foo_destroy(deinit self):
        ...


@explicit_destroy("Use custom_bar_destroy().")
trait LinearBar:
    def custom_bar_destroy(deinit self):
        ...


# Test: First trait has custom message - uses that message
# expected-error @below {{Use custom_foo_destroy().}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def take_foo_and_bar[T: LinearFoo & LinearBar](var value: T):
    pass


# Test: First trait has no custom message - uses generic "Unhandled" message
# (Documents current behavior where iteration order matters)
# expected-error @below {{Unhandled explicit_destroy type ExplicitDestroyNoMessage}}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def take_no_msg_first[T: ExplicitDestroyNoMessage & LinearBar](var value: T):
    pass


# ===----------------------------------------------------------------------=== #
# @explicit_destroy("") on linear traits (valid but poor form)
# ===----------------------------------------------------------------------=== #


# Test: Empty string message is valid on a linear trait (trait without
# ImplicitlyDestructible). The empty message will be used as the error.
@explicit_destroy("")
trait LinearWithEmptyMessage:
    def consume(deinit self):
        ...


# expected-error @below {{abandoned without being explicitly destroyed: }}
# expected-note @below {{consider adding trait conformance to ImplicitlyDestructible}}
def take_linear_empty_message[T: LinearWithEmptyMessage](var value: T):
    pass
