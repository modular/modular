# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from std.builtin.rebind import downcast
from std.reflection import (
    struct_field_names,
    struct_field_ref,
    struct_field_types,
)


trait CanDoSomething:
    # An unsafe default that assumes all fields implement CanDoSomething.
    def do_something(self):
        comptime names = struct_field_names[Self]()

        comptime for i in range(names.size):
            print(materialize[names[i]](), ": ", sep="", end="")
            trait_downcast[CanDoSomething](
                struct_field_ref[i](self)
            ).do_something()


@fieldwise_init
struct Overriding(CanDoSomething, ImplicitlyCopyable):
    def do_something(self):
        print("overriding")


@fieldwise_init
struct Overriding2(CanDoSomething, ImplicitlyCopyable):
    def do_something(self):
        print("overriding2")


@fieldwise_init
struct WrapperStruct(CanDoSomething):
    var x: Overriding
    var y: Overriding2


def call_do_something[T: CanDoSomething](ref t: T):
    t.do_something()


def closure_fields():
    var a: Int32 = 42
    var b: Int32 = 27

    def test() {var a, var b} -> Int32:
        return a + b

    # COM: reset `b` value to be 31
    trait_downcast[TrivialRegisterPassable](
        __struct_field_ref(1, __struct_field_ref(0, test))
    ) = rebind[
        type_of(
            trait_downcast[TrivialRegisterPassable](
                __struct_field_ref(1, __struct_field_ref(0, test))
            )
        )
    ](
        Int32(31)
    )

    print("closure_fields: ", test())

    # COM: reset `b` value to be 100
    trait_downcast[TrivialRegisterPassable](
        __struct_field_ref(1, __struct_field_ref(0, test))
    ) = rebind[
        type_of(
            trait_downcast[TrivialRegisterPassable](
                __struct_field_ref(1, __struct_field_ref(0, test))
            )
        )
    ](
        Int32(100)
    )

    print("closure_fields: ", test())


def main():
    var my_struct = WrapperStruct(Overriding(), Overriding2())
    call_do_something(my_struct)
    # CHECK: x: overriding
    # CHECK: y: overriding2

    closure_fields()
    # CHECK: closure_fields: 73
    # CHECK: closure_fields: 142
