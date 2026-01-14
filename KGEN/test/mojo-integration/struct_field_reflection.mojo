# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from reflection import (
    struct_field_names,
    struct_field_types,
)


trait CanDoSomething:
    # An unsafe default that assumes all fields implement CanDoSomething.
    fn do_something(self):
        comptime names = struct_field_names[Self]()

        @parameter
        for i in range(names.size):
            print(materialize[names[i]](), ": ", sep="", end="")
            trait_downcast[CanDoSomething](
                __struct_field_ref(i, self)
            ).do_something()


@fieldwise_init
struct Overriding(CanDoSomething, ImplicitlyCopyable):
    fn do_something(self):
        print("overriding")


@fieldwise_init
struct Overriding2(CanDoSomething, ImplicitlyCopyable):
    fn do_something(self):
        print("overriding2")


@fieldwise_init
struct WrapperStruct(CanDoSomething):
    var x: Overriding
    var y: Overriding2


fn call_do_something[T: CanDoSomething](ref t: T):
    t.do_something()


fn main():
    var my_struct = WrapperStruct(Overriding(), Overriding2())
    call_do_something(my_struct)
    # CHECK: x: overriding
    # CHECK: y: overriding2
