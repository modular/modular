# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


@value
struct SomeStruct:
    var x: Int


fn take_closure_and_print(
    g: fn (Int) capturing -> Int,
    x: Int,
):
    print(g(x))


fn test_take_closure_and_print(x: Int):
    var v = SomeStruct(x)

    @parameter
    fn FOO(y: Int) -> Int:
        print(v.x)
        return y

    v.x = 5

    let capture_struct_closure: fn (Int) capturing -> Int = FOO
    let u: Int = 3
    take_closure_and_print(capture_struct_closure, u)


fn main():
    let x = 39
    # CHECK: 5
    # CHECK: 3
    test_take_closure_and_print(x)
