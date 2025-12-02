# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s


trait A:
    pass


trait B:
    pass


trait C:
    pass


struct S(A, B):
    pass


struct S1(C):
    pass


fn foo[T: AnyType]():
    @parameter
    if conforms_to(T, A & B):
        print("T conforms to 'A & B'")
    else:
        print("T does not conform to 'A & B'")


fn bar[T: AnyType]() where conforms_to(T, A & B):
    print("overload A")
    return


fn bar[T: AnyType]() where conforms_to(T, C):
    print("overload B")
    return


fn indirect_target[T: AnyType]() where conforms_to(T, C):
    print("selected indirectly")
    return


fn indirect[T: C]():
    indirect_target[T]()
    return


fn main():
    # CHECK: S conforms to 'A & B'
    @parameter
    if conforms_to(S, A & B):
        print("S conforms to 'A & B'")

    # CHECK: S does not conform to 'C'
    @parameter
    if not conforms_to(S, C):
        print("S does not conform to 'C'")

    # CHECK: T conforms to 'A & B'
    foo[S]()

    # CHECK: T does not conform to 'A & B'
    foo[S1]()

    # CHECK: overload A
    bar[S]()

    # CHECK: overload B
    bar[S1]()

    # CHECK: selected indirectly
    indirect[S1]()
