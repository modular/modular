# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn simple_fn(x: Int):
    print(x)  # simple_fn stop


fn parametrized_fn[T: Writable](x: T):
    print(String.write(x))  # parametrized_fn stop


@fieldwise_init
struct Struct[T1: Writable]:
    fn parametrized_method[T2: Writable](self, x: Self.T1, y: T2):
        print(String.write(x))  # parametrized_method stop
        print(String.write(y))


fn main():
    print("start")  # breakpoint
    simple_fn(12)
    parametrized_fn[Int](13)
    Struct[Float32]().parametrized_method[Int](12.25, 13)
