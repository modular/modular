# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn simple_fn(x: Int):
    print(x)  # simple_fn stop


fn parametrized_fn[T: Stringable](x: T):
    print(String(x))  # parametrized_fn stop


@fieldwise_init
struct Struct[T1: Stringable]:
    fn parametrized_method[T2: Stringable](self, x: T1, y: T2):
        print(String(x))  # parametrized_method stop
        print(String(y))


fn main():
    print("start")  # breakpoint
    simple_fn(12)
    parametrized_fn[Int](13)
    Struct[Float32]().parametrized_method[Int](12.25, 13)
