# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Foo[X: AnyRegType, Y: AnyRegType]:
    fn __init__(inout self):
        pass

    fn getParametrized[T: AnyRegType](self, val: T) -> T:
        @parameter
        fn nested_function(z: T) -> T:
            return z  # breakpoint

        return nested_function(val)

    fn getFloat(self, x: Float32, y: Int) -> Float32:
        return self.getParametrized[Float32](Float32(4.125 + x + y))


fn main():
    print(Foo[Int, Int]().getFloat(1.125, 100))
