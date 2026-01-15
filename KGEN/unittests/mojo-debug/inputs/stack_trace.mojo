# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Foo[X: __TypeOfAllTypes, Y: __TypeOfAllTypes]:
    fn __init__(out self):
        pass

    fn getParametrized[T: __TypeOfAllTypes](self, val: T) -> T:
        @parameter
        fn nested_function(z: T) -> T:
            return z  # breakpoint

        return nested_function(val)

    fn getFloat(self, x: Float32, y: Int) -> Float32:
        var tmp = self.getParametrized[Float32](Float32(4.125 + x + y))
        stop_tail_call()
        return tmp


fn main():
    print(Foo[Int, Int]().getFloat(1.125, 100))
    stop_tail_call()


# We're inspecting a stack trace and want frames to show up predictably: don't
# let TCO remove them in this test.
@no_inline
fn stop_tail_call():
    pass
