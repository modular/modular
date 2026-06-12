# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Foo[X: TrivialRegisterPassable, Y: TrivialRegisterPassable]:
    def __init__(out self):
        pass

    def getParametrized[T: TrivialRegisterPassable](self, val: T) -> T:
        @parameter
        def nested_function(z: T) -> T:
            return z  # breakpoint

        return nested_function(val)

    def getFloat(self, x: Float32, y: Int) -> Float32:
        var tmp = self.getParametrized[Float32](Float32(4.125 + x + Float32(y)))
        stop_tail_call()
        return tmp


def main():
    print(Foo[Int, Int]().getFloat(1.125, 100))
    stop_tail_call()


# We're inspecting a stack trace and want frames to show up predictably: don't
# let TCO remove them in this test.
@no_inline
def stop_tail_call():
    pass
