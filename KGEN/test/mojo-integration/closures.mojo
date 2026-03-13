# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from std.sys import argv


from std.runtime.asyncrt import _run


@no_inline
def takeClosure(var writer: Coroutine[Int, ...]) -> Int:
    return _run(writer^)


@no_inline
def makeClosure(x: Int) -> Coroutine[Int, origin_of()._mlir_origin]:
    var z = x * x

    @__copy_capture(z)
    @parameter
    async def writer() -> Int:
        return z

    return writer()


def main():
    try:
        var x = atol(argv()[1])
        var y = atol(argv()[2])

        var writer = makeClosure(x)
        var w = takeClosure(writer^)
        # CHECK: 4
        print(w)
    except e:
        print(e)
