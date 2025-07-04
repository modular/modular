# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from sys import argv


from runtime.asyncrt import _run


@no_inline
fn takeClosure(var writer: Coroutine[Int]) -> Int:
    return _run(writer^)


@no_inline
fn makeClosure(x: Int) -> Coroutine[Int, __origin_of()]:
    var z = x * x

    @__copy_capture(z)
    @parameter
    async fn writer() -> Int:
        return z

    return writer()


fn main():
    try:
        var x = atol(argv()[1])
        var y = atol(argv()[2])

        var writer = makeClosure(x)
        var w = takeClosure(writer^)
        # CHECK: 4
        print(w)
    except e:
        print(e)
