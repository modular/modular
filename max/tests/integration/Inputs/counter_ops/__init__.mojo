# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from register import register_internal
from utils.index import IndexList
from buffer import NDBuffer
from buffer.dimlist import DimList
from python.python import _get_global_python_itf
from python import Python, PythonObject
from os import abort


struct Counter(Movable):
    var a: Int
    var b: Int

    fn __init__(out self):
        self.a = 0
        self.b = 0
        print("counter init (no arg)")

    fn __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b
        print("counter init", a, b)

    fn __moveinit__(out self, owned other: Self):
        self.a = other.a
        self.b = other.b

    fn __del__(owned self):
        print("counter del")

    fn bump(mut self):
        self.a += 1
        self.b += self.a
        print("bumped", self.a, self.b)


@register_internal("make_counter_from_tensor")
fn make_counter(init: NDBuffer[DType.int32, 1]) -> Counter:
    print("making. init:", init[0], init[1])
    return Counter(Int(init[0]), Int(init[1]))


@register_internal("make_counter")
fn make_counter() -> Counter:
    print("making")
    return Counter()


# TODO(MSDK-950): Avoid DCE in the graph compiler and remove return value.
@register_internal("bump_counter")
fn bump_counter(mut c: Counter, output: NDBuffer[DType.bool, 1, DimList(1)]):
    print("bumping")
    c.bump()
    output[0] = True


@register_internal("read_counter")
fn read_counter(c: Counter, output: NDBuffer[DType.int32, 1, DimList(2)]):
    output[0] = c.a
    output[1] = c.b


@register_internal("bump_python_counter")
fn bump_python_counter(
    counter: PythonObject,
) -> PythonObject:
    var cpython = _get_global_python_itf().cpython()
    var state = cpython.PyGILState_Ensure()
    try:
        cpython.check_init_error()
        new_counter = counter.copy()
        new_counter.bump()
        return new_counter
    except e:
        abort(e)
    finally:
        cpython.PyGILState_Release(state)

    return None
