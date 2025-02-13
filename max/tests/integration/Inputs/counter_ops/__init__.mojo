# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort

import compiler_internal as compiler
from buffer.dimlist import DimList
from python import Python, PythonObject
from python.python import _get_global_python_itf
from register import register_internal
from tensor import ManagedTensorSlice

from utils.index import IndexList


struct Counter[stride: Int](Movable):
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
        self.a += Self.stride
        self.b += self.a
        print("bumped", self.a, self.b)


@compiler.register("make_counter_from_tensor", num_dps_outputs=0)
struct MakeCounterFromTensor:
    @staticmethod
    fn execute[
        stride: Int,
    ](init: ManagedTensorSlice[DType.int32, 1]) -> Counter[stride]:
        print("making. init:", init[0], init[1])
        return Counter[stride](Int(init[0]), Int(init[1]))


@compiler.register("make_counter")
struct MakeCounter:
    @staticmethod
    fn execute[stride: Int]() -> Counter[stride]:
        print("making")
        return Counter[stride]()


@compiler.register("bump_counter", num_dps_outputs=0)
struct BumpCounter:
    @staticmethod
    fn execute[
        stride: Int,
    ](mut c: Counter[stride]) -> None:
        print("bumping")
        c.bump()


@compiler.register("read_counter")
struct ReadCounter:
    @staticmethod
    fn execute[
        stride: Int
    ](output: ManagedTensorSlice[DType.int32, 1], c: Counter[stride]):
        output[0] = c.a
        output[1] = c.b


@compiler.register("bump_python_counter", num_dps_outputs=0)
struct BumpPythonCounter:
    @staticmethod
    fn execute[stride: Int](counter: PythonObject) -> PythonObject:
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
