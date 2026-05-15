"""Benchmark Python -> Mojo FFI: stdlib def_function vs def_py_c_fastcall_function."""

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep,
)
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t


def noop(x: PythonObject) raises -> PythonObject:
    return x


def add(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return PythonObject(Int(py=a) + Int(py=b))


def noop_fastcall(
    _self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, ImmutAnyOrigin],
    nargs: Py_ssize_t,
) -> PyObjectPtr:
    ref cpy = Python().cpython()
    var item = args[0]
    cpy.Py_IncRef(item)
    return item


def add_fastcall(
    _self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, ImmutAnyOrigin],
    nargs: Py_ssize_t,
) -> PyObjectPtr:
    try:
        var a = PythonObject(from_borrowed=args[0])
        var b = PythonObject(from_borrowed=args[1])
        return PythonObject(Int(py=a) + Int(py=b)).steal_data()
    except:
        return PyObjectPtr()


@parameter
def bench_a_def_function_noop(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_bench_a")
    m.def_function[noop]("noop")
    var mod = m.finalize()
    var f = mod.noop
    var arg = PythonObject(1)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var r = f(arg)
            keep(r)

    b.iter[call_fn]()
    keep(mod)
    keep(arg)
    keep(f)


@parameter
def bench_b_fastcall_noop(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_bench_b")
    m.def_py_c_fastcall_function(noop_fastcall, "noop")
    var mod = m.finalize()
    var f = mod.noop
    var arg = PythonObject(1)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var r = f(arg)
            keep(r)

    b.iter[call_fn]()
    keep(mod)
    keep(arg)
    keep(f)


@parameter
def bench_a_def_function_add(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_bench_a2")
    m.def_function[add]("add")
    var mod = m.finalize()
    var f = mod.add
    var arg_a = PythonObject(1)
    var arg_b = PythonObject(2)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var r = f(arg_a, arg_b)
            keep(r)

    b.iter[call_fn]()
    keep(mod)
    keep(arg_a)
    keep(arg_b)
    keep(f)


@parameter
def bench_b_fastcall_add(mut b: Bencher) raises:
    _ = Python()
    var m = PythonModuleBuilder("_ffi_bench_b2")
    m.def_py_c_fastcall_function(add_fastcall, "add")
    var mod = m.finalize()
    var f = mod.add
    var arg_a = PythonObject(1)
    var arg_b = PythonObject(2)

    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):
            var r = f(arg_a, arg_b)
            keep(r)

    b.iter[call_fn]()
    keep(mod)
    keep(arg_a)
    keep(arg_b)
    keep(f)


def main() raises:
    _ = Python()
    var m = Bench(BenchConfig(num_repetitions=5, max_runtime_secs=2.0))
    m.bench_function[bench_a_def_function_noop](
        BenchId("a_def_function_noop")
    )
    m.bench_function[bench_b_fastcall_noop](BenchId("b_fastcall_noop"))
    m.bench_function[bench_a_def_function_add](
        BenchId("a_def_function_add")
    )
    m.bench_function[bench_b_fastcall_add](BenchId("b_fastcall_add"))
    print(m)
