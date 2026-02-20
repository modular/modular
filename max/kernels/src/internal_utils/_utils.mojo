# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import time
from collections import Optional
from math import ceildiv, floor
from sys import argv, env_get_string
from builtin.device_passable import DevicePassable

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    ThroughputMeasure,
    clobber_memory,
    keep,
)
from buffer import Dim, DimList
from buffer.dimlist import _make_tuple
from compile import compile_info
from gpu import *
from gpu.host import DeviceBuffer, DeviceContext
from random import Random
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from tensor import DynamicTensor
from utils import IndexList


# ===----------------------------------------------------------------------=== #
# Unsigned division/modulo helpers for Int
# ===----------------------------------------------------------------------=== #
# These helpers allow performing unsigned division and modulo operations on Int
# arguments while using unsigned hardware operations internally. This is useful
# when migrating from UInt to Int, as unsigned division is faster on NVIDIA GPUs.


@always_inline
fn ufloordiv(a: Int, b: Int) -> Int:
    """Perform unsigned division (`//`) on Int arguments.

    This function treats both arguments as unsigned values and performs
    unsigned division, which is faster than signed division on NVIDIA GPUs.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        The quotient of unsigned division.
    """
    return Int(UInt(a) // UInt(b))


struct ValOrDim[dim: Dim = Dim()](Defaultable):
    var value: Int

    fn __init__(out self):
        comptime assert (
            not Self.dim.is_dynamic()
        ), "Can't construct a dynamic dim with no runtime value"
        self.value = Self.dim.get()

    fn __init__(out self, v: Int):
        self.value = v


struct InitializationType(DevicePassable, Equatable, TrivialRegisterPassable):
    var _value: Int
    comptime zero = InitializationType(0)
    comptime one = InitializationType(1)
    comptime uniform_distribution = InitializationType(2)
    comptime arange = InitializationType(3)
    comptime fill = InitializationType(4)

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "InitializationType"

    fn __init__(out self, value: Int):
        self._value = value

    fn __init__(out self, value: Float64):
        self._value = Int(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @staticmethod
    fn from_str(str: String) raises -> Self:
        if str == "zero":
            return InitializationType.zero
        elif str == "one":
            return InitializationType.one
        elif str == "uniform_distribution":
            return InitializationType.uniform_distribution
        elif str == "arange":
            return InitializationType.arange
        elif str == "fill":
            return InitializationType.fill
        else:
            raise Error("Invalid initialization type")


# TODO: refactor the following to run exactly once.
fn bench_compile_time[
    func_type: __TypeOfAllTypes,
    //,
    func: func_type,
    emission_kind: StaticString = "asm",
](mut m: Bench, name: String) raises:
    comptime assert emission_kind in ("asm", "llvm", "ptx")

    # TODO: add docstring, this function should be used on its own or at the end of measured benchmarks.
    @always_inline
    @parameter
    fn bench_call(mut b: Bencher) raises:
        @always_inline
        @parameter
        fn bench_iter() raises:
            comptime if emission_kind == "asm" or emission_kind == "llvm":
                var s = compile_info[func, emission_kind=emission_kind]().asm
                keep(s.unsafe_ptr())
            elif emission_kind == "ptx":
                with DeviceContext() as ctx:
                    var func = ctx.compile_function_unchecked[func]()
                    # Ensure that the compilation step is not optimized away.
                    keep(UnsafePointer(to=func))
                    clobber_memory()

        b.iter[bench_iter]()

    # To ensure consistency of Bench.dump_report, we should set
    # the value of all measured metrics m to 0.
    var measures: List[ThroughputMeasure] = List[ThroughputMeasure]()
    if len(m.info_vec) > 0:
        ref ref_measures = m.info_vec[0].measures
        for i in range(len(ref_measures)):
            metric = ref_measures[i].metric
            measures.append(ThroughputMeasure(metric, 0))

    m.bench_function[bench_call](
        BenchId("bench_compile" + "/" + emission_kind, name), measures=measures
    )


fn parse_shape[name: StaticString]() -> List[Int]:
    """Parse string to get an integer-valued shape (2+ dims) define.

    For example, the following shapes:
    - shape = x123 => (0,123)
    - 123 = Not applicable
    - 123x = (123,0)
    - 123x456 = (123,456)

    Parameters:
        name: The name of the define.

    Returns:
        A List[Int] parameter value.
    """
    comptime zero = "0".unsafe_ptr()[0]
    comptime x_ptr = "x".unsafe_ptr()[0]
    comptime name_unsafe_ptr = name.unsafe_ptr()

    var vals: List[Int] = List[Int]()
    var sum: Int = 0

    comptime for i in range(len(name)):
        comptime diff = Int(name_unsafe_ptr[i] - zero)
        comptime assert name_unsafe_ptr[i] == x_ptr or 0 <= diff <= 9

        comptime if name_unsafe_ptr[i] == x_ptr:
            vals.append(sum)
            sum = 0
            continue
        sum = sum * 10 + diff
    vals.append(sum)
    return vals^


fn env_get_shape[name: StaticString, default: StaticString]() -> List[Int]:
    """Try to get an integer-valued shape (2+ dims) define.
    Compilation fails if the name is not defined.

    For example, the following shapes:
    - shape = x123 => (0,123)
    - 123 = Not applicable
    - 123x = (123,0)
    - 123x456 = (123,456)

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A List[Int] parameter value.
    """
    comptime shape_str = env_get_string[name, default]()
    comptime shape: List[Int] = parse_shape[shape_str]()
    return materialize[shape]()


fn int_list_to_tuple[x: List[Int]]() -> IndexList[len(x)]:
    var t = IndexList[len(x)]()

    comptime for i in range(len(x)):
        comptime xi = x[i]
        t[i] = xi
    return t


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim[]:
    return ValOrDim(d)


fn arg_parse(handle: String, default: Int) raises -> Int:
    # TODO: add constraints on dtype of return value
    var args = argv()
    for i in range(len(args)):
        if args[i].startswith("--" + handle):
            var name_val = args[i].split("=")
            return Int(name_val[1])
    return default


fn arg_parse(handle: String, default: Bool) raises -> Bool:
    var args = argv()
    for i in range(len(args)):
        if args[i].startswith("--" + handle):
            var name_val = args[i].split("=")
            if name_val[1] == "True":
                return True
            elif name_val[1] == "False":
                return False
    return default


fn arg_parse(handle: String, default: String) raises -> String:
    # TODO: add constraints on dtype of return value
    var args = argv()
    for i in range(len(args)):
        if args[i].startswith("--" + handle):
            var name_val = args[i].split("=")
            return String(name_val[1])
    return default


fn arg_parse(handle: String, default: Float64) raises -> Float64:
    # TODO: add constraints on dtype of return value
    var args = argv()
    for i in range(len(args)):
        if args[i].startswith("--" + handle):
            var name_val = args[i].split("=")
            return atof(name_val[1])
    return default


@fieldwise_init
struct Mode(Stringable, TrivialRegisterPassable):
    var _value: Int
    var handle: StaticString
    comptime NONE = Self(0x0, "none")
    comptime RUN = Self(0x1, "run")
    comptime BENCHMARK = Self(0x2, "benchmark")
    comptime VERIFY = Self(0x4, "verify")
    comptime SEP = "+"

    fn __init__(out self, handle: String = "run+benchmark+verify") raises:
        var handle_lower = handle.lower().split(Self.SEP)
        self = Self.NONE
        for h in handle_lower:
            if String(Self.RUN.handle) == h:
                self.append(Self.RUN)
            elif String(Self.BENCHMARK.handle) == h:
                self.append(Self.BENCHMARK)
            elif String(Self.VERIFY.handle) == h:
                self.append(Self.VERIFY)

    fn append(mut self, other: Self):
        self._value |= other._value

    fn __str__(self) -> String:
        s = List[String]()
        if Self.RUN == self:
            s.append(Self.RUN.handle)
        if Self.BENCHMARK == self:
            s.append(Self.BENCHMARK.handle)
        if Self.VERIFY == self:
            s.append(Self.VERIFY.handle)
        if Self.NONE == self:
            s.append(Self.NONE.handle)
        return StaticString(Self.SEP).join(s)

    fn __eq__(self, mode: Self) -> Bool:
        if mode._value == self._value == Self.NONE._value:
            return True
        return True if self._value & mode._value else False


fn update_bench_config_args(mut b: Bench) raises:
    # TODO: refactor and move to bencher.mojo when internal_utils is available in oss.

    # b.config.out_file = Path(arg_parse("bench-out-file", String(b.config.out_file)))
    b.config.min_runtime_secs = arg_parse(
        "bench-min-runtime-secs", b.config.min_runtime_secs
    )
    b.config.max_runtime_secs = arg_parse(
        "bench-max-runtime-secs", b.config.max_runtime_secs
    )
    b.config.num_warmup_iters = arg_parse(
        "bench-num-warmup-iters", b.config.num_warmup_iters
    )
    # set bench-max-batch-size=1 for single iteration
    b.config.max_batch_size = arg_parse(
        "bench-max-batch-size", b.config.max_batch_size
    )
    # set bench-max-iters=0 for single iteration
    b.config.max_iters = arg_parse("bench-max-iters", b.config.max_iters)
    b.config.num_repetitions = arg_parse(
        "bench-num-repetitions", b.config.num_repetitions
    )
    b.config.flush_denormals = arg_parse(
        "bench-flush-denormals", b.config.flush_denormals
    )


struct Timer:
    var start: Float64
    var current: Float64

    var report: List[String]

    fn __init__(out self):
        self.start = Float64(time.perf_counter_ns())
        self.current = self.start
        self.report = List[String]()

    fn measure(mut self, msg: String):
        var current = Float64(time.perf_counter_ns())
        var elapsed = current - self.current
        self.current = current
        self.report.append("[" + msg + "] " + String(elapsed / 1e6) + " (ms)")

    fn print(self) raises:
        for i in range(len(self.report)):
            print(self.report[i])
        print(
            "[total-elapsed] "
            + String((self.current - self.start) / 1e6)
            + " (ms)"
        )
        print(
            "-----------------------------------------------------------------------"
        )


# TODO: limited support for 1D, generalize to n-D
fn init_vector_gpu[
    dtype: DType
](
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    len: Int,
    mode: InitializationType,
    value: Scalar[dtype],
):
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    @parameter
    fn apply(values: SIMD[dtype, 4]):
        comptime for i in range(4):
            comptime if i == 3:
                if tid >= UInt(len):
                    return
            x[tid] = values[i]
            tid += stride

    var values = SIMD[dtype, 4]()
    if mode == InitializationType.zero:
        values = SIMD[dtype, 4](0)
    elif mode == InitializationType.one:
        values = SIMD[dtype, 4](1)
    elif mode == InitializationType.fill:
        values = SIMD[dtype, 4](value)
    elif mode == InitializationType.uniform_distribution:
        var rng = Random(offset=UInt64(tid))
        values = SIMD[dtype, 4](rng.step_uniform())
    elif mode == InitializationType.arange:
        values = SIMD[dtype, 4](
            UInt64(tid).cast[dtype](),
            UInt64(tid + stride).cast[dtype](),
            UInt64(tid + UInt(2 * Int(stride))).cast[dtype](),
            UInt64(tid + UInt(3 * Int(stride))).cast[dtype](),
        )
    apply(values)


fn init_vector_launch[
    dtype: DType, block_dim: Int = 256
](
    out_device: DeviceBuffer[dtype],
    length: Int,
    init_type: InitializationType,
    context: DeviceContext,
    value: Optional[Scalar[dtype]] = None,
) raises:
    var num_blocks = ceildiv(ceildiv(length, 4), block_dim)
    # using num-threads = 1/4th of length to initialize the array

    comptime kernel = init_vector_gpu[dtype]
    context.enqueue_function_experimental[kernel](
        out_device,
        length,
        init_type,
        value.or_else(0),
        grid_dim=(num_blocks),
        block_dim=(block_dim),
    )


fn _pretty_print_float(val: Float64) -> String:
    """Converts float to string, omitting fractional part if not needed.

    Examples:
        _pretty_print_float(2.0) returns "2"
        _pretty_print_float(2.5) returns "2.5"
    """
    if floor(val) == val:
        return String(Int(val))
    return String(val)


fn human_readable_size(size: Int) -> String:
    """Formats a byte size into human-readable form (KB, MB, GB).

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string (e.g., "4KB", "256MB", "2GB").
    """
    comptime KB = 1024
    comptime MB = KB * KB
    comptime GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"
    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"
    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"
    return String(size) + "B"
