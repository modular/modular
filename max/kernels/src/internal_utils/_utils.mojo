# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import InlineArray
from collections.string import StaticString
from os import abort
from pathlib import Path
from random import rand, random_float64
from sys import argv, bitwidthof, env_get_string, is_defined
from sys.info import alignof

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    clobber_memory,
    keep,
)
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from builtin.dtype import _integral_type_of
from compile import compile_info
from gpu.host import DeviceBuffer, DeviceContext
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from memory import UnsafePointer
from memory.unsafe import bitcast
from stdlib.builtin.io import _snprintf
from tensor_internal import DynamicTensor, ManagedTensorSlice
from testing import assert_almost_equal, assert_equal, assert_true

from utils import Index, IndexList
from utils.index import product


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(out self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    @implicit
    fn __init__(out self, v: Int):
        self.value = v


@value
struct HostNDBuffer[
    type: DType,
    rank: Int,
    /,
    shape: DimList = DimList.create_unknown[rank](),
]:
    var tensor: NDBuffer[type, rank, MutableAnyOrigin, shape]

    @always_inline
    fn __init__(
        out self,
    ):
        constrained[
            shape.all_known[rank](),
            (
                "Must provided dynamic_shape as argument to constructor if not"
                " all shapes are statically known"
            ),
        ]()
        self.tensor = NDBuffer[type, rank, MutableAnyOrigin, shape](
            UnsafePointer[Scalar[type]].alloc(shape.product().get()),
        )

    @always_inline
    @implicit
    fn __init__(out self, tensor: NDBuffer[type, rank, _, shape]):
        self.tensor = tensor

    @always_inline
    @implicit
    fn __init__(
        out self,
        dynamic_shape: IndexList[rank, **_],
    ):
        self.tensor = NDBuffer[type, rank, _, shape](
            UnsafePointer[Scalar[type]].alloc(product(dynamic_shape, rank)),
            dynamic_shape,
        )

    @always_inline
    @implicit
    fn __init__(
        out self,
        dynamic_shape: DimList,
    ):
        self = Self(_make_tuple[rank](dynamic_shape))

    @always_inline
    fn __del__(owned self):
        self.tensor.data.free()

    def copy_to_device(
        self, ctx: DeviceContext
    ) -> DeviceNDBuffer[type, rank, shape]:
        var retval = DeviceNDBuffer[type, rank, shape](
            self.tensor.dynamic_shape, ctx=ctx
        )
        ctx.enqueue_copy(retval.buffer, self.tensor.data)
        return retval^

    fn to_layout_tensor(
        self,
        out result: LayoutTensor[
            type, Layout.row_major(IntTuple(shape)), MutableAnyOrigin
        ],
    ):
        result = __type_of(result)(
            self.tensor.data,
            RuntimeLayout[__type_of(result).layout](
                self.tensor.get_shape(), self.tensor.get_strides()
            ),
        )


@value
struct DeviceNDBuffer[
    type: DType,
    rank: Int,
    /,
    shape: DimList = DimList.create_unknown[rank](),
]:
    var buffer: DeviceBuffer[type]
    var tensor: NDBuffer[
        type,
        rank,
        MutableAnyOrigin,
        shape,
    ]

    @always_inline
    fn __init__(
        out self,
        *,
        ctx: DeviceContext,
    ) raises:
        constrained[
            shape.all_known[rank](),
            (
                "Must provided dynamic_shape as argument to constructor if not"
                " all shapes are statically known"
            ),
        ]()
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.enqueue_create_buffer[type](shape.product().get())
        self.tensor = NDBuffer[type, rank, MutableAnyOrigin, shape](
            self.buffer._unsafe_ptr(),
        )

    @always_inline
    fn __init__(
        out self,
        dynamic_shape: IndexList[rank, **_],
        *,
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.enqueue_create_buffer[type](
            product(dynamic_shape, rank)
        )
        self.tensor = NDBuffer[type, rank, MutableAnyOrigin, shape](
            self.buffer._unsafe_ptr(), dynamic_shape
        )

    @always_inline
    fn __init__(
        out self,
        dynamic_shape: DimList,
        *,
        ctx: DeviceContext,
    ) raises:
        self = Self(_make_tuple[rank](dynamic_shape), ctx=ctx)

    @always_inline
    fn __init__(
        out self,
        dynamic_shape: IndexList[rank] = _make_tuple[rank](shape),
        *,
        stride: IndexList[rank],
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.enqueue_create_buffer[type](
            product(dynamic_shape, rank)
        )
        self.tensor = NDBuffer[type, rank, MutableAnyOrigin, shape](
            self.buffer._unsafe_ptr(), dynamic_shape, stride
        )

    @always_inline
    fn __init__(
        out self,
        dynamic_shape: DimList,
        *,
        stride: IndexList[rank],
        ctx: DeviceContext,
    ) raises:
        self = Self(_make_tuple[rank](dynamic_shape), stride=stride, ctx=ctx)

    def copy_from_device(
        self, ctx: DeviceContext
    ) -> HostNDBuffer[type, rank, shape]:
        var retval = HostNDBuffer[type, rank, shape](self.tensor.dynamic_shape)
        ctx.enqueue_copy(retval.tensor.data, self.buffer)
        return retval^

    fn to_layout_tensor(
        ref self,
        out result: LayoutTensor[
            type, Layout.row_major(IntTuple(shape)), __origin_of(self.buffer)
        ],
    ):
        result = __type_of(result)(
            self.buffer,
            RuntimeLayout[__type_of(result).layout](
                self.tensor.get_shape(), self.tensor.get_strides()
            ),
        )


# TODO: add address_space: AddressSpace = AddressSpace.GENERIC
@value
struct TestTensor[type: DType, rank: Int]:
    var ndbuffer: NDBuffer[type, rank, MutableAnyOrigin]
    var shape: DimList
    var num_elements: Int

    fn __init__(
        out self,
        shape: DimList,
        values: List[Scalar[type]] = List[Scalar[type]](),
    ):
        self.num_elements = Int(shape.product[rank]())
        self.shape = shape
        self.ndbuffer = NDBuffer[type, rank](
            UnsafePointer[Scalar[type]].alloc(self.num_elements), shape
        )
        if len(values) == 1:
            for i in range(self.num_elements):
                self.ndbuffer.data[i] = values[0]
            return

        if len(values) == self.num_elements:
            for i in range(self.num_elements):
                self.ndbuffer.data[i] = values[i]

    fn __copyinit__(out self, other: Self):
        self.num_elements = other.num_elements
        self.shape = other.shape
        self.ndbuffer = NDBuffer[type, rank](
            UnsafePointer[Scalar[type]].alloc(self.num_elements), self.shape
        )
        for i in range(self.num_elements):
            self.ndbuffer.data[i] = other.ndbuffer.data[i]

    fn __del__(owned self):
        self.ndbuffer.data.free()

    fn to_managed_tensor_slice(self) -> DynamicTensor[type, rank].Type:
        return DynamicTensor[type, rank].Type(self.ndbuffer)


@value
struct InitializationType:
    var _value: Int
    alias zero = InitializationType(0)
    alias one = InitializationType(1)
    alias uniform_distribution = InitializationType(2)
    alias arange = InitializationType(3)

    fn __init__(out self, value: Int):
        self._value = value

    fn __init__(out self, value: Float64):
        self._value = Int(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

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
        else:
            raise Error("Invalid initialization type")


fn initialize(
    buffer: NDBuffer[mut=True, **_], init_type: InitializationType
) raises:
    if init_type == InitializationType.zero:
        buffer.zero()
    elif init_type == InitializationType.one:
        buffer.fill(1)
    elif init_type == InitializationType.uniform_distribution:
        random(buffer)
    elif init_type == InitializationType.arange:
        arange(buffer)
    else:
        raise Error("Invalid initialization type")


@parameter
@always_inline
fn arange(buffer: NDBuffer[mut=True, *_]):
    @parameter
    if buffer.rank == 2:
        for i in range(buffer.dim[0]()):
            for j in range(buffer.dim[1]()):
                buffer[IndexList[buffer.rank](i, j)] = i * buffer.dim[1]() + j
    else:
        for i in range(len(buffer)):
            buffer.data[i] = i


fn zero(buffer: NDBuffer):
    buffer.zero()


fn fill(buffer: NDBuffer[mut=True, *_], val: Scalar):
    buffer.fill(val.cast[buffer.type]())


fn bench_compile_time[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    emission_kind: StaticString = "asm",
](mut m: Bench, name: String) raises:
    constrained[emission_kind in ("asm", "llvm", "ptx")]()

    # TODO: add docstring, this function should be used on its own or at the end of measured benchmarks.
    @always_inline
    @parameter
    fn bench_call(mut b: Bencher) raises:
        @always_inline
        @parameter
        fn bench_iter() raises:
            @parameter
            if emission_kind == "asm" or emission_kind == "llvm":
                var s = compile_info[func, emission_kind=emission_kind]().asm
                keep(s.unsafe_ptr())
            elif emission_kind == "ptx":
                with DeviceContext() as ctx:
                    var func = ctx.compile_function[func]()
                    # Ensure that the compilation step is not optimized away.
                    keep(UnsafePointer(to=func))
                    clobber_memory()

        b.iter[bench_iter]()

    # To ensure consistency of Bench.dump_report, we should set
    # the value of all measured metrics m to 0.
    var measures: List[ThroughputMeasure] = List[ThroughputMeasure]()
    if len(m.info_vec) > 0:
        var ref_measures = m.info_vec[0].measures
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
    alias zero = "0".unsafe_ptr()[0]
    alias x_ptr = "x".unsafe_ptr()[0]
    alias name_unsafe_ptr = name.unsafe_ptr()

    var vals: List[Int] = List[Int]()
    var sum: Int = 0

    @parameter
    for i in range(len(name)):
        alias diff = Int(name_unsafe_ptr[i] - zero)
        constrained[Bool(name_unsafe_ptr[i] == x_ptr) or Bool(0 <= diff <= 9)]()

        @parameter
        if name_unsafe_ptr[i] == x_ptr:
            vals.append(sum)
            sum = 0
            continue
        sum = sum * 10 + diff
    vals.append(sum)
    return vals


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
    alias shape_str = env_get_string[name, default]()
    alias shape: List[Int] = parse_shape[shape_str]()
    return shape


fn int_list_to_tuple[x: List[Int]]() -> IndexList[len(x)]:
    var t = IndexList[len(x)]()

    @parameter
    for i in range(len(x)):
        t[i] = x[i]
    return t


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
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


@always_inline
fn _str_fmt_width[str_max: Int = 256](s: String, str_width: Int) -> String:
    """Return `s` padded on the left with spaces so that the total length is
    exactly `str_width`.  If `s` is already longer than `str_width` it is
    returned unchanged. This re-implementation avoids the previous reliance on `_snprintf`
    """
    debug_assert(str_width > 0, "str_width must be positive")
    var current_len = len(s)
    if current_len >= str_width:
        return s

    # pads to the right to match numpy
    var out = s
    var pad_len = str_width - current_len
    out += " " * pad_len
    return out


@always_inline
fn _compute_max_scalar_str_len[rank: Int](x: NDBuffer[_, rank]) -> Int:
    """Return the maximum string length of all scalar elements in `x`."""
    var max_len: Int = 0
    for i in range(x.num_elements()):
        max_len = max(max_len, len(String(x.data[i])))
    return max_len


@always_inline
fn ndbuffer_to_str[
    rank: Int,
    axis: Int = 0,
](
    x: NDBuffer[_, rank],
    prev: IndexList[rank] = IndexList[rank](),
    width: Int = 8,
    indent: String = "",
) -> String:
    """Pretty-print an NDBuffer similar to NumPy's ndarray formatting.
    The tensor is rendered as nested bracketed lists, one row (or sub-tensor)
    per line, without the index headers previously produced.
    """
    var cur = prev

    var effective_width = width
    if axis == 0 and indent == "":
        # ensure we have enough space to print the longest scalar plus one
        # leading space for separation.
        var computed = _compute_max_scalar_str_len[rank](x) + 1
        effective_width = max(effective_width, computed)

    # maximum desired terminal width before wrapping.
    var max_line_width = 120

    var out_str = String()

    @parameter
    if axis == rank - 1:
        # base case – print a 1-d slice
        var num_elems = x.dynamic_shape[axis]
        out_str += "["
        var line_len: Int = 0
        for i in range(num_elems):
            cur[axis] = i
            var formatted = _str_fmt_width(String(x[cur]), effective_width)

            # decide if we need to wrap before appending this element.
            if i > 0:
                if line_len + 1 + effective_width > max_line_width:
                    out_str += (
                        "\n" + indent + " "
                    )  # align under opening bracket
                    line_len = 0
                else:
                    # use single separator since previous value already padded right
                    out_str += " "
                    line_len += 1

            if i == num_elems - 1:
                out_str += String(x[cur])
            else:
                out_str += formatted
            line_len += effective_width
        out_str += "]"
    else:
        # Recursive case – print higher-rank slices, line by line.
        out_str += "["
        var next_indent = indent + " "
        for i in range(x.dynamic_shape[axis]):
            cur[axis] = i
            if i > 0:
                out_str += "\n" + next_indent
            out_str += ndbuffer_to_str[rank, axis + 1](
                x, cur, effective_width, next_indent
            )
        out_str += "]"
    return out_str


fn array_equal[
    type: DType,
    rank: Int,
](x_array: NDBuffer[type, rank], y_array: NDBuffer[type, rank],) raises:
    """Assert two ndbuffers have identical type, rank, length, and values."""

    assert_true(
        x_array.dynamic_shape.flattened_length()
        == y_array.dynamic_shape.flattened_length()
    )
    for i in range(x_array.dynamic_shape.flattened_length()):
        assert_equal(x_array.data[i], y_array.data[i])


@value
@register_passable("trivial")
struct Mode(Stringable):
    var _value: Int
    var handle: StaticString
    alias NONE = Self(0x0, "none")
    alias RUN = Self(0x1, "run")
    alias BENCHMARK = Self(0x2, "benchmark")
    alias VERIFY = Self(0x4, "verify")
    alias SEP = "+"

    fn __init__(out self, handle: String = "run+benchmark+verify") raises:
        var handle_lower = handle.lower().split(Self.SEP)
        self = Self.NONE
        for h in handle_lower:
            if String(Self.RUN.handle) == h[]:
                self.append(Self.RUN)
            elif String(Self.BENCHMARK.handle) == h[]:
                self.append(Self.BENCHMARK)
            elif String(Self.VERIFY.handle) == h[]:
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


fn random[
    dtype: DType
](buffer: NDBuffer[mut=True, dtype, **_], min: Float64 = 0, max: Float64 = 1):
    @parameter
    if dtype.is_float8():
        var size = buffer.num_elements()
        for i in range(size):
            var rnd = (random_float64(min, max) - 0.5) * 2.0
            buffer.data[i] = rnd.cast[dtype]()
    else:
        rand(buffer.data, buffer.num_elements(), min=min, max=max)


fn update_bench_config(mut b: Bench) raises:
    # TODO: refactor and move to bencher.mojo when internal_utils is available in oss.

    # b.config.out_file = Path(arg_parse("bench-out-file", String(b.config.out_file)))
    b.config.min_runtime_secs = arg_parse(
        "bench-min-runtime-secs", b.config.min_runtime_secs
    )
    b.config.max_runtime_secs = arg_parse(
        "bench-max-runtime-secs", b.config.max_runtime_secs
    )
    b.config.min_warmuptime_secs = arg_parse(
        "bench-min-warmuptime-secs", b.config.min_warmuptime_secs
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
