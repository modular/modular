# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from random import rand
from sys import argv, env_get_string, is_defined
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
from compile import _internal_compile_code
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
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
    var tensor: NDBuffer[type, rank, shape]

    @always_inline
    fn __init__(
        mut self,
    ):
        constrained[
            shape.all_known[rank](),
            (
                "Must provided dynamic_shape as argument to constructor if not"
                " all shapes are statically known"
            ),
        ]()
        self.tensor = NDBuffer[type, rank, shape](
            UnsafePointer[Scalar[type]].alloc(shape.product().get()),
        )

    @always_inline
    @implicit
    fn __init__(out self, tensor: NDBuffer[type, rank, shape]):
        self.tensor = tensor

    @always_inline
    @implicit
    fn __init__(
        mut self,
        dynamic_shape: IndexList[rank, **_],
    ):
        self.tensor = NDBuffer[type, rank, shape](
            UnsafePointer[Scalar[type]].alloc(product(dynamic_shape, rank)),
            dynamic_shape,
        )

    @always_inline
    @implicit
    fn __init__(
        mut self,
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
        shape,
    ]

    @always_inline
    fn __init__(
        mut self,
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
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.unsafe_ptr(),
        )

    @always_inline
    fn __init__(
        mut self,
        dynamic_shape: IndexList[rank, **_],
        *,
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.enqueue_create_buffer[type](
            product(dynamic_shape, rank)
        )
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.unsafe_ptr(), dynamic_shape
        )

    @always_inline
    fn __init__(
        mut self,
        dynamic_shape: DimList,
        *,
        ctx: DeviceContext,
    ) raises:
        self = Self(_make_tuple[rank](dynamic_shape), ctx=ctx)

    @always_inline
    fn __init__(
        mut self,
        dynamic_shape: IndexList[rank] = _make_tuple[rank](shape),
        *,
        stride: IndexList[rank],
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.enqueue_create_buffer[type](
            product(dynamic_shape, rank)
        )
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.unsafe_ptr(), dynamic_shape, stride
        )

    @always_inline
    fn __init__(
        mut self,
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


# TODO: add address_space: AddressSpace = AddressSpace.GENERIC
@value
struct TestTensor[type: DType, rank: Int]:
    var ndbuffer: NDBuffer[type, rank]
    var shape: DimList
    var num_elements: Int

    fn __init__(
        mut self,
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


fn initialize(buffer: NDBuffer, init_type: InitializationType) raises:
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
fn arange(buffer: NDBuffer):
    @parameter
    if buffer.rank == 2:
        for i in range(buffer.dim[0]()):
            for j in range(buffer.dim[1]()):
                buffer[IndexList[buffer.rank](i, j)] = i * buffer.dim[1]() + j
    else:
        for i in range(len(buffer)):
            buffer.data[i] = i


fn random(buffer: NDBuffer, min: Float64 = 0, max: Float64 = 1):
    rand(buffer.data, buffer.num_elements(), min=min, max=max)


fn zero(buffer: NDBuffer):
    buffer.zero()


fn fill(buffer: NDBuffer, val: Scalar):
    buffer.fill(val.cast[buffer.type]())


fn bench_compile_time[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    emission_kind: StringLiteral = "asm",
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
                var s: String = _internal_compile_code[
                    func, emission_kind=emission_kind
                ]()
                keep(s.unsafe_ptr())
            elif emission_kind == "ptx":
                with DeviceContext() as ctx:
                    var func = ctx.compile_function[func]()
                    # Ensure that the compilation step is not optimized away.
                    keep(UnsafePointer.address_of(func))
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


fn parse_shape[name: StringLiteral]() -> List[Int]:
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


fn env_get_shape[name: StringLiteral, default: StringLiteral]() -> List[Int]:
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


@always_inline
fn _str_fmt_width[max_width: Int = 256](str: String, str_width: Int) -> String:
    """Formats string with a given width.

    Returns:
        sprintf("%-*s", str_width, str)
    """
    debug_assert(str_width > 0, "Should have str_width>0")

    var x = String._buffer_type()
    x.reserve(max_width)
    x._len += _snprintf["%-*s"](x.data, max_width, str_width, str.unsafe_ptr())
    debug_assert(
        len(x) < max_width, "Attempted to access outside array bounds!"
    )
    x._len += 1
    return String(x)


fn ndbuffer_to_str[
    rank: Int,
    axis: Int = 0,
](
    x: NDBuffer[_, rank],
    prev: IndexList[rank] = IndexList[rank](),
    width: Int = 8,
    space_in: String = "",
) -> String:
    """Pretty print a rank-dimensional NDBuffer.

    Returns:
        String(NDBuffer[rank]).
    """
    var cur = prev

    space = space_in
    for _ in range(axis):
        space += " "
    var s = String()

    for i in range(axis):
        s += String(prev[i]) + ","
    s = space + String("(") + s + String(")")

    var out_str = s + String(":[\n") + space
    for i in range(x.dynamic_shape[axis]):
        cur[axis] = i

        @parameter
        if axis == rank - 1:
            out_str += _str_fmt_width(String(x[cur]), width)
        else:
            out_str += ndbuffer_to_str[rank, axis + 1](x, cur, width, space)
    out_str += "]\n"
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
