# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from memory import UnsafePointer
from random import rand
from sys.info import alignof
from sys import env_get_string, is_defined

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from compile import compile_code
from gpu.host.device_context import DeviceBuffer, DeviceContext
from testing import assert_almost_equal, assert_equal, assert_true

from utils import StaticIntTuple
from utils.index import product


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(inout self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    fn __init__(inout self, v: Int):
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
        inout self,
        dynamic_shape: StaticIntTuple[rank] = _make_tuple[rank](shape),
    ):
        self.tensor = NDBuffer[type, rank, shape](
            UnsafePointer[Scalar[type]].alloc(product(dynamic_shape, rank)),
            dynamic_shape,
        )

    @always_inline
    fn __init__(
        inout self,
        dynamic_shape: DimList,
    ):
        self.__init__(_make_tuple[rank](dynamic_shape))

    @always_inline
    fn __del__(owned self):
        self.tensor.data.free()


@value
struct DeviceNDBuffer[
    type: DType,
    rank: Int,
    /,
    shape: DimList = DimList.create_unknown[rank](),
]:
    var buffer: DeviceBuffer[type]
    var tensor: NDBuffer[type, rank, shape]

    @always_inline
    fn __init__(
        inout self,
        dynamic_shape: StaticIntTuple[rank] = _make_tuple[rank](shape),
        *,
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.create_buffer[type](product(dynamic_shape, rank))
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.ptr, dynamic_shape
        )

    @always_inline
    fn __init__(
        inout self,
        dynamic_shape: DimList,
        *,
        ctx: DeviceContext,
    ) raises:
        self.__init__(_make_tuple[rank](dynamic_shape), ctx=ctx)

    @always_inline
    fn __init__(
        inout self,
        dynamic_shape: StaticIntTuple[rank] = _make_tuple[rank](shape),
        *,
        stride: StaticIntTuple[rank],
        ctx: DeviceContext,
    ) raises:
        # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
        self.buffer = ctx.create_buffer[type](product(dynamic_shape, rank))
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.ptr, dynamic_shape, stride
        )

    @always_inline
    fn __init__(
        inout self,
        dynamic_shape: DimList,
        *,
        stride: StaticIntTuple[rank],
        ctx: DeviceContext,
    ) raises:
        self.__init__(_make_tuple[rank](dynamic_shape), stride=stride, ctx=ctx)


# TODO: add address_space: AddressSpace = AddressSpace.GENERIC
@value
struct TestTensor[type: DType, rank: Int]:
    var ndbuffer: NDBuffer[type, rank]
    var shape: DimList
    var num_elements: Int

    fn __init__(
        inout self,
        shape: DimList,
        values: List[Scalar[type]] = List[Scalar[type]](),
    ):
        self.num_elements = int(shape.product[rank]())
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

    fn __copyinit__(inout self, other: Self):
        self.num_elements = other.num_elements
        self.shape = other.shape
        self.ndbuffer = NDBuffer[type, rank](
            UnsafePointer[Scalar[type]].alloc(self.num_elements), self.shape
        )
        for i in range(self.num_elements):
            self.ndbuffer.data[i] = other.ndbuffer.data[i]

    fn __del__(owned self):
        self.ndbuffer.data.free()


@parameter
@always_inline
fn linspace(buffer: NDBuffer):
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = i * buffer.dim[1]() + j


fn random(buffer: NDBuffer, min: Float64 = 0, max: Float64 = 1):
    rand(buffer.data, buffer.num_elements(), min=min, max=max)


fn zero(buffer: NDBuffer):
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = 0


fn fill[type: DType](buffer: NDBuffer, val: Scalar[type]):
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = val.cast[buffer.type]()


fn bench_compile_time[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    emission_kind: StringLiteral = "asm",
](inout m: Bench, name: String) raises:
    constrained[emission_kind in ("asm", "llvm", "ptx")]()

    # TODO: add docstring, this function should be used on its own or at the end of measured benchmarks.
    @always_inline
    @parameter
    fn bench_call(inout b: Bencher) raises:
        @always_inline
        @parameter
        fn bench_iter() raises:
            @parameter
            if emission_kind == "asm" or emission_kind == "llvm":
                var s: String = compile_code[
                    func, emission_kind=emission_kind
                ]()
                keep(s.unsafe_ptr())
            elif emission_kind == "ptx":
                # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
                with DeviceContext() as ctx:
                    var func = ctx.compile_function[func]()
                    var s: String = func.cuda_function._impl.asm
                    keep(s.unsafe_ptr())

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
        alias diff = int(name_unsafe_ptr[i] - zero)
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


fn env_get_dtype[name: StringLiteral, default: DType]() -> DType:
    """Try to get an DType-valued define. If the name is not defined, return
    a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        An DType parameter value.
    """

    @parameter
    if is_defined[name]():
        return DType._from_str(env_get_string[name]())
    else:
        return default


fn env_get_bool[name: StringLiteral, default: Bool = False]() -> Bool:
    """Try to get an Bool-valued define. If the name is not defined, return
    a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A Bool parameter value.
    """
    alias b = env_get_string[name, "False"]()

    @parameter
    if is_defined[name]():
        if b == "True":
            return True
        elif b == "False":
            return False
    return default


fn int_list_to_tuple[x: List[Int]]() -> StaticIntTuple[len(x)]:
    var t = StaticIntTuple[len(x)]()

    @parameter
    for i in range(len(x)):
        t[i] = x[i]
    return t


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)
