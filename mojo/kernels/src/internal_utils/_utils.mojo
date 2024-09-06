# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from random import rand
from sys.info import alignof

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from buffer import DimList, NDBuffer
from buffer.dimlist import _make_tuple
from compile import compile_code
from gpu.host.device_context import DeviceBuffer, DeviceContext
from testing import assert_almost_equal, assert_equal, assert_true

from utils import StaticIntTuple
from utils.index import product


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
](
    inout m: Bench,
    name: String,
    metrics: List[BenchMetric] = List[BenchMetric](),
) raises:
    constrained[emission_kind in ("asm", "llvm", "ptx")]()

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
                with DeviceContext() as ctx:
                    var func = ctx.compile_function[func]()
                    var s: String = func.cuda_function._impl.asm
                    keep(s.unsafe_ptr())

        b.iter[bench_iter]()

    # To ensure consistency of Bench.dump_report, should set the list of BenchMetrics in metrics to 0.
    var measures: List[ThroughputMeasure] = List[ThroughputMeasure]()
    for i in range(len(metrics)):
        measures.append(ThroughputMeasure(metrics[i], 0))

    m.bench_function[bench_call](
        BenchId("bench_compile" + "/" + emission_kind, name), measures=measures
    )
