# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.info import alignof

from buffer import DimList, NDBuffer
from buffer.dimlist import _make_tuple
from testing import assert_equal, assert_true

from utils import InlineArray
from gpu.host.device_context import DeviceBuffer, DeviceContext
from utils.index import product
from random import random_float64
from testing import assert_equal, assert_almost_equal


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
            DTypePointer[type].alloc(product(dynamic_shape, rank)),
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
            DTypePointer[type].alloc(self.num_elements), shape
        )
        if len(values) == self.num_elements:
            for i in range(self.num_elements):
                self.ndbuffer.data[i] = values[i]

    fn __copyinit__(inout self, other: Self):
        self.num_elements = other.num_elements
        self.shape = other.shape
        self.ndbuffer = NDBuffer[type, rank](
            DTypePointer[type].alloc(self.num_elements), self.shape
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
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = random_float64(min, max)


fn zero(buffer: NDBuffer):
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = 0


fn fill[type: DType](buffer: NDBuffer, val: Scalar[type]):
    for i in range(buffer.dim[0]()):
        for j in range(buffer.dim[1]()):
            buffer[(i, j)] = val
