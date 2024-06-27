# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import isclose
from sys.info import alignof

from buffer import DimList, NDBuffer
from buffer.list import _make_tuple
from testing import assert_equal, assert_true

from utils import InlineArray
from gpu.host.device_context import DeviceBuffer, DeviceContext
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
    fn __init__(inout self):
        self.tensor = NDBuffer[type, rank, shape](
            DTypePointer[type].alloc(int(shape.product[len(shape)]()))
        )

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
        dynamic_shape: StaticIntTuple[rank] = _make_tuple[rank](shape),
        *,
        stride: StaticIntTuple[rank],
        ctx: DeviceContext,
    ) raises:
        self.buffer = ctx.create_buffer[type](product(dynamic_shape, rank))
        self.tensor = NDBuffer[type, rank, shape](
            self.buffer.ptr, dynamic_shape, stride
        )


fn get_minmax[dtype: DType](x: DTypePointer[dtype], N: Int) -> SIMD[dtype, 2]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return SIMD[dtype, 2](min_val, max_val)


fn compare[
    dtype: DType, N: Int, verbose: Bool = True
](x: DTypePointer[dtype], y: DTypePointer[dtype], label: String) -> SIMD[
    dtype, 4
]:
    alias alignment = alignof[dtype]()

    var atol = DTypePointer[dtype].alloc(N, alignment=alignment)
    var rtol = DTypePointer[dtype].alloc(N, alignment=alignment)

    # TODO: parallelize and unroll this loop
    for i in range(N):
        var d = abs(x[i] - y[i])
        var e = abs(d / y[i])
        atol[i] = d
        rtol[i] = e

    var atol_minmax = get_minmax[dtype](atol, N)
    var rtol_minmax = get_minmax[dtype](rtol, N)
    if verbose:
        print(label)
        print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
        print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
        print("==========================================================")
    atol.free()
    rtol.free()
    return SIMD[dtype, 4](
        atol_minmax[0], atol_minmax[1], rtol_minmax[0], rtol_minmax[1]
    )


fn array_equal[
    type: DType,
    rank: Int,
](x: DTypePointer[type], y: DTypePointer[type], num_elements: Int,) -> Bool:
    for i in range(num_elements):
        if not isclose(x[i], y[i]):
            print("FAIL: mismatch at idx ", end="")
            print(i)
            return False
    return True


# TODO: call the above function in this
fn array_equal[
    type: DType, rank: Int
](x: NDBuffer[type, rank], y: NDBuffer[type, rank]) -> Bool:
    for i in range(x.num_elements()):
        if not isclose(x.data[i], y.data[i]):
            print("FAIL: mismatch at idx ", end="")
            print(x.get_nd_index(i))
            return False
    return True


fn array_equal[
    type: DType, rank: Int
](x: TestTensor[type, rank], y: TestTensor[type, rank]) -> Bool:
    return array_equal(x.ndbuffer, y.ndbuffer)


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
