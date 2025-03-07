# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo --debug-level full -D ENABLE_ASSERTIONS %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/test-tensor
# RUN: %mojo-build -D ENABLE_ASSERTIONS %s -o %t/test-tensor
# RUN: %t/test-tensor

from max.driver import (
    DynamicTensor,
    ManagedTensorSlice,
    Tensor,
    cpu,
)
from max.tensor import Tensor as OldTensor
from max.tensor import TensorShape, TensorSpec
from testing import assert_equal, assert_raises, assert_true

from buffer import Dim, NDBuffer
from collections import InlineArray

from tensor_internal import IOUnknown
from tensor_internal.managed_tensor_slice import StaticTensorSpec

from layout.int_tuple import UNKNOWN_VALUE

from utils import Index, IndexList


def test_tensor():
    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

    assert_equal(tensor.spec().rank, 2)
    assert_equal(tensor.spec().type, DType.float32)

    tensor[0, 0] = 0
    tensor[0, 1] = 1
    tensor[1, 0] = 2
    tensor[1, 1] = 3

    assert_equal(tensor[0, 0], 0)
    assert_equal(tensor[1, 1], 3)


def test_round_trip():
    tensor = Tensor[DType.float32, 2](TensorShape(10, 2))

    val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    dt2 = tensor^.to_device_tensor()
    assert_true("DeviceTensor(Device(type=cpu,target_info(" in String(dt2))
    assert_true("Spec(10x2xfloat32))" in String(dt2))

    tensor2 = dt2^.to_tensor[DType.float32, 2]()
    assert_equal(tensor2[1, 0], 3)


def test_copy():
    cpu1 = cpu()
    cpu2 = cpu()

    src_dev_tensor = cpu1.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    dst_dev_tensor = cpu2.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    src = src_dev_tensor^.to_tensor[DType.float32, 2]()

    val = 1
    for i in range(10):
        for j in range(2):
            src[i, j] = val
            val += 1

    src_dev_tensor = src^.to_device_tensor()
    src_dev_tensor.copy_into(dst_dev_tensor)
    dst_dev_tensor2 = src_dev_tensor.copy_to(cpu2)

    dst = dst_dev_tensor^.to_tensor[DType.float32, 2]()
    dst2 = dst_dev_tensor2^.to_tensor[DType.float32, 2]()

    with assert_raises(contains="already allocated on"):
        _ = src_dev_tensor.copy_to(cpu1)

    src = src_dev_tensor^.to_tensor[DType.float32, 2]()
    for i in range(10):
        for j in range(2):
            assert_equal(src[i, j], dst[i, j])
            assert_equal(src[i, j], dst2[i, j])


def test_unsafe_slice():
    var shape = (10, 2)
    var tensor = Tensor[DType.float32, 2](TensorShape(shape))

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var unsafe_slice = DynamicTensor[DType.float32, 2].Type(
        tensor.unsafe_ptr(), shape
    )

    assert_equal(unsafe_slice[1, 1], 4)

    _ = tensor^


def test_unsafe_slice_simd():
    var shape = (10, 10)
    var tensor = Tensor[DType.float32, 2](TensorShape(shape))

    var val = 1
    for i in range(10):
        for j in range(10):
            tensor[i, j] = val
            val += 1

    var unsafe_slice = tensor.unsafe_slice(slice(0, 5, 1), slice(0, 8, 2))
    var value = unsafe_slice.load[4](IndexList[2](0, 1))
    assert_equal(value, SIMD[DType.float32, 4](3, 5, 7, 9))

    unsafe_slice.store(IndexList[2](0, 1), SIMD[DType.float32, 4](0))
    value = unsafe_slice.load[4](IndexList[2](0, 1))
    assert_equal(value, SIMD[DType.float32, 4](0))

    _ = tensor^


def test_kv_cache():
    alias type = DType.float32
    alias shape = (2,)
    tensors = List[Tensor[type, len(shape)]]()
    for _ in range(2):
        tensor = Tensor[type, len(shape)](TensorShape(shape))
        tensor[0] = 1
        tensor[1] = 2

    for t in tensors:
        assert_equal(
            t[][0],
            1,
        )
        assert_equal(
            t[][1],
            2,
        )


def test_raw_data():
    alias type = DType.float32
    alias shape = (1,)
    t = Tensor[DType.float32, len(shape)](TensorShape(shape))
    ptr = t.unsafe_ptr()
    t[0] = 22
    assert_equal(ptr.load(), t[0])


def test_take():
    alias type = DType.float32
    alias shape = (1,)
    alias TensorType = Tensor[type, len(shape)]
    tensors = List[TensorType]()
    for i in range(2):
        tensors.append(TensorType(TensorShape(shape)))

        tensors[i][0] = 2

    def consume_and_check(owned tensor: TensorType):
        assert_equal(tensor[0], 2)

    for tensor in tensors:
        consume_and_check(tensor[].take())


def test_print():
    tensor = Tensor[DType.float32, 2](TensorShape(10, 2))

    val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1
    expected = (
        "Tensor([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0],..., [15.0, 16.0],[17.0,"
        " 18.0],[19.0, 20.0]], dtype=float32, shape=10x2)"
    )
    s = String(tensor).replace("\n", "")
    assert_equal(s, expected)


def test_move():
    alias shape = (1,)
    t = Tensor[DType.float32, len(shape)](TensorShape(shape))
    t[0] = 1.0
    dev = cpu()
    t1 = t^.move_to(dev)
    assert_equal(t1[0], 1.0)


def test_from_max_tensor():
    old = OldTensor[DType.float32](
        TensorShape(
            1,
        ),
        1.0,
    )
    new = Tensor[rank=1](old)
    assert_equal(new[0], 1.0)

    with assert_raises(contains="mismatch in rank, expected 1 given 2"):
        _ = Tensor[rank=2](old)


def test_copy_error():
    cpu = cpu()
    src_dev_tensor = cpu.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    dst_dev_tensor = cpu.allocate(
        TensorSpec(DType.float32, 10, 1),
    )

    with assert_raises(contains="do not match"):
        src_dev_tensor.copy_into(dst_dev_tensor)


fn test_construction_from_managed_tensor_slice() raises:
    alias dtype = DType.float32
    alias rank = 2
    alias rows = 2
    alias cols = 10
    alias row_stride = 1
    alias col_stride = 2

    var static_buffer = NDBuffer[dtype, rank, (rows, cols)]().stack_allocation()
    var static_tensor_slice = ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[dtype, rank]
        .create_unknown()
        .with_layout[rank]((rows, cols), (row_stride, col_stride)),
    ](static_buffer)
    var static_layout_tensor = static_tensor_slice.to_layout_tensor()

    # Assert that the layout really is static
    # RuntimeTuple stores the static value in the `S` parameter
    assert_equal(static_layout_tensor.runtime_layout.shape[0].S.value(), rows)
    assert_equal(static_layout_tensor.runtime_layout.shape[1].S.value(), cols)
    assert_equal(
        static_layout_tensor.runtime_layout.stride[0].S.value(), row_stride
    )
    assert_equal(
        static_layout_tensor.runtime_layout.stride[1].S.value(), col_stride
    )

    var stack_buffer = InlineArray[Scalar[dtype], rows * cols](
        uninitialized=True
    )
    var dynamic_tensor_slice = ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ](stack_buffer.unsafe_ptr(), (rows, cols), (row_stride, col_stride))
    var dynamic_layout_tensor = dynamic_tensor_slice.to_layout_tensor()

    # Assert that the static layout is unknown
    assert_equal(
        dynamic_layout_tensor.runtime_layout.shape[0].S.value(), UNKNOWN_VALUE
    )
    assert_equal(
        dynamic_layout_tensor.runtime_layout.shape[1].S.value(), UNKNOWN_VALUE
    )
    assert_equal(
        dynamic_layout_tensor.runtime_layout.stride[0].S.value(),
        UNKNOWN_VALUE,
    )
    assert_equal(
        dynamic_layout_tensor.runtime_layout.stride[1].S.value(),
        UNKNOWN_VALUE,
    )

    # Assert that the dynamic layout is expected
    assert_equal(dynamic_layout_tensor.runtime_layout.shape[0].get_int(), rows)
    assert_equal(dynamic_layout_tensor.runtime_layout.shape[1].get_int(), cols)
    assert_equal(
        dynamic_layout_tensor.runtime_layout.stride[0].get_int(), row_stride
    )
    assert_equal(
        dynamic_layout_tensor.runtime_layout.stride[1].get_int(), col_stride
    )
    var partially_dynamic_tensor_slice = ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[dtype, rank]
        .create_unknown()
        .with_layout[rank]((Dim(), cols), (Dim(), col_stride)),
    ](stack_buffer.unsafe_ptr(), (rows, cols), (row_stride, col_stride))
    var partially_dynamic_layout_tensor = partially_dynamic_tensor_slice.to_layout_tensor()

    # cols are static, but rows are dynamic
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.shape[0].S.value(),
        UNKNOWN_VALUE,
    )
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.shape[1].S.value(),
        cols,
    )
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.stride[0].S.value(),
        UNKNOWN_VALUE,
    )
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.stride[1].S.value(),
        col_stride,
    )
    # Test the dynamic parts
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.shape[0].get_int(), rows
    )
    assert_equal(
        partially_dynamic_layout_tensor.runtime_layout.stride[0].get_int(),
        row_stride,
    )


def main():
    test_tensor()
    test_unsafe_slice()
    test_round_trip()
    test_copy()
    test_kv_cache()
    test_raw_data()
    test_take()
    test_unsafe_slice_simd()
    test_print()
    test_move()
    test_from_max_tensor()
    test_copy_error()
