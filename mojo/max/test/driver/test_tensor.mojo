# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from driver import CPUDescriptor, cpu_device
from driver import Tensor, TensorSlice, UnsafeTensorSlice
from tensor import TensorSpec
from utils import Index
from testing import assert_equal, assert_raises


def test_tensor():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    tensor = dt^.to_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    tensor[0, 0] = 0
    tensor[0, 1] = 1
    tensor[1, 0] = 2
    tensor[1, 1] = 3

    assert_equal(tensor[0, 0], 0)
    assert_equal(tensor[1, 1], 3)


def test_tensor_slice():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 3, 3),
    )
    tensor = dt^.to_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    for i in range(3):
        for j in range(3):
            tensor[i, j] = i + j

    # tensor
    # 0 1 2
    # 1 2 3
    # 2 3 4

    slice = tensor[0:2, :]

    # slice
    # 0 1 2
    # 1 2 3

    slice_spec = slice.get_static_spec()
    assert_equal(slice_spec.get_rank(), 2)
    assert_equal(slice_spec[0], 2)
    assert_equal(slice_spec[1], 3)

    assert_equal(slice[0, 0], 0)
    assert_equal(slice[1, 2], 3)

    inner_slice = tensor[0:2, 1:2]

    # inner_slice
    # 1
    # 2

    inner_slice_spec = inner_slice.get_static_spec()
    assert_equal(inner_slice_spec.get_rank(), 2)
    assert_equal(inner_slice_spec[0], 2)
    assert_equal(inner_slice_spec[1], 1)

    assert_equal(inner_slice[0, 0], 1)
    assert_equal(inner_slice[1, 0], 2)


def test_slice_with_step():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 18),
    )
    tensor = dt^.to_tensor[DType.float32, 1]()

    index = 0
    for _ in range(3):
        for j in range(6):
            tensor[index] = j
            index += 1

    assert_equal(tensor[1], 1)

    stepped_slice = tensor[0:18:3]
    assert_equal(stepped_slice[1], 3)


def test_2dslice_with_step():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    tensor = dt^.to_tensor[DType.float32, 2]()

    val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    stepped_slice = tensor[::2, :]
    assert_equal(stepped_slice[1, 0], 5)


def test_2dslice_with_step_row_column():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 10, 10),
    )
    tensor = dt^.to_tensor[DType.float32, 2]()

    val = 1
    for i in range(10):
        for j in range(10):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 11)
    assert_equal(tensor[1, 1], 12)
    assert_equal(tensor[2, 2], 23)

    stepped_slice = tensor[1::2, 1::2]
    assert_equal(stepped_slice[0, 0], 12)
    assert_equal(stepped_slice[0, 1], 14)
    assert_equal(stepped_slice[1, 1], 34)

    slice_spec = stepped_slice.get_static_spec()
    assert_equal(slice_spec.get_rank(), 2)
    assert_equal(slice_spec.dtype(), DType.float32)
    assert_equal(slice_spec[0], 5)
    assert_equal(slice_spec[1], 5)

    inner_slice = tensor[1::3, 1::3]
    inner_slice_spec = inner_slice.get_static_spec()
    assert_equal(inner_slice_spec.get_rank(), 2)
    assert_equal(inner_slice_spec.dtype(), DType.float32)
    assert_equal(inner_slice_spec[0], 3)
    assert_equal(inner_slice_spec[1], 3)
    assert_equal(inner_slice[0, 0], 12)
    assert_equal(inner_slice[1, 1], 45)


def test_4dslice_with_step():
    var dev = cpu_device()

    var shape = Index(7, 8, 13, 9)
    var dt = dev.allocate(TensorSpec(DType.float32, shape))
    var tensor = dt^.to_tensor[DType.float32, 4]()

    # np.arange
    var val = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for w in range(shape[3]):
                    tensor[i, j, k, w] = val
                    val += 1

    assert_equal(tensor[3, 7, 12, 0], 3735)

    var stepped_slice = tensor[3::2, 1::, 1::3, 0::2]
    assert_equal(stepped_slice[0, 0, 0, 0], 2934)
    assert_equal(stepped_slice[1, 5, 2, 1], 5447)


def test_round_trip():
    dev = cpu_device()

    dt = dev.allocate(TensorSpec(DType.float32, 10, 2), str("mytensor"))
    tensor = dt^.to_tensor[DType.float32, 2]()

    val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    dt2 = tensor^.to_device_tensor()
    assert_equal(
        str(dt2),
        "DeviceTensor(mytensor,Device(type=CPU),Spec(10x2xfloat32))",
    )

    tensor2 = dt2^.to_tensor[DType.float32, 2]()
    assert_equal(tensor2[1, 0], 3)


def test_copy():
    cpu1 = cpu_device(CPUDescriptor(numa_id=0))
    cpu2 = cpu_device(CPUDescriptor(numa_id=1))

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


def test_set_through_slice():
    dev = cpu_device()

    dt = dev.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    tensor = dt^.to_tensor[DType.float32, 2]()

    val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    slice = tensor[1:, :]
    assert_equal(slice[0, 0], 3)

    slice[0, 0] = 4

    assert_equal(slice[0, 0], 4)
    assert_equal(tensor[1, 0], 4)


def test_unsafe_slice():
    var dev = cpu_device()

    var shape = Index(10, 2)
    var dt = dev.allocate(TensorSpec(DType.float32, shape))
    var tensor = dt^.to_tensor[DType.float32, shape.size]()

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var unsafe_slice = UnsafeTensorSlice[DType.float32, 2](
        tensor.unsafe_ptr(), shape
    )

    assert_equal(unsafe_slice[1, 1], 4)

    _ = tensor^


def test_unsafe_slice_from_tensor():
    var dev = cpu_device()

    var shape = Index(10, 1)
    var dt = dev.allocate(TensorSpec(DType.float32, shape))
    var tensor = dt^.to_tensor[DType.float32, shape.size]()

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[i, j] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var unsafe_slice = tensor.unsafe_slice(
        slice(1, None, 1), slice(None, None, None)
    )

    var slice = tensor[1::, ::]
    assert_equal(unsafe_slice[0, 0], 3)
    assert_equal(slice[0, 0], 3)

    unsafe_slice[Index(0, 0)] = 4
    slice[0, 1] = 5

    assert_equal(unsafe_slice[0, 0], 4)
    assert_equal(unsafe_slice[0, 1], 5)
    assert_equal(slice[0, 0], 4)
    assert_equal(slice[0, 1], 5)
    assert_equal(tensor[1, 0], 4)
    assert_equal(tensor[1, 1], 5)  # keeps slice alive


def test_kv_cache():
    cpu = cpu_device()
    alias type = DType.float32
    alias shape = (2,)
    tensors = List[Tensor[type, len(shape)]]()
    for _ in range(2):
        dt = cpu.allocate(TensorSpec(type, shape))
        tensor = dt.to_tensor[type, len(shape)]()
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
    dev = cpu_device()

    alias type = DType.float32
    alias shape = (1,)
    dt = dev.allocate(
        TensorSpec(type, shape),
    )

    t = dt^.to_tensor[DType.float32, len(shape)]()
    ptr = t.unsafe_ptr()
    t[0] = 22
    assert_equal(Scalar.load(ptr), t[0])


def main():
    test_tensor()
    test_tensor_slice()
    test_unsafe_slice()
    test_unsafe_slice_from_tensor()
    test_slice_with_step()
    test_2dslice_with_step()
    test_4dslice_with_step()
    test_2dslice_with_step_row_column()
    test_4dslice_with_step()
    test_round_trip()
    test_copy()
    test_set_through_slice()
    test_kv_cache()
    test_raw_data()
