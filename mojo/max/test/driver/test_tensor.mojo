# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from driver import CPUDescriptor, cpu_device
from driver import Tensor, TensorSlice
from tensor import TensorSpec
from utils import Index
from testing import assert_equal, assert_raises


def test_tensor():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    tensor[Index(0, 0)] = 0
    tensor[Index(0, 1)] = 1
    tensor[Index(1, 0)] = 2
    tensor[Index(1, 1)] = 3

    assert_equal(tensor[0, 0], 0)
    assert_equal(tensor[1, 1], 3)


def test_tensor_slice():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 3, 3),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    for i in range(3):
        for j in range(3):
            tensor[Index(i, j)] = i + j

    # tensor
    # 0 1 2
    # 1 2 3
    # 2 3 4

    var slice = tensor[0:2, :]

    # slice
    # 0 1 2
    # 1 2 3

    var slice_spec = slice.get_static_spec()
    assert_equal(slice_spec.get_rank(), 2)
    assert_equal(slice_spec[0], 2)
    assert_equal(slice_spec[1], 3)

    assert_equal(slice[0, 0], 0)
    assert_equal(slice[1, 2], 3)

    var inner_slice = tensor[0:2, 1:2]

    # inner_slice
    # 1
    # 2

    var inner_slice_spec = inner_slice.get_static_spec()
    assert_equal(inner_slice_spec.get_rank(), 2)
    assert_equal(inner_slice_spec[0], 2)
    assert_equal(inner_slice_spec[1], 1)

    assert_equal(inner_slice[0, 0], 1)
    assert_equal(inner_slice[1, 0], 2)


def test_slice_with_step():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 18),
    )
    var tensor = dt^.get_tensor[DType.float32, 1]()

    var index = 0
    for _ in range(3):
        for j in range(6):
            tensor[index] = j
            index += 1

    assert_equal(tensor[1], 1)

    var stepped_slice = tensor[0:18:3]
    assert_equal(stepped_slice[1], 3)


def test_2dslice_with_step():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[Index(i, j)] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var stepped_slice = tensor[::2, :]
    assert_equal(stepped_slice[1, 0], 5)


def test_2dslice_with_step_row_column():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 10, 10),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(10):
            tensor[Index(i, j)] = val
            val += 1

    assert_equal(tensor[1, 0], 11)
    assert_equal(tensor[1, 1], 12)
    assert_equal(tensor[2, 2], 23)

    var stepped_slice = tensor[1::2, 1::2]
    assert_equal(stepped_slice[0, 0], 12)
    assert_equal(stepped_slice[0, 1], 14)
    assert_equal(stepped_slice[1, 1], 34)

    var slice_spec = stepped_slice.get_static_spec()
    assert_equal(slice_spec.get_rank(), 2)
    assert_equal(slice_spec.dtype(), DType.float32)
    assert_equal(slice_spec[0], 5)
    assert_equal(slice_spec[1], 5)

    var inner_slice = tensor[1::3, 1::3]
    var inner_slice_spec = inner_slice.get_static_spec()
    assert_equal(inner_slice_spec.get_rank(), 2)
    assert_equal(inner_slice_spec.dtype(), DType.float32)
    assert_equal(inner_slice_spec[0], 3)
    assert_equal(inner_slice_spec[1], 3)
    assert_equal(inner_slice[0, 0], 12)
    assert_equal(inner_slice[1, 1], 45)


def test_round_trip():
    var dev = cpu_device()

    var dt = dev.allocate(TensorSpec(DType.float32, 10, 2), str("mytensor"))
    var tensor = dt^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[Index(i, j)] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var dt2 = tensor^.get_device_memory()
    assert_equal(
        str(dt2),
        "DeviceMemory(mytensor,Device(type=CPU),Spec(10x2xfloat32))",
    )

    var tensor2 = dt2^.get_tensor[DType.float32, 2]()
    assert_equal(tensor2[1, 0], 3)


def test_copy():
    var cpu1 = cpu_device(CPUDescriptor(numa_id=0))
    var cpu2 = cpu_device(CPUDescriptor(numa_id=1))

    var src_dev_tensor = cpu1.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    var dst_dev_tensor = cpu2.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    var src = src_dev_tensor^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            src[Index(i, j)] = val
            val += 1

    src_dev_tensor = src^.get_device_memory()
    src_dev_tensor.copy_into(dst_dev_tensor)
    var dst_dev_tensor2 = src_dev_tensor.copy_to(cpu2)

    var dst = dst_dev_tensor^.get_tensor[DType.float32, 2]()
    var dst2 = dst_dev_tensor2^.get_tensor[DType.float32, 2]()

    with assert_raises(contains="already allocated on"):
        _ = src_dev_tensor.copy_to(cpu1)

    src = src_dev_tensor^.get_tensor[DType.float32, 2]()
    for i in range(10):
        for j in range(2):
            assert_equal(src[i, j], dst[i, j])
            assert_equal(src[i, j], dst2[i, j])


def test_set_through_slice():
    var dev = cpu_device()

    var dt = dev.allocate(
        TensorSpec(DType.float32, 10, 2),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            tensor[Index(i, j)] = val
            val += 1

    assert_equal(tensor[1, 0], 3)

    var slice = tensor[1:, :]
    assert_equal(slice[0, 0], 3)

    slice[(0, 0)] = 4

    assert_equal(slice[0, 0], 4)
    assert_equal(tensor[1, 0], 4)


def test_kv_cache():
    var cpu = cpu_device()
    alias type = DType.float32
    alias shape = (2,)
    var tensors = List[Tensor[type, len(shape)]]()
    for _ in range(2):
        var dt = cpu.allocate(TensorSpec(type, shape))
        var tensor = dt.get_tensor[type, len(shape)]()
        tensor[0] = 1
        tensor[1] = 2

    for t in tensors:
        assert_equal(
            t[]._ptr[0],
            1,
        )
        assert_equal(
            t[]._ptr[1],
            2,
        )


def test_raw_data():
    var dev = cpu_device()

    alias type = DType.float32
    alias shape = (1,)
    var dt = dev.allocate(
        TensorSpec(type, shape),
    )

    var t = dt^.get_tensor[DType.float32, len(shape)]()
    var ptr = t.unsafe_ptr()
    t[0] = 22
    assert_equal(Scalar.load(ptr), t[0])


def main():
    test_tensor()
    test_tensor_slice()
    test_slice_with_step()
    test_2dslice_with_step()
    test_2dslice_with_step_row_column()
    test_round_trip()
    test_copy()
    test_set_through_slice()
    test_kv_cache()
    test_raw_data()
