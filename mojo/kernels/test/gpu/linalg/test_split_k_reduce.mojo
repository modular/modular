# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import isclose
from buffer import Dim, DimList, NDBuffer
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import split_k_reduce
from utils import StaticIntTuple
from testing import assert_almost_equal
from random import rand
from memory import memcpy


fn _size[rank: Int](dims: StaticIntTuple[rank]) -> Int:
    var size = 1

    @parameter
    for i in range(rank):
        size *= dims[i]
    return size


fn _create_device_buffer[
    dtype: DType, rank: Int, shape: DimList
](ctx: DeviceContext, dynamic_shape: StaticIntTuple[rank]) raises -> Tuple[
    DeviceBuffer[dtype], NDBuffer[dtype, rank, shape]
]:
    var storage = ctx.create_buffer[dtype](_size(dynamic_shape))
    return (
        storage,
        NDBuffer[dtype, rank, shape](storage.ptr, dynamic_shape=dynamic_shape),
    )


fn _create_host_buffer[
    dtype: DType, rank: Int, shape: DimList
](dynamic_shape: StaticIntTuple[rank]) raises -> NDBuffer[dtype, rank, shape]:
    var storage_ptr = UnsafePointer[Scalar[dtype]].alloc(_size(dynamic_shape))
    return NDBuffer[dtype, rank, shape](
        storage_ptr, dynamic_shape=dynamic_shape
    )


fn _get_test_name[
    type: DType, shape_a: DimList, shape_b: DimList
](shape_a_dim: StaticIntTuple[2], shape_b_dim: StaticIntTuple[2],) -> String:
    var test_str = String("test-case(")
    test_str += str(type)
    test_str += ") : "
    test_str += "a -> " + str(shape_a_dim) + " and "
    test_str += "b -> " + str(shape_b_dim)
    return test_str


fn _split_k_reduce_verify[
    type: DType, a_shape: DimList, b_shape: DimList
](
    inout A: NDBuffer[type, 2, a_shape],
    borrowed B: NDBuffer[type, 2, b_shape],
    num_partition: UInt,
):
    var M = A.dim[0]()
    var N = A.dim[1]()

    for i in range(M):
        for j in range(N):
            var idx = StaticIntTuple[2]((i, j))
            var vec = A[idx]
            for k in range(num_partition):
                vec += B[i, j + k * N]
            A.store(idx, vec)


fn split_k_reduce_test_case[
    dtype: DType, shape_a: DimList, shape_b: DimList, num_partition: UInt
](
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
    ctx: DeviceContext,
) raises:
    print(
        _get_test_name[dtype, shape_a, shape_b](shape_a_dim, shape_b_dim),
        end=" ",
    )

    var mat_a_dev = _create_device_buffer[dtype, 2, shape_a](ctx, shape_a_dim)
    var mat_b_dev = _create_device_buffer[dtype, 2, shape_b](ctx, shape_b_dim)
    var mat_a_host = _create_host_buffer[dtype, 2, shape_a](shape_a_dim)
    var mat_b_host = _create_host_buffer[dtype, 2, shape_b](shape_b_dim)
    var mat_a_ref = _create_host_buffer[dtype, 2, shape_a](shape_a_dim)

    alias a_row = shape_a.at[0]().get()
    alias a_col = shape_a.at[1]().get()
    alias b_row = shape_b.at[0]().get()
    alias b_col = shape_b.at[1]().get()

    rand[dtype](mat_a_host.data, a_row * a_col)
    rand[dtype](mat_b_host.data, b_row * b_col)
    memcpy[a_row * a_col](mat_a_ref.data, mat_a_host.data)

    _split_k_reduce_verify(mat_a_ref, mat_b_host, num_partition)

    ctx.enqueue_copy_to_device(mat_a_dev[0], mat_a_host.data)
    ctx.enqueue_copy_to_device(mat_b_dev[0], mat_b_host.data)

    split_k_reduce(mat_a_dev[1], mat_b_dev[1], num_partition, ctx)

    ctx.enqueue_copy_from_device(mat_a_host.data, mat_a_dev[0])

    ctx.synchronize()

    for m in range(shape_a_dim[0]):
        for n in range(shape_a_dim[1]):
            assert_almost_equal(mat_a_host[m, n], mat_a_ref[m, n])

    mat_a_host.data.free()
    mat_b_host.data.free()
    mat_a_ref.data.free()
    _ = mat_a_dev^
    _ = mat_b_dev^


def main():
    with DeviceContext() as ctx:
        alias num_part1 = 2
        split_k_reduce_test_case[
            DType.float32, DimList(2, 2), DimList(2, 2 * num_part1), num_part1
        ](StaticIntTuple[2](2, 2), StaticIntTuple[2](2, 2 * num_part1), ctx)

        alias num_part2 = 3
        split_k_reduce_test_case[
            DType.float32,
            DimList(16, 16),
            DimList(16, 16 * num_part2),
            num_part2,
        ](StaticIntTuple[2](16, 16), StaticIntTuple[2](16, 16 * num_part2), ctx)

        # test non-square dimension
        split_k_reduce_test_case[
            DType.float32,
            DimList(2, 4),
            DimList(2, 4 * num_part1),
            num_part1,
        ](StaticIntTuple[2](2, 4), StaticIntTuple[2](2, 4 * num_part1), ctx)

        split_k_reduce_test_case[
            DType.float32,
            DimList(16, 8),
            DimList(16, 8 * num_part2),
            num_part2,
        ](StaticIntTuple[2](16, 8), StaticIntTuple[2](16, 8 * num_part2), ctx)
