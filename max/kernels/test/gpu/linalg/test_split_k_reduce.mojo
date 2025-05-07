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
# RUN: %mojo-no-debug %s

from math import isclose
from random import rand

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import split_k_reduce
from linalg.utils_gpu import MatmulConfig
from memory import UnsafePointer, memcpy
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index


fn _size[rank: Int](dims: IndexList[rank]) -> Int:
    var size = 1

    @parameter
    for i in range(rank):
        size *= dims[i]
    return size


fn _create_device_buffer[
    dtype: DType, rank: Int, shape: DimList
](ctx: DeviceContext, dynamic_shape: IndexList[rank]) raises -> Tuple[
    DeviceBuffer[dtype], NDBuffer[dtype, rank, MutableAnyOrigin, shape]
]:
    var storage = ctx.enqueue_create_buffer[dtype](_size(dynamic_shape))
    return (
        storage,
        NDBuffer[dtype, rank, _, shape](
            storage._unsafe_ptr(), dynamic_shape=dynamic_shape
        ),
    )


fn _create_host_buffer[
    dtype: DType, rank: Int, shape: DimList
](dynamic_shape: IndexList[rank]) raises -> NDBuffer[
    dtype, rank, MutableAnyOrigin, shape
]:
    var storage_ptr = UnsafePointer[Scalar[dtype]].alloc(_size(dynamic_shape))
    return NDBuffer[dtype, rank, _, shape](
        storage_ptr, dynamic_shape=dynamic_shape
    )


fn _get_test_name[
    type: DType, shape_a: DimList, shape_b: DimList
](shape_a_dim: IndexList[2], shape_b_dim: IndexList[2],) -> String:
    return String(
        "test-case(", type, ") : a -> ", shape_a_dim, " and b ->", shape_b_dim
    )


fn _split_k_reduce_verify[
    type: DType, a_shape: DimList, b_shape: DimList
](
    mut A: NDBuffer[mut=True, type, 2, _, a_shape],
    B: NDBuffer[type, 2, _, b_shape],
    num_partition: UInt,
):
    var M = A.dim[0]()
    var N = A.dim[1]()

    for i in range(M):
        for j in range(N):
            var idx = IndexList[2]((i, j))
            var vec = A[idx]
            for k in range(num_partition):
                vec += B[i, j + k * N]
            A.store(idx, vec)


def test_split_k_reduce_rank3[
    c_type: DType,
    work_space_type: DType,
](M: Int, N: Int, num_partitions: Int, ctx: DeviceContext):
    print(
        "test_split_k_reduce_rank3",
        work_space_type,
        "->",
        c_type,
        num_partitions,
        M,
        N,
    )

    var c_host = UnsafePointer[Scalar[c_type]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[c_type]].alloc(M * N)

    # Randome buffer for host computation.
    var epilogue_data_host = UnsafePointer[Scalar[c_type]].alloc(M * N)
    rand[c_type](epilogue_data_host, M * N)

    var work_space_size = num_partitions * M * N
    var work_space_host = UnsafePointer[Scalar[work_space_type]].alloc(
        work_space_size
    )
    rand[work_space_type](work_space_host, work_space_size)

    # Naive host reduction. The accumulation is in FP32 since CPU may not have
    # native BF16 instructions.
    for i in range(M * N):
        var sum = work_space_host[i].cast[DType.float32]()
        for j in range(1, num_partitions):
            sum += work_space_host[i + j * M * N].cast[DType.float32]()
        sum += epilogue_data_host[i].cast[DType.float32]()
        c_host_ref[i] = sum.cast[c_type]()

    var work_space_device = ctx.enqueue_create_buffer[work_space_type](
        num_partitions * M * N
    )
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var epilogue_data_device = ctx.enqueue_create_buffer[c_type](M * N)

    ctx.enqueue_copy(work_space_device, work_space_host)
    ctx.enqueue_copy(epilogue_data_device, epilogue_data_host)

    var c = NDBuffer[c_type, 2](c_device._unsafe_ptr(), Index(M, N))
    var work_space = NDBuffer[work_space_type, 3](
        work_space_device._unsafe_ptr(), Index(num_partitions, M, N)
    )
    var epilogue_buffer = NDBuffer[c_type, 2](
        epilogue_data_device._unsafe_ptr(), Index(M, N)
    )

    @parameter
    @always_inline
    @__copy_capture(c, epilogue_buffer)
    fn epilogue_fn[
        _type: DType, _width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_type, _width]) capturing -> None:
        var another_val = rebind[SIMD[_type, _width]](
            epilogue_buffer.load[width=_width](idx)
        )
        c.store(idx, rebind[SIMD[c_type, _width]](val + another_val))

    split_k_reduce[elementwise_lambda_fn=epilogue_fn](c, work_space, ctx)

    ctx.enqueue_copy(c_host, c_device)

    alias rtol = 1e-4 if c_type is DType.float32 else 1e-2
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=rtol):
            print(
                i,
                c_host[i],
                c_host_ref[i],
                abs((c_host[i] - c_host_ref[i]) / c_host_ref[i]),
            )
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=rtol)

    _ = c
    _ = work_space
    _ = epilogue_buffer
    _ = epilogue_data_device
    _ = c_device
    _ = work_space_device

    c_host.free()
    c_host_ref.free()
    epilogue_data_host.free()
    work_space_host.free()


def main():
    with DeviceContext() as ctx:
        # Rank-3 work space.
        test_split_k_reduce_rank3[DType.bfloat16, DType.bfloat16](
            64, 64, 2, ctx
        )
        test_split_k_reduce_rank3[DType.bfloat16, DType.float32](32, 32, 5, ctx)
        test_split_k_reduce_rank3[DType.float32, DType.float32](32, 64, 3, ctx)
