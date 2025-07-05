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

from sys import sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.comm.allreduce import (
    MAX_GPUS,
    Signal,
    allreduce,
)
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils._utils import ValOrDim, dynamic, static
from testing import assert_almost_equal
from utils import IndexList, StaticTuple

from linalg.distributed_matmul import matmul_allreduce

alias overlap_with_dpl = True


fn overlap_matmul_allreduce_test[
    dtype: DType,
    ngpus: Int,
    partition_dim: Int,
    num_partitions: Int,
](
    list_of_ctx: List[DeviceContext], m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    # Demonstrate overlap matmul with allreduce.
    # The matmul is sharded in K dim. The original matmul before sharding is M x N x (K x ngpus).
    # The results of shape M x N is allreduced over ngpus.

    # To overlap, we partition in M dimension. The matmul can overlap with the allreduce
    # for previous partition.
    #      matmul part 0
    #      matmul part 1 | allreduce partition 0
    #      matmul part 2 | allreduce partition 1
    #      matmul part 3 | allreduce partition 2
    #                      allreduce partition 3
    #
    # Other than allreduce i depending on matmul i, there is no dependence. The best
    # performance is obtained by letting allreduce i wait on matmul i but launch
    # matmul i+1 asap. Matmul doesn't need to wait for any kernel.
    constrained[ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"]()

    print(
        "num_gpus",
        ngpus,
        "m",
        m.value,
        "n",
        n.value,
        "k",
        k.value,
        "stages",
        num_partitions,
        "split dim",
        partition_dim,
    )

    # Create matmul input and output buffers.
    var A_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var B_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var C_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var C_reduced_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var A_host_list = List[UnsafePointer[Scalar[dtype]]](capacity=ngpus)
    var B_host_list = List[UnsafePointer[Scalar[dtype]]](capacity=ngpus)
    var C_reduced_host_list = List[UnsafePointer[Scalar[dtype]]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](
        UnsafePointer[Signal]()
    )

    var mn = m.value * n.value
    var mk = m.value * k.value
    var nk = n.value * k.value

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    # Partitioned matmul has size M * N / num_partitions
    var temp_length = m.value * n.value // num_partitions
    var temp_buffer_num_bytes = ngpus * sizeof[dtype]() * temp_length

    # Initialize buffers for each GPU
    @parameter
    for i in range(ngpus):
        # Allocate A. B, C on device for matmul.
        A_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](mk))
        B_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](nk))
        C_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](mn))
        C_reduced_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](mn))

        # Allocate matmul inputs A B and final output C_reduced on host
        A_host_list.append(UnsafePointer[Scalar[dtype]].alloc(mk))
        B_host_list.append(UnsafePointer[Scalar[dtype]].alloc(nk))
        C_reduced_host_list.append(UnsafePointer[Scalar[dtype]].alloc(mn))

        # Initialize A with i and B with 1
        for j in range(mk):
            A_host_list[i][j] = i
        for j in range(nk):
            B_host_list[i][j] = 1.0

        # Copy A and B to device
        list_of_ctx[i].enqueue_copy(A_list[i], A_host_list[i])
        list_of_ctx[i].enqueue_copy(B_list[i], B_host_list[i])

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                sizeof[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

    # Create the list of NDBuffers.
    alias A_static_shape = DimList(m.dim, k.dim)
    alias B_static_shape = DimList(n.dim, k.dim)
    alias C_static_shape = DimList(m.dim, n.dim)
    var As = InlineArray[
        NDBuffer[dtype, 2, MutableAnyOrigin, A_static_shape], ngpus
    ](NDBuffer[dtype, 2, MutableAnyOrigin, A_static_shape]())
    var Bs = InlineArray[
        NDBuffer[dtype, 2, MutableAnyOrigin, B_static_shape], ngpus
    ](NDBuffer[dtype, 2, MutableAnyOrigin, B_static_shape]())
    var Cs = InlineArray[
        NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape], ngpus
    ](NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape]())
    var out_bufs = InlineArray[
        NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape], ngpus
    ](NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape]())

    # Setup the kernel NDBuffers
    @parameter
    for i in range(ngpus):
        As[i] = NDBuffer[dtype, 2, MutableAnyOrigin, A_static_shape](
            A_list[i].unsafe_ptr(), DimList(m.value, k.value)
        )
        Bs[i] = NDBuffer[dtype, 2, MutableAnyOrigin, B_static_shape](
            B_list[i].unsafe_ptr(), DimList(n.value, k.value)
        )
        Cs[i] = NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape](
            C_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )
        out_bufs[i] = NDBuffer[dtype, 2, MutableAnyOrigin, C_static_shape](
            C_reduced_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, 2, MutableAnyOrigin], ngpus
    ](NDBuffer[dtype, 2, MutableAnyOrigin]())

    @parameter
    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, 2](
            C_reduced_list[i].unsafe_ptr(), DimList(m.value, n.value)
        )

    # Prepare the output lambda
    @always_inline
    @parameter
    @__copy_capture(out_bufs_capture)
    fn outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        out_bufs_capture[input_index].store[width=_width, alignment=_alignment](
            rebind[IndexList[2]](coords), rebind[SIMD[dtype, _width]](val)
        )

    # Call the split matmul + AllReduce implementation
    for _ in range(10):

        @parameter
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        matmul_allreduce[
            ngpus=ngpus,
            partition_dim=partition_dim,
            outputs_lambda=outputs_lambda,
            overlap_with_dpl=overlap_with_dpl,
        ](
            As,
            Bs,
            Cs,
            out_bufs,
            rank_sigs,
            list_of_ctx,
            static[num_partitions](),
        )

    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Last allreduce
    var expected_sum = Scalar[dtype](0)

    @parameter
    for i in range(ngpus):
        expected_sum += i * k.value
        list_of_ctx[i].enqueue_copy(C_reduced_host_list[i], C_reduced_list[i])

    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results
    @parameter
    for i in range(ngpus):
        for j in range(mn):
            try:
                assert_almost_equal(C_reduced_host_list[i][j], expected_sum)
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", C_reduced_host_list[i][j])
                print("Expected:", expected_sum)
                raise e
    print("Verification passed")

    # Cleanup
    for i in range(ngpus):
        A_host_list[i].free()
        B_host_list[i].free()
        C_reduced_host_list[i].free()

    _ = signal_buffers^
    _ = A_list^
    _ = B_list^
    _ = C_list^
    _ = C_reduced_list^


def main():
    # Test hyperparameters.
    alias test_dtypes = (DType.bfloat16,)
    alias test_gpu_counts = (4, 8)

    # Run tests for each configuration.
    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        alias num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() < num_gpus:
            break

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        # Test all cases for this configuration.
        @parameter
        for dtype_idx in range(len(test_dtypes)):
            alias dtype = test_dtypes[dtype_idx]

            overlap_matmul_allreduce_test[
                dtype=dtype,
                ngpus=num_gpus,
                num_partitions=4,
                partition_dim=0,
            ](ctx, dynamic(8192), static[8192](), static[2048]())

            overlap_matmul_allreduce_test[
                dtype=dtype,
                ngpus=num_gpus,
                num_partitions=4,
                partition_dim=1,
            ](ctx, dynamic(8192), static[8192](), static[2048]())
