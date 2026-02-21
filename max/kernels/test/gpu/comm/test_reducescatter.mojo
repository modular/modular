# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from sys import size_of
from itertools import product

from buffer import NDBuffer
from buffer.dimlist import DimList
from collections import Optional
from comm import Signal, MAX_GPUS
from comm.reducescatter import (
    reducescatter,
    ReduceScatterConfig,
    elementwise_epilogue_type,
)
from internal_utils import human_readable_size
from comm_test_utils import test_value_for_gpu_element
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from testing import assert_almost_equal, assert_true
from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

# Shared test configurations
comptime test_lengths = (
    8 * 1024,  # Small
    8 * 1024 + 8,  # Ragged: +1 bf16 SIMD vector / +2 f32 SIMD vectors
    8 * 1024 + 24,  # Ragged: +3 bf16 SIMD vectors / +6 f32 SIMD vectors
    256 * 1024,  # Medium
    16 * 1024 * 1024,  # Large
    16 * 1024 * 1024 + 8,  # Ragged: +1 bf16 SIMD vector / +2 f32 SIMD vectors
    16 * 1024 * 1024 + 24,  # Ragged: +3 bf16 SIMD vectors / +6 f32 SIMD vectors
)

# Test hyperparameters
comptime test_dtypes = (DType.bfloat16, DType.float32)
comptime test_gpu_counts = (2, 4)


fn reducescatter_test[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    use_custom_epilogue: Bool = False,
](list_of_ctx: List[DeviceContext], length: Int) raises:
    """Test reduce-scatter operation.

    Each GPU receives 1/ngpus of the reduced data in its output partition.
    When use_custom_epilogue is True, tests with a negating epilogue.
    """
    constrained[ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"]()
    constrained[rank == 1, "this test code currently assumes rank 1"]()

    print(
        String(
            "====reducescatter-",
            dtype,
            "-",
            ngpus,
            "-",
            human_readable_size(size_of[dtype]() * length),
            "-custom_epilogue=" if use_custom_epilogue else "",
        )
    )

    # Compute partition sizes matching ReduceScatterConfig logic.
    # Lower ranks get an extra simd vector when there's a remainder.
    var rs_config = ReduceScatterConfig[dtype, ngpus](
        length, 0
    )  # dummy num_threads

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var output_lengths = List[Int](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    # Initialize buffers for each GPU
    for i in range(ngpus):
        var output_length = rs_config.rank_part(i)
        output_lengths.append(output_length)

        # Create input and output device buffers
        in_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        out_bufs_list.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](output_length)
        )

        # Create and initialize host buffers with unique values per GPU
        var host_buffer = alloc[Scalar[dtype]](length)
        host_buffers.append(host_buffer)

        # Initialize with unique per-GPU, per-element values for thorough testing
        for j in range(length):
            host_buffer[j] = test_value_for_gpu_element[dtype](i, j)

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](size_of[Signal]())
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

        # Copy data to device
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create input and output NDBuffers
    var in_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[dtype, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(output_lengths[i])
        )

    # Synchronize all devices before reduce-scatter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, rank, MutAnyOrigin], ngpus
    ](NDBuffer[dtype, rank, MutAnyOrigin]())

    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(output_lengths[i])
        )

    # Custom epilogue that negates values to distinguish from default
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
            rebind[IndexList[rank]](coords),
            rebind[SIMD[dtype, _width]](
                -val
            ),  # Negate to distinguish from default
        )

    # Perform reduce-scatter
    comptime for i in range(ngpus):
        reducescatter[
            ngpus=ngpus,
            output_lambda = Optional[elementwise_epilogue_type](
                outputs_lambda[input_index=i]
            ) if use_custom_epilogue else None,
        ](in_bufs, out_bufs[i], rank_sigs, list_of_ctx[i])

    # Synchronize all devices after reduce-scatter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results:
    # For each element j in GPU gpu_idx's output, we sum across all GPUs
    # at the global index rank_start(gpu_idx) + j
    for gpu_idx in range(ngpus):
        var gpu_output_length = output_lengths[gpu_idx]
        var result_host = alloc[Scalar[dtype]](gpu_output_length)
        list_of_ctx[gpu_idx].enqueue_copy(result_host, out_bufs_list[gpu_idx])
        list_of_ctx[gpu_idx].synchronize()

        # Verify each element in this GPU's partition
        for j in range(gpu_output_length):
            # Compute expected value: sum of test values across all GPUs
            # at global index rank_start(gpu_idx) + j
            # Use higher precision accumulation like allreduce does
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)
            var global_idx = rs_config.rank_start(gpu_idx) + j

            comptime for k in range(ngpus):
                var term_dtype = test_value_for_gpu_element[dtype](
                    k, global_idx
                )
                accum += Scalar[accum_t](term_dtype)
            var expected_sum = Scalar[dtype](accum)

            # Custom epilogue negates the result
            var expected = (
                -expected_sum if use_custom_epilogue else expected_sum
            )

            var actual = result_host[j]
            assert_almost_equal(
                actual,
                expected,
                msg=String(
                    "GPU ",
                    gpu_idx,
                    " partition element ",
                    j,
                    " (global ",
                    global_idx,
                    ") mismatch",
                ),
            )

        result_host.free()

    # Clean up
    for i in range(ngpus):
        host_buffers[i].free()


fn run_reducescatter_sweep[
    use_multimem: Bool = False,
]() raises:
    """Run a sweep of reduce-scatter tests across configurations."""
    var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
    for i in range(DeviceContext.number_of_devices()):
        list_of_ctx.append(DeviceContext(i))

    comptime for dtype_idx, ngpus_idx, length_idx, epilogue_idx in product(
        range(len(test_dtypes)),
        range(len(test_gpu_counts)),
        range(len(test_lengths)),
        range(2),  # Test both default and custom epilogue
    ):
        comptime dtype = test_dtypes[dtype_idx]
        comptime ngpus = test_gpu_counts[ngpus_idx]
        comptime length = test_lengths[length_idx]
        comptime use_custom_epilogue = epilogue_idx == 1

        if DeviceContext.number_of_devices() < ngpus:
            continue

        reducescatter_test[
            dtype=dtype,
            rank=1,
            ngpus=ngpus,
            use_custom_epilogue=use_custom_epilogue,
        ](list_of_ctx, length)


def main():
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

    # Run standard reduce-scatter sweep
    run_reducescatter_sweep()

    print("All reduce-scatter tests passed!")
