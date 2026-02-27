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

from sys import size_of, simd_width_of
from itertools import product

from layout import Coord, Idx, TileTensor
from layout._layout import row_major
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
comptime test_gpu_counts = (2, 4, 8)


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
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert rank == 1, "this test code currently assumes rank 1"

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

    # Synchronize all devices before reduce-scatter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime InputTileType = type_of(
        TileTensor[mut=False](
            in_bufs_list[0].unsafe_ptr(), row_major(Idx(length))
        )
    )
    var in_bufs = InlineArray[InputTileType, ngpus](uninitialized=True)

    comptime OutputTileType = type_of(
        TileTensor[mut=True](
            out_bufs_list[0].unsafe_ptr(),
            row_major(Idx(output_lengths[0])),
        )
    )
    var out_bufs = StaticTuple[OutputTileType, ngpus]()

    for i in range(ngpus):
        in_bufs[i] = InputTileType(
            in_bufs_list[i].unsafe_ptr(),
            row_major(Idx(length)),
        )

        out_bufs[i] = OutputTileType(
            out_bufs_list[i].unsafe_ptr(),
            row_major(Idx(output_lengths[i])),
        )

    # Custom epilogue that negates values to distinguish from default
    @always_inline
    @parameter
    @__copy_capture(out_bufs)
    fn outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None where (
        coords.flat_rank == 1
    ):
        out_bufs[input_index].store[width=_width, alignment=_alignment](
            coords,
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


fn run_reducescatter_sweep() raises:
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


fn reducescatter_axis_test[
    dtype: DType,
    ngpus: Int,
    axis: Int,
    use_custom_epilogue: Bool = False,
](list_of_ctx: List[DeviceContext], M: Int, D: Int) raises:
    """Test 2D axis-aware reduce-scatter.

    Each GPU gets a partition of the 2D (M, D) tensor along the specified axis.
    Verifies that each output element equals the sum of corresponding elements
    across all GPUs. When use_custom_epilogue is True, tests with a negating
    epilogue.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert axis == 0 or axis == 1
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    print(
        String(
            "====reducescatter-axis",
            axis,
            "-",
            dtype,
            "-",
            ngpus,
            "gpus-(",
            M,
            "x",
            D,
            ")-custom_epilogue=" if use_custom_epilogue else ")",
        )
    )

    # Compute partitioning to match what reducescatter entry point does.
    var axis_size: Int
    var unit_numel: Int
    if axis == 0:
        axis_size = M
        unit_numel = D
    else:
        axis_size = D // simd_width
        unit_numel = M * simd_width

    var config = ReduceScatterConfig[dtype, ngpus](axis_size, unit_numel, 0)

    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[UnsafePointer[Scalar[dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    for gpu_idx in range(ngpus):
        in_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](M * D)
        )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                config.rank_num_elements(gpu_idx)
            )
        )

        var h = alloc[Scalar[dtype]](M * D)
        host_in.append(h)
        for row in range(M):
            for col in range(D):
                h[row * D + col] = test_value_for_gpu_element[dtype](
                    gpu_idx, row * D + col
                )

        list_of_ctx[gpu_idx].enqueue_copy(in_bufs_list[gpu_idx], h)

        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                size_of[Signal]()
            )
        )
        list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
            signal_buffers[gpu_idx], 0
        )
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
        )

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Create 2D TileTensors.
    comptime InputTileType = type_of(
        TileTensor[mut=False](
            in_bufs_list[0].unsafe_ptr(), row_major((Idx(M), Idx(D)))
        )
    )
    var in_bufs = InlineArray[InputTileType, ngpus](uninitialized=True)

    comptime OutputTileType = type_of(
        TileTensor[mut=True](
            out_bufs_list[0].unsafe_ptr(), row_major((Idx(M), Idx(D)))
        )
    )
    var out_bufs = StaticTuple[OutputTileType, ngpus]()

    for i in range(ngpus):
        in_bufs[i] = InputTileType(
            in_bufs_list[i].unsafe_ptr(), row_major((Idx(M), Idx(D)))
        )

        if axis == 0:
            var my_rows = config.rank_units(i)
            out_bufs[i] = OutputTileType(
                out_bufs_list[i].unsafe_ptr(),
                row_major((Idx(my_rows), Idx(D))),
            )
        else:
            var my_cols = config.rank_units(i) * simd_width
            out_bufs[i] = OutputTileType(
                out_bufs_list[i].unsafe_ptr(),
                row_major((Idx(M), Idx(my_cols))),
            )

    # Custom epilogue that negates values to distinguish from default.
    @always_inline
    @parameter
    @__copy_capture(out_bufs)
    fn outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None where (
        coords.flat_rank == 2
    ):
        out_bufs[input_index].store[width=_width, alignment=_alignment](
            coords,
            rebind[SIMD[dtype, _width]](-val),
        )

    comptime for i in range(ngpus):
        reducescatter[
            ngpus=ngpus,
            axis=axis,
            output_lambda = Optional[elementwise_epilogue_type](
                outputs_lambda[input_index=i]
            ) if use_custom_epilogue else None,
        ](in_bufs, out_bufs[i], rank_sigs, list_of_ctx[i])

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results.
    for gpu_idx in range(ngpus):
        var out_size = config.rank_num_elements(gpu_idx)
        var result_host = alloc[Scalar[dtype]](out_size)
        list_of_ctx[gpu_idx].enqueue_copy(result_host, out_bufs_list[gpu_idx])
        list_of_ctx[gpu_idx].synchronize()

        if axis == 0:
            var row_start = config.rank_unit_start(gpu_idx)
            var my_rows = config.rank_units(gpu_idx)
            for r in range(my_rows):
                for c in range(D):
                    var global_flat = (row_start + r) * D + c
                    comptime accum_t = get_accum_type[dtype]()
                    var accum = Scalar[accum_t](0)
                    comptime for k in range(ngpus):
                        accum += Scalar[accum_t](
                            test_value_for_gpu_element[dtype](k, global_flat)
                        )
                    var expected_sum = Scalar[dtype](accum)
                    var expected = (
                        -expected_sum if use_custom_epilogue else expected_sum
                    )
                    assert_almost_equal(
                        result_host[r * D + c],
                        expected,
                        msg=String(
                            "GPU ",
                            gpu_idx,
                            " axis=0 (",
                            r,
                            ",",
                            c,
                            ") mismatch",
                        ),
                    )
        else:
            var col_start = config.rank_unit_start(gpu_idx) * simd_width
            var my_cols = config.rank_units(gpu_idx) * simd_width
            for r in range(M):
                for c in range(my_cols):
                    var global_flat = r * D + (col_start + c)
                    comptime accum_t = get_accum_type[dtype]()
                    var accum = Scalar[accum_t](0)
                    comptime for k in range(ngpus):
                        accum += Scalar[accum_t](
                            test_value_for_gpu_element[dtype](k, global_flat)
                        )
                    var expected_sum = Scalar[dtype](accum)
                    var expected = (
                        -expected_sum if use_custom_epilogue else expected_sum
                    )
                    assert_almost_equal(
                        result_host[r * my_cols + c],
                        expected,
                        msg=String(
                            "GPU ",
                            gpu_idx,
                            " axis=1 (",
                            r,
                            ",",
                            c,
                            ") mismatch",
                        ),
                    )

        result_host.free()

    for i in range(ngpus):
        host_in[i].free()


# 2D test shapes: (M, D)
comptime test_2d_shapes = (
    (16, 128),
    (15, 128),  # Ragged rows (odd number not divisible by ngpus)
    (9, 7168),  # Ragged, Deepseek V3.1 token dim
    (32, 256),
)


fn run_reducescatter_axis_sweep() raises:
    """Run a sweep of 2D axis-aware reduce-scatter tests."""
    var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
    for i in range(DeviceContext.number_of_devices()):
        list_of_ctx.append(DeviceContext(i))

    comptime for dtype_idx, ngpus_idx, shape_idx, epilogue_idx in product(
        range(len(test_dtypes)),
        range(len(test_gpu_counts)),
        range(len(test_2d_shapes)),
        range(2),
    ):
        comptime dtype = test_dtypes[dtype_idx]
        comptime ngpus = test_gpu_counts[ngpus_idx]
        comptime M = test_2d_shapes[shape_idx][0]
        comptime D = test_2d_shapes[shape_idx][1]
        comptime use_custom_epilogue = epilogue_idx == 1

        if DeviceContext.number_of_devices() < ngpus:
            continue

        # axis=0: scatter rows
        reducescatter_axis_test[
            dtype=dtype,
            ngpus=ngpus,
            axis=0,
            use_custom_epilogue=use_custom_epilogue,
        ](list_of_ctx, M, D)

        # axis=1: scatter columns
        reducescatter_axis_test[
            dtype=dtype,
            ngpus=ngpus,
            axis=1,
            use_custom_epilogue=use_custom_epilogue,
        ](list_of_ctx, M, D)


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

    # Run standard 1D reduce-scatter sweep
    run_reducescatter_sweep()

    # Run 2D axis-aware reduce-scatter sweep
    run_reducescatter_axis_sweep()

    print("All reduce-scatter tests passed!")
