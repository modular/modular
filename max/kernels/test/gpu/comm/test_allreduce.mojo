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

import time
from sys import size_of, has_amd_gpu_accelerator, simd_width_of
from itertools import product

from buffer import NDBuffer
from buffer.dimlist import DimList
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.allreduce import (
    _allreduce_naive_single,
    allreduce,
    elementwise_epilogue_type,
)
import comm.vendor.ccl as vendor_ccl
from internal_utils import human_readable_size
from gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
)
from testing import assert_almost_equal, assert_true
from collections import Optional

from utils import IndexList, StaticTuple

# Shared test configurations
comptime test_lengths = (
    0,  # No elements
    8 * 1024,  # Small latency bound
    8 * 1024 + 8,  # Ragged: small +8-element offset over base
    8 * 1024 + 24,  # Ragged: larger +24-element offset over base
    128 * 1024,  # Larger latency bound
    256 * 1024,  # Smallest bandwidth bound
    16 * 1024 * 1024,  # Bandwidth bound
    16 * 1024 * 1024 + 8,  # Ragged: small +8-element offset over base
    16 * 1024 * 1024 + 24,  # Ragged: larger +24-element offset over base
    64 * 1024 * 1024,  # Bandwidth bound: 8192 chunk size at dim = 8192
)

# Test hyperparameters.
comptime test_dtypes = (DType.bfloat16, DType.float32)
comptime test_gpu_counts = (2, 4, 8)


fn allreduce_test[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    use_custom_epilogue: Bool = False,
](list_of_ctx: List[DeviceContext], length: Int) raises:
    # Using multimem with zero length raises CUDA_ERROR_INVALID_VALUE
    # when setting up the buffers.
    if use_multimem and length == 0:
        return

    comptime num_buffers = 1 if use_multimem else ngpus

    comptime assert ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"
    comptime assert rank == 1, "this test code currently assumes rank 1"

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length

    # Initialize buffers for each GPU
    for i in range(ngpus):
        # Create and store device buffers
        if not use_multimem:
            in_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[dtype](length)
            )
        out_bufs_list.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = alloc[Scalar[dtype]](length)
        host_buffers.append(host_buffer)

        # Initialize host buffer with values (i + 1).0
        var host_nd_buf = NDBuffer[dtype, rank](host_buffer, DimList(length))
        host_nd_buf.fill(Scalar[dtype](i + 1))

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

        # Copy data to device for non-multimem path
        if not use_multimem:
            list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create and initialize input and output buffers.
    var in_bufs = InlineArray[
        NDBuffer[dtype, rank, ImmutAnyOrigin], num_buffers
    ](fill={})
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )

    if use_multimem:
        var multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), length
        )
        for i in range(ngpus):
            var unicast_buf = multicast_buf.unicast_buffer_for(list_of_ctx[i])
            list_of_ctx[i].enqueue_copy(unicast_buf, host_buffers[i])
        # All GPUs use the same multicast pointer
        in_bufs[0] = NDBuffer[dtype, rank](
            multicast_buf.multicast_buffer_for(list_of_ctx[0]).unsafe_ptr(),
            DimList(length),
        )
    else:
        for i in range(ngpus):
            in_bufs[i] = NDBuffer[dtype, rank](
                in_bufs_list[i].unsafe_ptr(), DimList(length)
            )

    for i in range(ngpus):
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, rank, MutAnyOrigin], ngpus
    ](NDBuffer[dtype, rank, MutAnyOrigin]())

    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
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
                -val  # Negate to distinguish from default epilogue
            ),
        )

    # Precompute expected sum across GPUs for verification.
    var expected_sum = Scalar[dtype](0)
    for i in range(ngpus):
        expected_sum += Scalar[dtype](i + 1)

    group_start()

    comptime for i in range(ngpus):
        allreduce[
            ngpus=ngpus,
            output_lambda = Optional[elementwise_epilogue_type](
                outputs_lambda[input_index=i]
            ) if use_custom_epilogue else None,
            use_multimem=use_multimem,
        ](in_bufs, out_bufs[i], rank_sigs, list_of_ctx[i])
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Vendor RCCL comparison (non-multimem path only and only if available).
    comptime if not use_multimem and has_amd_gpu_accelerator():
        try:
            # Prepare distinct outputs for vendor path to avoid aliasing.
            var out_dev_vendor = List[DeviceBuffer[dtype]](capacity=ngpus)
            var out_bufs_vendor = InlineArray[
                NDBuffer[dtype, rank, MutAnyOrigin], ngpus
            ](fill={})
            for i in range(ngpus):
                out_dev_vendor.append(
                    list_of_ctx[i].enqueue_create_buffer[dtype](length)
                )
                out_bufs_vendor[i] = NDBuffer[dtype, rank](
                    out_dev_vendor[i].unsafe_ptr(), DimList(length)
                )

            # Test RCCL.
            with vendor_ccl.group():
                comptime for i in range(ngpus):
                    vendor_ccl.allreduce[ngpus=ngpus](
                        in_bufs,
                        out_bufs_vendor[i],
                        rank_sigs,
                        list_of_ctx[i],
                    )

            for i in range(ngpus):
                list_of_ctx[i].synchronize()

            # Verify RCCL results
            for i in range(ngpus):
                list_of_ctx[i].enqueue_copy(host_buffers[i], out_dev_vendor[i])
            for i in range(ngpus):
                for j in range(length):
                    assert_almost_equal(host_buffers[i][j], expected_sum)
        except:
            # Vendor path unavailable or failed; skip silently like vendor_blas fallback
            pass

    # Copy results back and verify
    for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    var mocl_expected_sum = (
        expected_sum if not use_custom_epilogue else -expected_sum
    )
    # Verify results
    for i in range(ngpus):
        for j in range(length):
            try:
                assert_almost_equal(host_buffers[i][j], mocl_expected_sum)
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", mocl_expected_sum)
                raise e^

    # (RCCL verification is performed above within the benchmark block.)

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
    _ = signal_buffers^
    _ = in_bufs_list^
    _ = out_bufs_list^


fn _get_test_str[
    dtype: DType,
    use_multimem: Bool,
    use_custom_epilogue: Bool = False,
](ngpus: Int, length: Int) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var epilogue_tag = "-custom_epilogue" if use_custom_epilogue else ""
    return String(
        "====allreduce-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        epilogue_tag,
        "-",
        human_readable_size(size_of[dtype]() * length),
    )


def allreduce_naive_test() -> None:
    """Explicit smoke test for the allreduce naive path."""
    print("====allreduce-naive-smoke-DType.float32-2-8Ki elements")
    comptime ngpus = 2
    comptime length = 8 * 1024

    # Create contexts for two devices
    var ctxs = List[DeviceContext]()
    for i in range(ngpus):
        ctxs.append(DeviceContext(device_id=i))

    # Allocate input/output buffers and initialize inputs
    var in_dev = List[DeviceBuffer[DType.float32]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[DType.float32]](capacity=ngpus)
    var host_ptrs = List[UnsafePointer[Float32, MutExternalOrigin]](
        capacity=ngpus
    )

    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[DType.float32](length))
        out_dev.append(ctxs[i].enqueue_create_buffer[DType.float32](length))
        var h = alloc[Float32](length)
        host_ptrs.append(h)
        var h_nd = NDBuffer[DType.float32, 1](h, DimList(length))
        h_nd.fill(Float32(i + 1))
        ctxs[i].enqueue_copy(in_dev[i], host_ptrs[i])

    # Wrap as NDBuffers for the kernel API
    var in_bufs = InlineArray[NDBuffer[DType.float32, 1, MutAnyOrigin], ngpus](
        fill={}
    )
    var out_bufs = InlineArray[NDBuffer[DType.float32, 1, MutAnyOrigin], ngpus](
        fill={}
    )

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[DType.float32, 1](
            in_dev[i].unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[DType.float32, 1](
            out_dev[i].unsafe_ptr(), DimList(length)
        )

    # Prepare an output lambda that writes into the correct device's out buffer.
    var out_bufs_capture = StaticTuple[
        NDBuffer[DType.float32, 1, MutAnyOrigin], ngpus
    ](NDBuffer[DType.float32, 1, MutAnyOrigin]())
    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[DType.float32, 1](
            out_dev[i].unsafe_ptr(), DimList(length)
        )

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
            rebind[IndexList[1]](coords),
            rebind[SIMD[DType.float32, _width]](val),
        )

    # Launch naive allreduce per device
    comptime for i in range(ngpus):
        _allreduce_naive_single[
            dtype = DType.float32,
            rank=1,
            ngpus=ngpus,
            output_lambda = outputs_lambda[input_index=i],
        ](in_bufs, out_bufs[i], 216, ctxs[i])

    # Synchronize and verify
    for i in range(ngpus):
        ctxs[i].synchronize()

    var expected = Float32(0)
    for i in range(ngpus):
        expected += Float32(i + 1)
        ctxs[i].enqueue_copy(host_ptrs[i], out_dev[i])

    for i in range(ngpus):
        for j in range(length):
            assert_almost_equal(host_ptrs[i][j], expected)

    for i in range(ngpus):
        host_ptrs[i].free()


@parameter
fn run_allreduce_sweep[use_multimem: Bool]() raises:
    # Run tests for each configuration.
    comptime for gpu_idx, dtype_idx, length_idx, epilogue_idx in product(
        range(len(test_gpu_counts)),
        range(len(test_dtypes)),
        range(len(test_lengths)),
        range(2),  # Test both default and custom epilogue
    ):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() < num_gpus:
            continue

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        comptime dtype = test_dtypes[dtype_idx]
        comptime length = test_lengths[length_idx]
        comptime use_custom_epilogue = epilogue_idx == 1

        # Some checks for raggedness
        comptime simd_width = simd_width_of[dtype, get_gpu_target()]()
        comptime assert (
            length % simd_width == 0
        ), "Length must be multiple of simd_width"

        print(
            _get_test_str[dtype, use_multimem, use_custom_epilogue](
                num_gpus, length
            )
        )
        try:
            allreduce_test[
                dtype=dtype,
                rank=1,
                ngpus=num_gpus,
                use_multimem=use_multimem,
                use_custom_epilogue=use_custom_epilogue,
            ](ctx, length)
        except e:
            if "OUT_OF_MEMORY" in String(e):
                print(
                    "Out of memory error occurred for ",
                    _get_test_str[
                        dtype,
                        use_multimem,
                        use_custom_epilogue,
                    ](num_gpus, length),
                )
            elif (
                use_multimem
                and "multimem is only supported on SM90+ GPUs" in String(e)
            ):
                print(
                    "Skipping multimem test - SM90+ not supported by"
                    " compilation target"
                )
            else:
                raise e^


def main():
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

    # First, explicitly exercise the naive allreduce path by calling it directly.
    allreduce_naive_test()

    # Standard (non-multimem) sweep
    run_allreduce_sweep[use_multimem=False]()
