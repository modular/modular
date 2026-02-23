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

from collections import InlineArray
from math import ceildiv
from sys import env_get_bool, env_get_dtype, env_get_int, size_of, simd_width_of

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from comm.sync import can_enable_p2p
from comm.broadcast import broadcast
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
)
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from testing import assert_true


@always_inline
@parameter
fn _input_value[dtype: DType](root: Int, j: Int) -> Scalar[dtype]:
    """Generate position-based input value that includes root rank.

    Each element has a unique value based on position, and includes the root
    rank to verify the correct source GPU was used.
    """
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](root + 1) + Scalar[dtype](j % 251))


fn _get_test_str[
    dtype: DType, use_multimem: Bool, use_vendor_ccl: Bool, cache_busting: Bool
](ngpus: Int, num_bytes: Int, root: Int) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var vendorccl_tag = "-vendorccl" if use_vendor_ccl else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    return String(
        "broadcast-",
        dtype,
        "-",
        ngpus,
        "gpus-root",
        root,
        multimem_tag,
        vendorccl_tag,
        cache_tag,
        "-",
        human_readable_size(num_bytes),
    )


fn bench_broadcast[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    cache_busting: Bool,
    use_vendor_ccl: Bool = False,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    root: Int,
    max_num_blocks: Optional[Int],
) raises:
    comptime assert ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"
    comptime assert rank == 1, "this test code currently assumes rank 1"

    var name = String(
        _get_test_str[dtype, use_multimem, use_vendor_ccl, cache_busting](
            ngpus, num_bytes, root
        )
    )

    var length = num_bytes // size_of[dtype]()

    comptime simd_size = simd_width_of[dtype, target = get_gpu_target()]()
    var cb_in = CacheBustingBuffer[dtype](
        length, simd_size, list_of_ctx[root], cache_busting
    )

    # Create output device buffers for all GPUs
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)

    # Create signal buffers for synchronization AND payload space
    # Two-stage broadcast needs payload space for each GPU's chunk
    var chunk_bytes = ceildiv(num_bytes, ngpus)
    var signal_buf_size = size_of[Signal]() + chunk_bytes
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    # Multicast buffer for output (when use_multimem=True)
    var out_multicast_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    # Initialize output and signal buffers for each GPU
    comptime if use_multimem:
        out_multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), length
        )
        out_multicast_ptr = out_multicast_buf.multicast_buffer_for(
            list_of_ctx[0]
        ).unsafe_ptr()

        comptime for gpu_idx in range(ngpus):
            # For multimem, we use unicast buffers for verification/copy-back
            out_bufs_list.append(
                out_multicast_buf.unicast_buffer_for(list_of_ctx[gpu_idx])
            )

            # Create and initialize signal buffers (with payload space for 2-stage)
            signal_buffers.append(
                list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                    signal_buf_size
                )
            )
            list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
                signal_buffers[gpu_idx], 0
            )
            rank_sigs[gpu_idx] = (
                signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
            )
    else:
        comptime for gpu_idx in range(ngpus):
            # Create output buffer for this GPU
            out_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
            )

            # Create and initialize signal buffers (with payload space for 2-stage)
            signal_buffers.append(
                list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                    signal_buf_size
                )
            )
            list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
                signal_buffers[gpu_idx], 0
            )
            rank_sigs[gpu_idx] = (
                signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
            )

    # Create and initialize host buffer for root with position-based values
    var host_buffer = alloc[Scalar[dtype]](cb_in.alloc_size())
    for i in range(cb_in.alloc_size() // cb_in.stride):
        for j in range(length):
            host_buffer[i * cb_in.stride + j] = _input_value[dtype](root, j)

    # Copy host data to input buffer on root GPU
    list_of_ctx[root].enqueue_copy(cb_in.device_buffer(), host_buffer)

    # Create NDBuffer wrappers for outputs
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )

    comptime if use_multimem:
        # All GPUs use the same multicast pointer for output
        for i in range(ngpus):
            out_bufs[i] = NDBuffer[dtype, rank](
                out_multicast_ptr, DimList(length)
            )
            list_of_ctx[i].synchronize()
    else:
        for i in range(ngpus):
            out_bufs[i] = NDBuffer[dtype, rank](
                out_bufs_list[i].unsafe_ptr(), DimList(length)
            )
            # Ensure setup has propagated.
            list_of_ctx[i].synchronize()

    # Zero device output buffers once before benchmarking so verification isn't
    # affected by any stale data in case a kernel path doesn't overwrite fully.
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_memset(out_bufs_list[i], 0)

    # Pre-initialize vendor CCL communicators from the main thread.
    # ncclCommInitAll is not thread-safe, so we must initialize before
    # spawning worker threads.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_broadcast_available():
            raise "Vendor CCL not available; skipping vendor path."
        vendor_ccl.init_comms(ngpus)

    @parameter
    @always_inline
    fn bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            var in_buf_offset = NDBuffer[dtype, rank, MutAnyOrigin](
                cb_in.offset_ptr(cache_iter),
                DimList(length),
            )

            # Run broadcast - root's input goes to all outputs
            comptime broadcast_kernel = vendor_ccl.broadcast if use_vendor_ccl else broadcast
            broadcast_kernel[ngpus, use_multimem=use_multimem](
                in_buf_offset,
                out_bufs[ctx_idx],
                rank_sigs,
                ctx_inner,
                root,
                max_num_blocks,
            )

        bencher.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_iter](
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    var max_time = b.info_vec[0].result.mean(unit="ms")
    var gbps = Float64(num_bytes) / (max_time * 1000 * 1000)
    # For broadcast, busbw = algbw (factor of 1).
    # All data must leave the root, which is the bottleneck.
    # See: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    var busbw = gbps
    print(
        "|",
        name,
        "| slowest mean time",
        max_time,
        "ms |",
        "algbw:",
        gbps,
        "GB/s |",
        "busbw:",
        busbw,
        "GB/s |",
    )

    # Zero output and signal buffers, then run one more broadcast for verification.
    # This ensures we're verifying fresh results, not stale data from
    # a previous iteration that might mask a broken kernel.
    # Signal buffers must also be zeroed since 2-stage uses the payload as scratch.
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_memset(signal_buffers[i], 0)
        list_of_ctx[i].enqueue_memset(out_bufs_list[i], 0)
        list_of_ctx[i].synchronize()

    # Create input buffer for verification (no cache offset)
    var in_buf_verify = NDBuffer[dtype, rank, MutAnyOrigin](
        cb_in.unsafe_ptr(),
        DimList(length),
    )

    # Run one broadcast for verification
    comptime for i in range(ngpus):
        comptime broadcast_kernel = vendor_ccl.broadcast if use_vendor_ccl else broadcast
        broadcast_kernel[ngpus, use_multimem=use_multimem](
            in_buf_verify,
            out_bufs[i],
            rank_sigs,
            list_of_ctx[i],
            root,
            max_num_blocks,
        )

    # Copy results back and verify - reuse host_buffer for each GPU
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffer, out_bufs_list[i])
        list_of_ctx[i].synchronize()

        # Verify results - all GPUs should have root's data
        for j in range(length):
            var expected = _input_value[dtype](root, j)
            if host_buffer[j] != expected:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffer[j])
                print("Expected:", expected)
                raise Error("Verification failed")

    # Cleanup
    host_buffer.free()
    _ = signal_buffers^
    _ = cb_in^


def main():
    var num_bytes = arg_parse("num_bytes", 64 * 1024 * 1024)
    var root = arg_parse("root", 0)

    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = env_get_int["num_gpus", 2]()
    comptime rank = env_get_int["rank", 1]()
    comptime use_multimem = env_get_bool["use_multimem", False]()
    comptime use_vendor_ccl = env_get_bool["use_vendor_ccl", False]()
    comptime cache_busting = True

    # Allow overriding max_num_blocks from command line for tuning.
    var max_nb = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)
    assert_true(root >= 0 and root < num_gpus, "root must be in [0, num_gpus)")

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not can_enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    bench_broadcast[
        dtype=dtype,
        rank=rank,
        ngpus=num_gpus,
        use_multimem=use_multimem,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes, root, max_num_blocks)
