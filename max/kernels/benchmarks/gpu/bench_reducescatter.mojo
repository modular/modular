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
from math import align_up
from sys import env_get_bool, env_get_dtype, env_get_int, size_of, simd_width_of
from utils.numerics import get_accum_type

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
from comm.reducescatter import reducescatter, ReduceScatterConfig
from comm import MAX_GPUS, Signal
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import (
    arg_parse,
    pytorch_like_tolerances_for,
    human_readable_size,
)

from testing import assert_almost_equal, assert_true


@always_inline
@parameter
fn _per_gpu_value[
    dtype: DType,
](gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


fn _get_test_str[
    dtype: DType,
    use_multimem: Bool,
    cache_busting: Bool,
](ngpus: Int, num_bytes: Int, ragged: Bool) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    var ragged_tag = "-ragged" if ragged else ""
    return String(
        "reducescatter-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        cache_tag,
        ragged_tag,
        "-",
        human_readable_size(num_bytes),
    )


fn bench_reducescatter[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    cache_busting: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
    ragged: Bool,
) raises:
    constrained[ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"]()
    constrained[rank == 1, "this test code currently assumes rank 1"]()

    var name = String(
        _get_test_str[dtype, use_multimem, cache_busting](
            ngpus, num_bytes, ragged
        )
    )
    print("Running " + name)

    # Total input size per GPU
    var input_length = num_bytes // size_of[dtype]()

    # Use ReduceScatterConfig to compute per-GPU partition info (w/ dummy nthreads)
    var rs_config = ReduceScatterConfig[dtype, ngpus](input_length, 0)
    var output_lengths = List[Int](capacity=ngpus)
    var rank_starts = List[Int](capacity=ngpus)
    for gpu_idx in range(ngpus):
        rank_starts.append(rs_config.rank_start(gpu_idx))
        output_lengths.append(rs_config.rank_part(gpu_idx))

    comptime num_buffers = 1 if use_multimem else ngpus

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

    # Cache busting: allocate larger buffer to avoid cache reuse
    var stride = align_up(input_length, rs_config.simd_width)
    comptime m512 = 512 * 1024 * 1024
    var cache_elems = (
        align_up(m512, stride * size_of[dtype]()) // size_of[dtype]()
    )

    # Initialize buffers for each GPU
    for gpu_idx in range(ngpus):
        # Create input and output device buffers
        in_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](cache_elems)
        )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                output_lengths[gpu_idx]
            )
        )

        # Create and initialize host buffers
        var host_buffer = alloc[Scalar[dtype]](cache_elems)
        host_buffers.append(host_buffer)

        # Fill with repeated GPU-specific values for cache busting
        for i in range(cache_elems // stride):
            for j in range(input_length):
                host_buffer[i * stride + j] = _per_gpu_value[dtype](gpu_idx, j)

        # Copy to device
        list_of_ctx[gpu_idx].enqueue_copy(in_bufs_list[gpu_idx], host_buffer)

        # Create and initialize signal buffers
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

    # Create input and output NDBuffers
    var in_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], num_buffers](
        fill={}
    )
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )

    for i in range(ngpus):
        in_bufs[i if not use_multimem else 0] = NDBuffer[dtype, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(input_length)
        )
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(output_lengths[i])
        )
        list_of_ctx[i].synchronize()

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher, ctx: DeviceContext, ctx_idx: Int) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            # Offset the input buffer if cache_busting
            var offset = 0

            comptime if cache_busting:
                offset = (cache_iter * stride) % cache_elems

            comptime for i in range(ngpus):
                in_bufs[i] = NDBuffer[dtype, rank](
                    in_bufs_list[i].unsafe_ptr() + offset,
                    DimList(input_length),
                )

            reducescatter[ngpus=ngpus, use_multimem=use_multimem](
                in_bufs,
                out_bufs[ctx_idx],
                rank_sigs,
                ctx_inner,
                max_num_blocks,
            )

        b.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_iter](
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    # Copy results back and verify
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results
    # For low-precision dtypes (e.g., bfloat16), inputs were quantized to `dtype`
    # before reduction on device. Mirror the device path here by:
    #  - quantizing each per-GPU term to `dtype` by calling _per_gpu_value[dtype](...)
    #  - accumulating in Float32
    #  - finally casting to `dtype` for the expected value
    comptime for i in range(ngpus):
        for j in range(output_lengths[i]):
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)
            var global_idx = rank_starts[i] + j

            comptime for k in range(ngpus):
                var term_dtype = _per_gpu_value[dtype](k, global_idx)
                accum += Scalar[accum_t](term_dtype)
            var expected_sum = Scalar[dtype](accum)
            try:
                var rtol, atol = pytorch_like_tolerances_for[dtype]()
                assert_almost_equal(
                    host_buffers[i][j], expected_sum, atol=atol, rtol=rtol
                )
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e^

    # Clean up
    for i in range(ngpus):
        host_buffers[i].free()


def main():
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = env_get_int["num_gpus", 2]()
    comptime rank = env_get_int["rank", 1]()
    comptime ragged = env_get_bool["ragged", False]()

    var max_nb = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)

    comptime use_multimem = env_get_bool["multimem", False]()
    comptime cache_busting = True

    # When ragged, add (ngpus/2) * simd_width elements to create uneven partitions
    comptime simd_size = simd_width_of[dtype, target = get_gpu_target()]()

    comptime if ragged:
        num_bytes += (num_gpus // 2) * simd_size * size_of[dtype]()

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)

    # Create GPU contexts
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not can_enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    bench_reducescatter[
        dtype=dtype,
        rank=rank,
        ngpus=num_gpus,
        use_multimem=use_multimem,
        cache_busting=cache_busting,
    ](m, ctx, num_bytes, max_num_blocks, ragged)
