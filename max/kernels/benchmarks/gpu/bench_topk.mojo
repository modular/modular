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

from math import ceildiv, iota
from random import random_float64

from algorithm.reduction import max as reduce_max
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from internal_utils import arg_parse

from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor

from nn.topk import _top_k_cpu, _topk_gpu, topk_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList
from sys import env_get_int, env_get_bool, env_get_dtype, env_get_string
from sys.info import size_of


fn bench_topk_batched[
    dtype: DType, out_idx_type: DType, rank: Int
](
    ctx: DeviceContext, mut m: Bench, test_case: TestCase, fill_fn_name: String
) raises:
    # Fetch arguments

    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    var in_size = batch_size * N
    var topk_vals_size = batch_size * K
    var topk_idxs_size = batch_size * out_idx_len

    var in_buffer_ptr = alloc[Scalar[dtype]](in_size)
    var topk_vals_ptr = alloc[Scalar[dtype]](topk_vals_size)
    var topk_idxs_ptr = alloc[Scalar[out_idx_type]](topk_idxs_size)

    var in_buffer = TileTensor(
        in_buffer_ptr,
        row_major((Idx(batch_size), Idx(N))),
    )
    var topk_vals = TileTensor(
        topk_vals_ptr,
        row_major((Idx(batch_size), Idx(K))),
    )
    var topk_idxs = TileTensor(
        topk_idxs_ptr, row_major((Idx(batch_size), Idx(out_idx_len)))
    )

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer, fill_fn_name)

    # Move data to device
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_vals_buffer = ctx.enqueue_create_buffer[dtype](
        topk_vals_size
    )
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        topk_idxs_size
    )

    var device_in = TileTensor(
        device_in_buffer.unsafe_ptr(), row_major((Idx(batch_size), Idx(N)))
    )
    var device_out_vals = TileTensor(
        device_out_vals_buffer.unsafe_ptr(),
        row_major((Idx(batch_size), Idx(K))),
    )
    var device_out_idxs = TileTensor(
        device_out_idxs_buffer, row_major((Idx(batch_size), Idx(out_idx_len)))
    )

    if not num_blocks_per_input:
        num_blocks_per_input = min(ceildiv(N, block_size), 8)

    var local_topk_size = batch_size * num_blocks_per_input * K
    var device_local_topk_vals_buffer = ctx.enqueue_create_buffer[dtype](
        local_topk_size
    )
    var device_local_topk_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        local_topk_size
    )

    var device_local_topk_vals = TileTensor(
        device_local_topk_vals_buffer.unsafe_ptr(),
        row_major((Idx(batch_size), Idx(num_blocks_per_input * K))),
    )
    var device_local_topk_idxs = TileTensor(
        device_local_topk_idxs_buffer.unsafe_ptr(),
        row_major((Idx(batch_size), Idx(num_blocks_per_input * K))),
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)

    var K_dev_buffer = ctx.enqueue_create_buffer[DType.int64](batch_size)
    var k = TileTensor(K_dev_buffer, row_major(Idx(batch_size)))
    var K_host_ptr = alloc[Int64](batch_size)
    var K_host_buffer = TileTensor(K_host_ptr, row_major(Idx(batch_size)))
    for i in range(batch_size):
        K_host_ptr[i] = Int64(K)

    var max_k = Int(
        reduce_max(Span(ptr=K_host_buffer.ptr, length=K_host_buffer.numel()))
    )

    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)
    ctx.synchronize()

    @parameter
    @always_inline
    @__copy_capture(K_dev_buffer)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in,
                device_local_topk_vals,
                device_local_topk_idxs,
                device_out_vals,
                device_out_idxs,
                k=TileTensor(k.ptr, row_major(Idx(Int64(batch_size))))
                .as_any_origin()
                .as_immut(),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )

        b.iter_custom[kernel_launch](ctx)

    var kernel_name = String(
        "bench-topk", "/N=", N, "/K=", K, "/batch_size=", batch_size
    )

    var num_bytes = device_in.numel() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = alloc[Scalar[dtype]](topk_vals_size)
        var topk_idxs_cpu_ptr = alloc[Int64](topk_vals_size)
        var topk_vals_cpu = TileTensor(
            topk_vals_cpu_ptr, row_major((Idx(batch_size), Idx(K)))
        )
        var topk_idxs_cpu = TileTensor(
            topk_idxs_cpu_ptr, row_major((Idx(batch_size), Idx(K)))
        )

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer,
            max_k,
            rank - 1,
            topk_vals_cpu,
            topk_idxs_cpu,
            1,
            True,
            k=K_host_buffer.as_any_origin().as_immut(),
        )

        for i in range(topk_vals.numel()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == DType.float32:
                assert_equal(
                    topk_idxs_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

        topk_vals_cpu_ptr.free()
        topk_idxs_cpu_ptr.free()

    # Cleanup host pointers
    in_buffer_ptr.free()
    topk_vals_ptr.free()
    topk_idxs_ptr.free()
    K_host_ptr.free()

    # Consume device buffers
    _ = device_in_buffer^
    _ = device_out_vals_buffer^
    _ = device_out_idxs_buffer^
    _ = device_local_topk_vals_buffer^
    _ = device_local_topk_idxs_buffer^
    _ = K_dev_buffer^


fn bench_topk_multi_rank[
    dtype: DType,
    rank: Int,
    out_idx_type: DType = DType.int,
](
    ctx: DeviceContext,
    mut m: Bench,
    input_shape: IndexList[rank],
    test_case: TestCase,
    fill_fn_name: String,
) raises:
    # Fetch arguments
    # var input_shape = test_case.input_shape
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input: Int = min(
        ceildiv(input_shape.flattened_length(), block_size), 8
    ) if not test_case.num_blocks_per_input else test_case.num_blocks_per_input

    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    var in_size = input_shape.flattened_length()
    var out_vals_size = out_vals_shape.flattened_length()
    var out_idxs_size = out_idxs_shape.flattened_length()

    var in_buffer_ptr = alloc[Scalar[dtype]](in_size)
    var topk_vals_ptr = alloc[Scalar[dtype]](out_vals_size)
    var topk_idxs_ptr = alloc[Scalar[out_idx_type]](out_idxs_size)

    var in_buffer = TileTensor(in_buffer_ptr, row_major(Coord(input_shape)))
    var topk_vals = TileTensor(topk_vals_ptr, row_major(Coord(out_vals_shape)))
    var topk_idxs = TileTensor(topk_idxs_ptr, row_major(Coord(out_idxs_shape)))

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer, fill_fn_name)

    # Move data to device
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_vals_buffer = ctx.enqueue_create_buffer[dtype](out_vals_size)
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_size
    )

    var device_in = TileTensor(
        device_in_buffer.unsafe_ptr(), row_major(Coord(input_shape))
    )
    var device_out_vals = TileTensor(
        device_out_vals_buffer.unsafe_ptr(), row_major(Coord(out_vals_shape))
    )
    var device_out_idxs = TileTensor(
        device_out_idxs_buffer.unsafe_ptr(), row_major(Coord(out_idxs_shape))
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)
    var batch_size: Int

    comptime if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = input_shape.flattened_length() // last_dim

    var K_host_ptr = alloc[Int64](batch_size)
    var K_host_buffer = TileTensor(K_host_ptr, row_major(Idx(batch_size)))
    for i in range(batch_size):
        K_host_ptr[i] = Int64(K)

    var K_dev_buffer = ctx.enqueue_create_buffer[DType.int64](batch_size)
    var k = TileTensor(K_dev_buffer, row_major(Idx(batch_size)))
    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)
    ctx.synchronize()
    var max_k = Int(
        reduce_max(Span(ptr=K_host_buffer.ptr, length=K_host_buffer.numel()))
    )

    @parameter
    @always_inline
    @__copy_capture(k)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in,
                device_out_vals,
                device_out_idxs,
                k=TileTensor(k.ptr, row_major(Idx(Int64(batch_size))))
                .as_any_origin()
                .as_immut(),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )

        b.iter_custom[kernel_launch](ctx)

    var kernel_name = "topk-multirank"
    var num_bytes = device_in.numel() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name), [ThroughputMeasure(BenchMetric.bytes, num_bytes)]
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    comptime if not sampling:
        var topk_vals_cpu_ptr = alloc[Scalar[dtype]](out_vals_size)
        var topk_idxs_cpu_ptr = alloc[Int64](out_vals_size)
        var topk_vals_cpu = TileTensor(
            topk_vals_cpu_ptr, row_major(Coord(out_vals_shape))
        )
        var topk_idxs_cpu = TileTensor(
            topk_idxs_cpu_ptr, row_major(Coord(out_idxs_shape))
        )

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer,
            max_k,
            rank - 1,
            topk_vals_cpu,
            topk_idxs_cpu,
            1,
            True,
            k=K_host_buffer.as_any_origin().as_immut(),
        )

        for i in range(topk_vals.numel()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            comptime if dtype == DType.float32:
                assert_equal(
                    topk_idxs_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

        topk_vals_cpu_ptr.free()
        topk_idxs_cpu_ptr.free()

    # Cleanup host pointers
    in_buffer_ptr.free()
    topk_vals_ptr.free()
    topk_idxs_ptr.free()
    K_host_ptr.free()

    # Consume device buffers
    _ = device_in_buffer^
    _ = device_out_vals_buffer^
    _ = device_out_idxs_buffer^
    _ = K_dev_buffer^


fn fill_random[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.numel()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.ptr[i] = random_value.cast[dtype]()


fn fill_constant[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...]):
    var total_elements = buffer.numel()
    for i in range(total_elements):
        if i % 3 == 1:
            buffer.ptr[i] = 1.0
        else:
            buffer.ptr[i] = 0.0


fn fill_iota[
    rank: Int, dtype: DType
](mut buf: TileTensor[mut=True, dtype, ...]):
    iota(
        buf.ptr,
        coord_to_index_list(buf.layout.shape_coord()).flattened_length(),
    )


fn fill_buffer[
    rank: Int, dtype: DType
](mut buffer: TileTensor[mut=True, dtype, ...], mode: String) raises:
    if mode == "fill_constant":
        fill_constant[rank, dtype](buffer)
    elif mode == "fill_random":
        fill_random[rank, dtype](buffer)
    elif mode == "fill_iota":
        fill_iota[rank, dtype](buffer)
    else:
        raise Error("fill mode not found")


@fieldwise_init
struct TestCase[_sampling: Bool, _largest: Bool = True](ImplicitlyCopyable):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var N: Int
    var K: Int
    var block_size: Int
    var batch_size: Int
    var num_blocks_per_input: Int


fn main() raises:
    var N = arg_parse("N", -1)
    var K = arg_parse("K", -1)
    var block_size = arg_parse("block_size", 256)
    var batch_size = arg_parse("batch_size", -1)
    var num_blocks_per_input = arg_parse("num_blocks_per_input", 0)
    var fill_fn_name = arg_parse("fill_fn_name", "fill_random")

    comptime dtype = env_get_dtype["dtype", DType.float32]()
    comptime rank = env_get_int["rank", 2]()
    comptime out_idx_type = env_get_dtype["out_idx_type", DType.int]()
    comptime sampling = env_get_bool["sampling", False]()
    comptime largest = env_get_bool["largest", True]()

    var m = Bench()
    with DeviceContext() as ctx:
        var test_case = TestCase[_sampling=sampling, _largest=largest](
            N=N,
            K=K,
            block_size=block_size,
            batch_size=batch_size,
            num_blocks_per_input=num_blocks_per_input,
        )
        bench_topk_batched[dtype, out_idx_type, rank](
            ctx, m, test_case, fill_fn_name
        )

        # TODO: enable the following in another benchmark.
        # bench_topk_multi_rank[dtype, rank, out_idx_type](ctx, m, IndexList[rank](1, 1024), test_case, fill_fn_name)

    m.dump_report()
