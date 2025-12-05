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

from collections import OptionalReg
from math import ceildiv, iota
from random import random_float64

from algorithm.reduction import max as reduce_max
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from internal_utils import arg_parse
from memory import LegacyUnsafePointer as UnsafePointer
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
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

    var in_buffer_ptr = UnsafePointer[Scalar[dtype]].alloc(in_size)
    var topk_vals_ptr = UnsafePointer[Scalar[dtype]].alloc(topk_vals_size)
    var topk_idxs_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        topk_idxs_size
    )

    var in_buffer = NDBuffer[dtype, rank](in_buffer_ptr, DimList(batch_size, N))
    var topk_vals = NDBuffer[dtype, rank](topk_vals_ptr, DimList(batch_size, K))
    var topk_idxs = NDBuffer[out_idx_type, rank](
        topk_idxs_ptr, DimList(batch_size, out_idx_len)
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

    var device_in = NDBuffer[dtype, rank](
        device_in_buffer.unsafe_ptr(), DimList(batch_size, N)
    )
    var device_out_vals = NDBuffer[dtype, rank](
        device_out_vals_buffer.unsafe_ptr(), DimList(batch_size, K)
    )
    var device_out_idxs = NDBuffer[out_idx_type, rank](
        device_out_idxs_buffer.unsafe_ptr(), DimList(batch_size, out_idx_len)
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

    var device_local_topk_vals = NDBuffer[dtype, rank](
        device_local_topk_vals_buffer.unsafe_ptr(),
        DimList(batch_size, num_blocks_per_input * K),
    )
    var device_local_topk_idxs = NDBuffer[out_idx_type, rank](
        device_local_topk_idxs_buffer.unsafe_ptr(),
        DimList(batch_size, num_blocks_per_input * K),
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)

    var K_dev_buffer = ctx.enqueue_create_buffer[DType.int64](batch_size)
    var K_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(batch_size)
    var K_host_buffer = NDBuffer[DType.int64, 1](K_host_ptr, batch_size)
    for i in range(batch_size):
        K_host_ptr[i] = K

    var max_k = Int(reduce_max(K_host_buffer))

    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)
    ctx.synchronize()

    comptime k_layout = Layout.row_major(UNKNOWN_VALUE)
    var k_lt = LayoutTensor[DType.int64, k_layout, MutAnyOrigin](
        K_dev_buffer.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(IndexList[1](batch_size)),
    )

    # Create LayoutTensors for kernel calls
    comptime in_layout = Layout.row_major[rank]()
    var device_in_lt = LayoutTensor[dtype, in_layout, MutAnyOrigin](
        device_in_buffer.unsafe_ptr(),
        RuntimeLayout[in_layout].row_major(IndexList[rank](batch_size, N)),
    )
    var device_local_topk_vals_lt = LayoutTensor[
        dtype, in_layout, MutAnyOrigin
    ](
        device_local_topk_vals_buffer.unsafe_ptr(),
        RuntimeLayout[in_layout].row_major(
            IndexList[rank](batch_size, num_blocks_per_input * K)
        ),
    )
    var device_local_topk_idxs_lt = LayoutTensor[
        out_idx_type, in_layout, MutAnyOrigin
    ](
        device_local_topk_idxs_buffer.unsafe_ptr(),
        RuntimeLayout[in_layout].row_major(
            IndexList[rank](batch_size, num_blocks_per_input * K)
        ),
    )
    var device_out_vals_lt = LayoutTensor[dtype, in_layout, MutAnyOrigin](
        device_out_vals_buffer.unsafe_ptr(),
        RuntimeLayout[in_layout].row_major(IndexList[rank](batch_size, K)),
    )
    var device_out_idxs_lt = LayoutTensor[
        out_idx_type, in_layout, MutAnyOrigin
    ](
        device_out_idxs_buffer.unsafe_ptr(),
        RuntimeLayout[in_layout].row_major(
            IndexList[rank](batch_size, out_idx_len)
        ),
    )

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
                device_in_lt,
                device_local_topk_vals_lt,
                device_local_topk_idxs_lt,
                device_out_vals_lt,
                device_out_idxs_lt,
                k=OptionalReg(
                    LayoutTensor[
                        DType.int64,
                        Layout.row_major(UNKNOWN_VALUE),
                        MutAnyOrigin,
                    ](
                        k_lt.ptr,
                        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                            k_lt.runtime_layout.shape.value.canonicalize(),
                            k_lt.runtime_layout.stride.value.canonicalize(),
                        ),
                    )
                ),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )

        b.iter_custom[kernel_launch](ctx)

    var kernel_name = String(
        "bench-topk", "/N=", N, "/K=", K, "/batch_size=", batch_size
    )

    var num_bytes = device_in.size() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu_ptr = UnsafePointer[Scalar[dtype]].alloc(
            topk_vals_size
        )
        var topk_idxs_cpu_ptr = UnsafePointer[Scalar[DType.int64]].alloc(
            topk_vals_size
        )
        var topk_vals_cpu = NDBuffer[dtype, rank](
            topk_vals_cpu_ptr, DimList(batch_size, K)
        )
        var topk_idxs_cpu = NDBuffer[DType.int64, rank](
            topk_idxs_cpu_ptr, DimList(batch_size, K)
        )

        # Create LayoutTensors for CPU topk
        var k_host_lt = LayoutTensor[DType.int64, k_layout, MutAnyOrigin](
            K_host_ptr,
            RuntimeLayout[k_layout].row_major(IndexList[1](batch_size)),
        )
        var in_buffer_lt = LayoutTensor[dtype, in_layout, MutAnyOrigin](
            in_buffer_ptr,
            RuntimeLayout[in_layout].row_major(IndexList[rank](batch_size, N)),
        )
        var topk_vals_cpu_lt = LayoutTensor[dtype, in_layout, MutAnyOrigin](
            topk_vals_cpu_ptr,
            RuntimeLayout[in_layout].row_major(IndexList[rank](batch_size, K)),
        )
        var topk_idxs_cpu_lt = LayoutTensor[
            DType.int64, in_layout, MutAnyOrigin
        ](
            topk_idxs_cpu_ptr,
            RuntimeLayout[in_layout].row_major(IndexList[rank](batch_size, K)),
        )

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer_lt,
            max_k,
            rank - 1,
            topk_vals_cpu_lt,
            topk_idxs_cpu_lt,
            1,
            True,
            k=OptionalReg(
                LayoutTensor[
                    DType.int64,
                    Layout.row_major(UNKNOWN_VALUE),
                    MutAnyOrigin,
                ](
                    k_host_lt.ptr,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                        k_host_lt.runtime_layout.shape.value.canonicalize(),
                        k_host_lt.runtime_layout.stride.value.canonicalize(),
                    ),
                )
            ),
        )

        for i in range(topk_vals.num_elements()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            @parameter
            if dtype is DType.float32:
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

    var in_buffer_ptr = UnsafePointer[Scalar[dtype]].alloc(in_size)
    var topk_vals_ptr = UnsafePointer[Scalar[dtype]].alloc(out_vals_size)
    var topk_idxs_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(out_idxs_size)

    var in_buffer = NDBuffer[dtype, rank](in_buffer_ptr, input_shape)
    var topk_vals = NDBuffer[dtype, rank](topk_vals_ptr, out_vals_shape)
    var topk_idxs = NDBuffer[out_idx_type, rank](topk_idxs_ptr, out_idxs_shape)

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer, fill_fn_name)

    # Move data to device
    var device_in_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var device_out_vals_buffer = ctx.enqueue_create_buffer[dtype](out_vals_size)
    var device_out_idxs_buffer = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_size
    )

    var device_in = NDBuffer[dtype, rank](
        device_in_buffer.unsafe_ptr(), input_shape
    )
    var device_out_vals = NDBuffer[dtype, rank](
        device_out_vals_buffer.unsafe_ptr(), out_vals_shape
    )
    var device_out_idxs = NDBuffer[out_idx_type, rank](
        device_out_idxs_buffer.unsafe_ptr(), out_idxs_shape
    )

    ctx.enqueue_copy(device_in_buffer, in_buffer_ptr)
    var batch_size: Int

    @parameter
    if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = Int(input_shape.flattened_length() / last_dim)

    var K_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(batch_size)
    var K_host_buffer = NDBuffer[DType.int64, 1](K_host_ptr, batch_size)
    for i in range(batch_size):
        K_host_ptr[i] = K

    var K_dev_buffer = ctx.enqueue_create_buffer[DType.int64](batch_size)
    ctx.enqueue_copy(K_dev_buffer, K_host_ptr)
    ctx.synchronize()
    var max_k = Int(reduce_max(K_host_buffer))

    comptime k_layout = Layout.row_major(UNKNOWN_VALUE)
    var k_lt = LayoutTensor[DType.int64, k_layout, MutAnyOrigin](
        K_dev_buffer.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(IndexList[1](batch_size)),
    )

    # Create LayoutTensors for kernel calls
    comptime mr_layout = Layout.row_major[rank]()
    var device_in_lt = LayoutTensor[dtype, mr_layout, MutAnyOrigin](
        device_in_buffer.unsafe_ptr(),
        RuntimeLayout[mr_layout].row_major(input_shape),
    )
    var device_out_vals_lt = LayoutTensor[dtype, mr_layout, MutAnyOrigin](
        device_out_vals_buffer.unsafe_ptr(),
        RuntimeLayout[mr_layout].row_major(out_vals_shape),
    )
    var device_out_idxs_lt = LayoutTensor[
        out_idx_type, mr_layout, MutAnyOrigin
    ](
        device_out_idxs_buffer.unsafe_ptr(),
        RuntimeLayout[mr_layout].row_major(out_idxs_shape),
    )

    @parameter
    @always_inline
    @__copy_capture(K_dev_buffer)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in_lt,
                device_out_vals_lt,
                device_out_idxs_lt,
                k=OptionalReg(
                    LayoutTensor[
                        DType.int64,
                        Layout.row_major(UNKNOWN_VALUE),
                        MutAnyOrigin,
                    ](
                        k_lt.ptr,
                        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                            k_lt.runtime_layout.shape.value.canonicalize(),
                            k_lt.runtime_layout.stride.value.canonicalize(),
                        ),
                    )
                ),
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )

        b.iter_custom[kernel_launch](ctx)

    var kernel_name = "topk-multirank"
    var num_bytes = device_in.size() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name), [ThroughputMeasure(BenchMetric.bytes, num_bytes)]
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_ptr, device_out_vals_buffer)
    ctx.enqueue_copy(topk_idxs_ptr, device_out_idxs_buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu_ptr = UnsafePointer[Scalar[dtype]].alloc(
            out_vals_size
        )
        var topk_idxs_cpu_ptr = UnsafePointer[Scalar[DType.int64]].alloc(
            out_vals_size
        )
        var topk_vals_cpu = NDBuffer[dtype, rank](
            topk_vals_cpu_ptr, out_vals_shape
        )
        var topk_idxs_cpu = NDBuffer[DType.int64, rank](
            topk_idxs_cpu_ptr, out_idxs_shape
        )

        var k_host_lt = LayoutTensor[DType.int64, k_layout, MutAnyOrigin](
            K_host_ptr,
            RuntimeLayout[k_layout].row_major(IndexList[1](batch_size)),
        )
        var in_buffer_lt = LayoutTensor[dtype, mr_layout, MutAnyOrigin](
            in_buffer_ptr,
            RuntimeLayout[mr_layout].row_major(input_shape),
        )
        var topk_vals_cpu_lt = LayoutTensor[dtype, mr_layout, MutAnyOrigin](
            topk_vals_cpu_ptr,
            RuntimeLayout[mr_layout].row_major(out_vals_shape),
        )
        var topk_idxs_cpu_lt = LayoutTensor[
            DType.int64, mr_layout, MutAnyOrigin
        ](
            topk_idxs_cpu_ptr,
            RuntimeLayout[mr_layout].row_major(out_idxs_shape),
        )

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer_lt,
            max_k,
            rank - 1,
            topk_vals_cpu_lt,
            topk_idxs_cpu_lt,
            1,
            True,
            k=OptionalReg(
                LayoutTensor[
                    DType.int64,
                    Layout.row_major(UNKNOWN_VALUE),
                    MutAnyOrigin,
                ](
                    k_host_lt.ptr,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                        k_host_lt.runtime_layout.shape.value.canonicalize(),
                        k_host_lt.runtime_layout.stride.value.canonicalize(),
                    ),
                )
            ),
        )

        for i in range(topk_vals.num_elements()):
            assert_almost_equal(
                topk_vals_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            @parameter
            if dtype is DType.float32:
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
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


fn fill_constant[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        if i % 3 == 1:
            buffer.data[i] = 1.0
        else:
            buffer.data[i] = 0.0


fn fill_iota[rank: Int, dtype: DType](mut buf: NDBuffer[mut=True, dtype, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


fn fill_buffer[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank], mode: String) raises:
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
