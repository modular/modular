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
from internal_utils import DeviceNDBuffer, HostNDBuffer, arg_parse
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
    alias largest = test_case.largest
    alias sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    var in_buffer = HostNDBuffer[dtype, rank](DimList(batch_size, N))
    var topk_vals = HostNDBuffer[dtype, rank](DimList(batch_size, K))
    var topk_idxs = HostNDBuffer[out_idx_type, rank](
        DimList(batch_size, out_idx_len)
    )

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer.tensor, fill_fn_name)

    # Move data to device
    var device_in = DeviceNDBuffer[dtype, rank](DimList(batch_size, N), ctx=ctx)
    var device_out_vals = DeviceNDBuffer[dtype, rank](
        DimList(batch_size, K), ctx=ctx
    )
    var device_out_idxs = DeviceNDBuffer[out_idx_type, rank](
        DimList(batch_size, out_idx_len), ctx=ctx
    )

    if not num_blocks_per_input:
        num_blocks_per_input = min(ceildiv(N, block_size), 8)

    var device_local_topk_vals = DeviceNDBuffer[dtype, rank](
        DimList(batch_size, num_blocks_per_input * K), ctx=ctx
    )
    var device_local_topk_idxs = DeviceNDBuffer[out_idx_type, rank](
        DimList(batch_size, num_blocks_per_input * K), ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)

    var K_device_buffer = DeviceNDBuffer[DType.int64, 1](
        DimList(batch_size), ctx=ctx
    )
    var K_host_buffer = HostNDBuffer[DType.int64, 1](DimList(batch_size))
    for i in range(batch_size):
        K_host_buffer.tensor.data[i] = K

    var max_k = Int(reduce_max(K_host_buffer.tensor))

    ctx.enqueue_copy(K_device_buffer.buffer, K_host_buffer.tensor.data)
    ctx.synchronize()

    var k_lt = K_device_buffer.to_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(K_device_buffer)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in.to_layout_tensor(),
                device_local_topk_vals.to_layout_tensor(),
                device_local_topk_idxs.to_layout_tensor(),
                device_out_vals.to_layout_tensor(),
                device_out_idxs.to_layout_tensor(),
                k=OptionalReg(
                    LayoutTensor[
                        K_device_buffer.dtype,
                        Layout.row_major(UNKNOWN_VALUE),
                        MutableAnyOrigin,
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

    var num_bytes = device_in.tensor.size() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name), ThroughputMeasure(BenchMetric.bytes, num_bytes)
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu = HostNDBuffer[dtype, rank](DimList(batch_size, K))
        var topk_idxs_cpu = HostNDBuffer[DType.int64, rank](
            DimList(batch_size, K)
        )

        var k_lt = K_host_buffer.to_layout_tensor()

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer.to_layout_tensor(),
            max_k,
            rank - 1,
            topk_vals_cpu.to_layout_tensor(),
            topk_idxs_cpu.to_layout_tensor(),
            1,
            True,
            k=OptionalReg(
                LayoutTensor[
                    K_device_buffer.dtype,  # or K_host_buffer?
                    Layout.row_major(UNKNOWN_VALUE),
                    MutableAnyOrigin,
                ](
                    k_lt.ptr,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                        k_lt.runtime_layout.shape.value.canonicalize(),
                        k_lt.runtime_layout.stride.value.canonicalize(),
                    ),
                )
            ),
        )

        for i in range(topk_vals.tensor.num_elements()):
            assert_almost_equal(
                topk_vals.tensor.data[i],
                topk_vals_cpu.tensor.data[i],
            )

            @parameter
            if dtype is DType.float32:
                assert_equal(
                    topk_idxs.tensor.data[i],
                    topk_idxs_cpu.tensor.data[i].cast[out_idx_type](),
                )

    _ = topk_vals
    _ = topk_idxs
    _ = in_buffer
    _ = device_in
    _ = device_local_topk_vals
    _ = device_local_topk_idxs
    _ = device_out_vals
    _ = device_out_idxs


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

    alias largest = test_case.largest
    alias sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    var in_buffer = HostNDBuffer[dtype, rank](input_shape)
    var topk_vals = HostNDBuffer[dtype, rank](out_vals_shape)
    var topk_idxs = HostNDBuffer[out_idx_type, rank](out_idxs_shape)

    # Fill the buffer
    fill_buffer[rank, dtype](in_buffer.tensor, fill_fn_name)

    # Move data to device
    var device_in = DeviceNDBuffer[dtype, rank](input_shape, ctx=ctx)
    var device_out_vals = DeviceNDBuffer[dtype, rank](out_vals_shape, ctx=ctx)
    var device_out_idxs = DeviceNDBuffer[out_idx_type, rank](
        out_idxs_shape, ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)
    var batch_size: Int

    @parameter
    if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = Int(input_shape.flattened_length() / last_dim)

    var K_host_buffer = HostNDBuffer[DType.int64, 1](DimList(batch_size))
    for i in range(batch_size):
        K_host_buffer.tensor.data[i] = K

    var K_device_buffer = DeviceNDBuffer[DType.int64, 1](
        DimList(batch_size), ctx=ctx
    )
    ctx.enqueue_copy(K_device_buffer.buffer, K_host_buffer.tensor.data)
    ctx.synchronize()
    var max_k = Int(reduce_max(K_host_buffer.tensor))

    var k_lt = K_device_buffer.to_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(K_device_buffer)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            topk_gpu[sampling=sampling, largest=largest](
                ctx,
                max_k,
                device_in.to_layout_tensor(),
                device_out_vals.to_layout_tensor(),
                device_out_idxs.to_layout_tensor(),
                k=OptionalReg(
                    LayoutTensor[
                        K_device_buffer.dtype,
                        Layout.row_major(UNKNOWN_VALUE),
                        MutableAnyOrigin,
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
    var num_bytes = device_in.tensor.size() * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(kernel_name), ThroughputMeasure(BenchMetric.bytes, num_bytes)
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu = HostNDBuffer[dtype, rank](out_vals_shape)
        var topk_idxs_cpu = HostNDBuffer[DType.int64, rank](out_idxs_shape)
        var k_lt = K_host_buffer.to_layout_tensor()

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_buffer.to_layout_tensor(),
            max_k,
            rank - 1,
            topk_vals_cpu.to_layout_tensor(),
            topk_idxs_cpu.to_layout_tensor(),
            1,
            True,
            k=OptionalReg(
                LayoutTensor[
                    K_host_buffer.dtype,
                    Layout.row_major(UNKNOWN_VALUE),
                    MutableAnyOrigin,
                ](
                    k_lt.ptr,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)](
                        k_lt.runtime_layout.shape.value.canonicalize(),
                        k_lt.runtime_layout.stride.value.canonicalize(),
                    ),
                )
            ),
        )

        for i in range(topk_vals.tensor.num_elements()):
            assert_almost_equal(
                topk_vals.tensor.data[i],
                topk_vals_cpu.tensor.data[i],
            )

            @parameter
            if dtype is DType.float32:
                assert_equal(
                    topk_idxs.tensor.data[i],
                    topk_idxs_cpu.tensor.data[i].cast[out_idx_type](),
                )


fn fill_random[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    alias min_val = -1e9
    alias max_val = 1e9
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
struct TestCase[_sampling: Bool, _largest: Bool = True](
    ImplicitlyCopyable, Movable
):
    alias sampling = _sampling
    alias largest = _largest
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

    alias dtype = env_get_dtype["dtype", DType.float32]()
    alias rank = env_get_int["rank", 2]()
    alias out_idx_type = env_get_dtype["out_idx_type", DType.int]()
    alias sampling = env_get_bool["sampling", False]()
    alias largest = env_get_bool["largest", True]()

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
