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
from random import random_float64, seed

from algorithm.reduction import max as reduce_max
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from memory import LegacyUnsafePointer as UnsafePointer
from nn.topk import _top_k_cpu, _topk_gpu, topk_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](mut m: Bench, ctx: DeviceContext, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(mut m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            func(ctx)

        m.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            kernel_name
        ),  # ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


fn test_case_batched[
    dtype: DType,
    fill_fn: fn[dtype: DType] (LayoutTensor[mut=True, dtype, **_]) capturing [
        _
    ] -> None,
    out_idx_type: DType = DType.int,
    rank: Int = 2,
](ctx: DeviceContext, test_case: TestCase) raises:
    # Fetch arguments
    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    # Allocate host memory
    var in_shape = IndexList[2](batch_size, N)
    var out_vals_shape = IndexList[2](batch_size, K)
    var out_idxs_shape = IndexList[2](batch_size, out_idx_len)

    var in_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        in_shape.flattened_length()
    )
    var topk_vals_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        out_vals_shape.flattened_length()
    )
    var topk_idxs_host_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        out_idxs_shape.flattened_length()
    )

    # Create NDBuffer for fill_fn (required by function signature)
    comptime in_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var in_tensor = LayoutTensor[dtype, in_layout](
        in_host_ptr, RuntimeLayout[in_layout].row_major(in_shape)
    )

    # Fill the buffer with consecutive values
    fill_fn(in_tensor)

    # Create device buffers
    var device_in = ctx.enqueue_create_buffer[dtype](
        in_shape.flattened_length()
    )
    var device_out_vals = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var device_out_idxs = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    var num_blocks_per_input_: Int = ceildiv(
        N, block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    var local_topk_shape = IndexList[2](batch_size, num_blocks_per_input_ * K)
    var device_local_topk_vals = ctx.enqueue_create_buffer[dtype](
        local_topk_shape.flattened_length()
    )
    var device_local_topk_idxs = ctx.enqueue_create_buffer[out_idx_type](
        local_topk_shape.flattened_length()
    )

    ctx.enqueue_copy(device_in, in_host_ptr)

    # Create K buffers
    var K_shape = IndexList[1](batch_size)
    var K_device_buffer = ctx.enqueue_create_buffer[DType.int64](
        K_shape.flattened_length()
    )
    var K_host_ptr = UnsafePointer[Int64].alloc(K_shape.flattened_length())
    # NDBuffer needed for reduce_max
    var K_host_tensor = NDBuffer[DType.int64, 1](K_host_ptr, K_shape)
    for i in range(batch_size):
        K_host_ptr[i] = K

    var max_k = Int(reduce_max(K_host_tensor))

    ctx.enqueue_copy(K_device_buffer, K_host_ptr)
    ctx.synchronize()

    # Create layout tensors for kernel calls
    comptime out_vals_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime out_idxs_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime local_topk_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE)

    var in_runtime_layout = RuntimeLayout[in_layout].row_major(
        IndexList[2](batch_size, N)
    )
    var out_vals_runtime_layout = RuntimeLayout[out_vals_layout].row_major(
        IndexList[2](batch_size, K)
    )
    var out_idxs_runtime_layout = RuntimeLayout[out_idxs_layout].row_major(
        IndexList[2](batch_size, out_idx_len)
    )
    var local_topk_runtime_layout = RuntimeLayout[local_topk_layout].row_major(
        IndexList[2](batch_size, num_blocks_per_input_ * K)
    )
    var k_runtime_layout = RuntimeLayout[k_layout].row_major(
        IndexList[1](batch_size)
    )

    var device_in_lt = LayoutTensor[dtype, in_layout](
        device_in, in_runtime_layout
    )
    var device_out_vals_lt = LayoutTensor[dtype, out_vals_layout](
        device_out_vals, out_vals_runtime_layout
    )
    var device_out_idxs_lt = LayoutTensor[out_idx_type, out_idxs_layout](
        device_out_idxs, out_idxs_runtime_layout
    )
    var device_local_topk_vals_lt = LayoutTensor[dtype, local_topk_layout](
        device_local_topk_vals, local_topk_runtime_layout
    )
    var device_local_topk_idxs_lt = LayoutTensor[
        out_idx_type, local_topk_layout
    ](device_local_topk_idxs, local_topk_runtime_layout)
    var k_lt = LayoutTensor[DType.int64, k_layout](
        K_device_buffer, k_runtime_layout
    )

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
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
            ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
            ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
            ctx.synchronize()

        comptime msg = "tk-smpl-gpu" if sampling else "tk-gpu"
        time_kernel[run_func](m, ctx, msg)

    _topk_gpu[sampling=sampling, largest=largest](
        ctx,
        max_k,  # max_k
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

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
    ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
    ctx.synchronize()

    @parameter
    if PRINT_OUTPUT:
        var _msg1: String = "Top-K values"
        var _msg2 = "Sample token index" if sampling else String(
            "Top K indices"
        )
        print(_msg1, "and", _msg2, "output available in host pointers")

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu_ptr = UnsafePointer[Scalar[dtype]].alloc(
            out_vals_shape.flattened_length()
        )
        var topk_idxs_cpu_ptr = UnsafePointer[Int64].alloc(
            out_vals_shape.flattened_length()
        )

        # Create layout tensors for CPU reference
        var in_host_lt = LayoutTensor[dtype, in_layout](
            in_host_ptr, in_runtime_layout
        )
        var topk_vals_cpu_lt = LayoutTensor[dtype, out_vals_layout](
            topk_vals_cpu_ptr, out_vals_runtime_layout
        )
        var topk_idxs_cpu_lt = LayoutTensor[DType.int64, out_vals_layout](
            topk_idxs_cpu_ptr, out_vals_runtime_layout
        )
        var k_host_lt = LayoutTensor[DType.int64, k_layout](
            K_host_ptr, k_runtime_layout
        )

        @parameter
        if DEBUG_BENCH:

            @always_inline
            @parameter
            fn run_func_cpu(ctx: DeviceContext) raises:
                _top_k_cpu[
                    dtype=dtype,
                    out_idx_type = DType.int64,
                    largest=largest,
                ](
                    in_host_lt,
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

            time_kernel[run_func_cpu](m, ctx, "topk-cpu")

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_host_lt,
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

        for i in range(out_vals_shape.flattened_length()):
            assert_almost_equal(
                topk_vals_host_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            @parameter
            if dtype is DType.float32:
                assert_equal(
                    topk_idxs_host_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

        # Free CPU reference buffers
        topk_vals_cpu_ptr.free()
        topk_idxs_cpu_ptr.free()

    # Free host pointers
    in_host_ptr.free()
    topk_vals_host_ptr.free()
    topk_idxs_host_ptr.free()
    K_host_ptr.free()

    # Free device buffers
    _ = device_in^
    _ = device_out_vals^
    _ = device_out_idxs^
    _ = device_local_topk_vals^
    _ = device_local_topk_idxs^
    _ = K_device_buffer^

    @parameter
    if DEBUG_BENCH:
        m.dump_report()


fn test_case_multi_rank[
    dtype: DType,
    fill_fn: fn[dtype: DType] (LayoutTensor[mut=True, dtype, **_]) capturing [
        _
    ] -> None,
    rank: Int,
    out_idx_type: DType = DType.int,
](ctx: DeviceContext, test_case: TestCaseMultiRank[rank=rank, *_]) raises:
    # Fetch arguments
    var input_shape = test_case.input_shape
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    # Allocate host memory
    var in_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        input_shape.flattened_length()
    )
    var topk_vals_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        out_vals_shape.flattened_length()
    )
    var topk_idxs_host_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        out_idxs_shape.flattened_length()
    )

    # Create NDBuffer for fill_fn (required by function signature)
    comptime in_layout = Layout.row_major[rank]()
    var in_tensor = LayoutTensor[dtype, in_layout](
        in_host_ptr, RuntimeLayout[in_layout].row_major(input_shape)
    )

    # Fill the buffer with consecutive values
    fill_fn(in_tensor)

    # Create device buffers
    var device_in = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_out_vals = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var device_out_idxs = ctx.enqueue_create_buffer[out_idx_type](
        out_idxs_shape.flattened_length()
    )

    ctx.enqueue_copy(device_in, in_host_ptr)
    var batch_size: Int

    @parameter
    if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:  # rank > 2
        var last_dim = input_shape[rank - 1]
        batch_size = Int(input_shape.flattened_length() / last_dim)

    # Create K buffers
    var K_shape = IndexList[1](batch_size)
    var K_host_ptr = UnsafePointer[Int64].alloc(K_shape.flattened_length())
    # NDBuffer needed for reduce_max
    var K_host_tensor = NDBuffer[DType.int64, 1](K_host_ptr, K_shape)
    for i in range(batch_size):
        K_host_ptr[i] = K

    var K_device_buffer = ctx.enqueue_create_buffer[DType.int64](
        K_shape.flattened_length()
    )
    ctx.enqueue_copy(K_device_buffer, K_host_ptr)
    ctx.synchronize()
    var max_k = Int(reduce_max(K_host_tensor))

    # Create layout tensors for kernel calls
    comptime out_vals_layout = Layout.row_major[rank]()
    comptime out_idxs_layout = Layout.row_major[rank]()
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE)

    var in_runtime_layout = RuntimeLayout[in_layout].row_major(input_shape)
    var out_vals_runtime_layout = RuntimeLayout[out_vals_layout].row_major(
        out_vals_shape
    )
    var out_idxs_runtime_layout = RuntimeLayout[out_idxs_layout].row_major(
        out_idxs_shape
    )
    var k_runtime_layout = RuntimeLayout[k_layout].row_major(
        IndexList[1](batch_size)
    )

    var device_in_lt = LayoutTensor[dtype, in_layout](
        device_in, in_runtime_layout
    )
    var device_out_vals_lt = LayoutTensor[dtype, out_vals_layout](
        device_out_vals, out_vals_runtime_layout
    )
    var device_out_idxs_lt = LayoutTensor[out_idx_type, out_idxs_layout](
        device_out_idxs, out_idxs_runtime_layout
    )
    var k_lt = LayoutTensor[DType.int64, k_layout](
        K_device_buffer, k_runtime_layout
    )

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

    # Copy results back to host
    ctx.enqueue_copy(topk_vals_host_ptr, device_out_vals)
    ctx.enqueue_copy(topk_idxs_host_ptr, device_out_idxs)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu_ptr = UnsafePointer[Scalar[dtype]].alloc(
            out_vals_shape.flattened_length()
        )
        var topk_idxs_cpu_ptr = UnsafePointer[Int64].alloc(
            out_idxs_shape.flattened_length()
        )

        # Create layout tensors for CPU reference
        var in_host_lt = LayoutTensor[dtype, in_layout](
            in_host_ptr, in_runtime_layout
        )
        var topk_vals_cpu_lt = LayoutTensor[dtype, out_vals_layout](
            topk_vals_cpu_ptr, out_vals_runtime_layout
        )
        var topk_idxs_cpu_lt = LayoutTensor[DType.int64, out_vals_layout](
            topk_idxs_cpu_ptr, out_vals_runtime_layout
        )
        var k_host_lt = LayoutTensor[DType.int64, k_layout](
            K_host_ptr, k_runtime_layout
        )

        _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
            in_host_lt,
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

        for i in range(out_vals_shape.flattened_length()):
            assert_almost_equal(
                topk_vals_host_ptr[i],
                topk_vals_cpu_ptr[i],
            )

            @parameter
            if dtype is DType.float32:
                assert_equal(
                    topk_idxs_host_ptr[i],
                    topk_idxs_cpu_ptr[i].cast[out_idx_type](),
                )

        # Free CPU reference buffers
        topk_vals_cpu_ptr.free()
        topk_idxs_cpu_ptr.free()

    # Free host pointers
    in_host_ptr.free()
    topk_vals_host_ptr.free()
    topk_idxs_host_ptr.free()
    K_host_ptr.free()

    # Free device buffers
    _ = device_in^
    _ = device_out_vals^
    _ = device_out_idxs^
    _ = K_device_buffer^


@parameter
fn fill_random[dtype: DType](buffer: LayoutTensor[mut=True, dtype, **_]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.size()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.ptr[i] = random_value.cast[dtype]()


@parameter
fn fill_constant[dtype: DType](buffer: LayoutTensor[mut=True, dtype, **_]):
    var total_elements = buffer.size()
    for i in range(total_elements):
        if i % 3 == 1:
            buffer.ptr[i] = 1.0
        else:
            buffer.ptr[i] = 0.0


@parameter
fn fill_iota[dtype: DType](buf: LayoutTensor[mut=True, dtype, **_]):
    iota(buf.ptr, buf.runtime_layout.shape.value.flattened_length())


struct TestCase[_sampling: Bool, _largest: Bool = True](ImplicitlyCopyable):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var N: Int
    var K: Int
    var block_size: Int
    var batch_size: Int
    var num_blocks_per_input: OptionalReg[Int]

    fn __init__(
        out self,
        N: Int,
        K: Int,
        block_size: Int,
        batch_size: Int,
        num_blocks_per_input: OptionalReg[Int] = None,
    ):
        self.N = N
        self.K = K
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_blocks_per_input = num_blocks_per_input


struct TestCaseMultiRank[_sampling: Bool, rank: Int, _largest: Bool = True](
    ImplicitlyCopyable
):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var input_shape: IndexList[Self.rank]
    var K: Int
    var block_size: OptionalReg[Int]
    var num_blocks_per_input: OptionalReg[Int]

    fn __init__(
        out self,
        input_shape: IndexList[Self.rank],
        K: Int,
        block_size: OptionalReg[Int] = None,
        num_blocks_per_input: OptionalReg[Int] = None,
    ):
        self.input_shape = input_shape
        self.K = K
        self.block_size = block_size
        self.num_blocks_per_input = num_blocks_per_input


fn print_test_case(test_case: TestCase):
    var num_blocks_per_in_msg = "auto"
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    print(
        "==== Running Top-K sampling=",
        test_case.sampling,
        ", N=",
        test_case.N,
        ", K=",
        test_case.K,
        ", block_size=",
        test_case.block_size,
        ", batch_size=",
        test_case.batch_size,
        ", num_blocks_per_input=",
        num_blocks_per_in_msg,
    )


fn print_test_case(test_case: TestCaseMultiRank):
    var num_blocks_per_in_msg = "auto"
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    var block_size_msg = "auto"
    if test_case.block_size:
        block_size_msg = String(test_case.block_size.value())
    print(
        "==== Running Top-K sampling=",
        test_case.sampling,
        ", input_shape=",
        test_case.input_shape,
        ", K=",
        test_case.K,
        ", block_size=",
        block_size_msg,
        ", num_blocks_per_input=",
        num_blocks_per_in_msg,
    )


fn test_min_topk[dtype: DType](ctx: DeviceContext) raises:
    comptime llama3_vocab_size = 128256

    comptime test_case0 = TestCase[_sampling=False, _largest=False](
        N=1024,
        K=1,
        block_size=256,
        batch_size=1,
    )
    print_test_case(test_case0)
    test_case_batched[
        dtype,
        fill_iota,
        out_idx_type = DType.uint64,
    ](ctx, test_case0)

    comptime test_case1 = TestCase[_sampling=False, _largest=False](
        N=32000,
        K=5,
        block_size=512,
        batch_size=16,
        num_blocks_per_input=8,
    )
    print_test_case(test_case1)
    test_case_batched[
        dtype,
        fill_iota,
    ](ctx, test_case1)

    comptime test_case2 = TestCase[_sampling=False, _largest=False](
        N=llama3_vocab_size,
        K=10,
        block_size=1024,
        batch_size=64,
        num_blocks_per_input=6,
    )
    print_test_case(test_case2)
    # Changed from fill_random to fill_iota for deterministic test data.
    # With random data, duplicate/similar values can cause GPU and CPU to
    # produce different (but equally valid) index orderings.
    test_case_batched[
        dtype,
        fill_iota,
    ](ctx, test_case2)


fn test_multi_rank[dtype: DType, sampling: Bool](ctx: DeviceContext) raises:
    comptime llama3_vocab_size = 128256

    comptime test_case_multi_rank1 = TestCaseMultiRank[
        _sampling=sampling, rank=1, _largest=True
    ](
        input_shape=IndexList[1](4096),
        K=10,
        block_size=256,
    )
    print_test_case(test_case_multi_rank1)
    test_case_multi_rank[dtype, fill_iota](ctx, test_case_multi_rank1)

    comptime test_case_multi_rank2 = TestCaseMultiRank[
        _sampling=sampling, rank=2, _largest=True
    ](
        input_shape=IndexList[2](10, 1024),
        K=5,
        block_size=512,
    )
    print_test_case(test_case_multi_rank2)
    test_case_multi_rank[dtype, fill_iota](ctx, test_case_multi_rank2)

    comptime test_case_multi_rank3 = TestCaseMultiRank[
        _sampling=sampling, rank=3, _largest=True
    ](
        input_shape=IndexList[3](2, 128, llama3_vocab_size),
        K=5,
        num_blocks_per_input=2,
    )
    print_test_case(test_case_multi_rank3)
    test_case_multi_rank[dtype, fill_iota](ctx, test_case_multi_rank3)


def main():
    comptime llama3_vocab_size = 128256
    with DeviceContext() as ctx:
        comptime dtype = DType.float32
        comptime bf16_type = DType.bfloat16

        # var test_cases: [TestCase] = []
        # var N_values = [1024, 32000, 128256]
        # var K_values = [1, 5, 10]
        # var block_size_values = [256, 512, 1024]
        # var batch_size_values = [1, 16, 64, 256]
        # var _samplingvalues = [False, True]

        comptime test_case0 = TestCase[_sampling=False](
            N=1024,
            K=256,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case0)
        test_case_batched[
            dtype,
            fill_iota,
            out_idx_type = DType.uint64,
        ](ctx, test_case0)

        comptime test_case1 = TestCase[_sampling=False](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case1)
        test_case_batched[
            dtype,
            fill_iota,
            out_idx_type = DType.uint64,
        ](ctx, test_case1)

        comptime test_case2 = TestCase[_sampling=False](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case2)
        test_case_batched[dtype, fill_iota](ctx, test_case2)

        comptime test_case3 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case3)
        # Changed from fill_random to fill_iota for deterministic test data
        test_case_batched[dtype, fill_iota](ctx, test_case3)

        comptime test_case4 = TestCase[_sampling=True](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case4)
        test_case_batched[
            dtype,
            fill_iota,
        ](ctx, test_case4)

        comptime test_case5 = TestCase[_sampling=True](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case5)
        test_case_batched[dtype, fill_iota](ctx, test_case5)

        comptime test_case6 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case6)
        test_case_batched[
            dtype,
            fill_random,
            out_idx_type = DType.int32,
        ](ctx, test_case6)

        comptime test_case7 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=16,
        )
        print_test_case(test_case7)
        test_case_batched[dtype, fill_iota](ctx, test_case7)

        comptime test_case8 = TestCase[_sampling=False](
            N=32000,
            K=25,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=2,
        )
        print_test_case(test_case8)
        test_case_batched[dtype, fill_iota](ctx, test_case8)

        comptime test_case9 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=1024,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case9)
        test_case_batched[dtype, fill_iota](ctx, test_case9)

        comptime test_case10 = TestCase[_sampling=True](
            N=1024,
            K=10,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case10)
        test_case_batched[dtype, fill_iota](ctx, test_case10)

        comptime test_case11 = TestCase[_sampling=True](
            N=32000,
            K=1,
            block_size=512,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case11)
        test_case_batched[bf16_type, fill_random](ctx, test_case11)

        comptime test_case12 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case12)
        test_case_batched[bf16_type, fill_random](ctx, test_case12)

        comptime test_case13 = TestCase[_sampling=False](
            N=1024,
            K=10,
            block_size=1024,
            batch_size=256,
        )
        print_test_case(test_case13)
        test_case_batched[
            bf16_type,
            fill_iota,
            out_idx_type = DType.uint64,
        ](ctx, test_case13)

        comptime test_case14 = TestCase[_sampling=False](
            N=32000,
            K=1,
            block_size=512,
            batch_size=1,
        )
        print_test_case(test_case14)
        test_case_batched[bf16_type, fill_random](ctx, test_case14)

        comptime test_case15 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case15)
        test_case_batched[bf16_type, fill_iota](ctx, test_case15)

        comptime test_case16 = TestCase[_sampling=True](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case16)
        test_case_batched[
            bf16_type,
            fill_iota,
            out_idx_type = DType.int64,
        ](ctx, test_case16)

        comptime test_case17 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=16,
        )
        print_test_case(test_case17)
        test_case_batched[bf16_type, fill_random](ctx, test_case17)

        comptime test_case18 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case18)
        test_case_batched[bf16_type, fill_random](ctx, test_case18)

        comptime test_case19 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case19)
        test_case_batched[bf16_type, fill_random](ctx, test_case19)

        # Test with identical values
        comptime test_case20 = TestCase[_sampling=False](
            N=50,
            K=25,
            block_size=256,
            batch_size=2,
        )
        print_test_case(test_case20)
        test_case_batched[dtype, fill_constant](ctx, test_case20)

        comptime test_case_21 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=75,
            block_size=512,
            batch_size=2,
            num_blocks_per_input=8,
        )
        print_test_case(test_case_21)
        test_case_batched[DType.float32, fill_random](ctx, test_case_21)

        comptime test_case_22 = TestCase[_sampling=False](
            N=50,
            K=25,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case_22)
        test_case_batched[DType.float32, fill_random](ctx, test_case_22)

        # Test with zero batch size
        comptime test_case_23 = TestCase[_sampling=False](
            N=1024,
            K=1,
            block_size=256,
            batch_size=0,
        )
        print_test_case(test_case_23)
        test_case_batched[dtype, fill_iota](ctx, test_case_23)

        # Run minimum top-k tests
        test_min_topk[dtype](ctx)

        # Run multi-rank tests
        test_multi_rank[dtype, False](ctx)
        test_multi_rank[dtype, True](ctx)
        test_multi_rank[bf16_type, False](ctx)
        test_multi_rank[bf16_type, True](ctx)
