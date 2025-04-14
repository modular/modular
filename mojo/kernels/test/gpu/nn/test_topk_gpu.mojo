# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1437
# UNSUPPORTED: H100-GPU
# RUN: %mojo-no-debug %s


from collections import OptionalReg
from math import ceildiv, iota
from os import abort
from random import random_float64

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from memory import UnsafePointer
from nn.topk import _top_k_cpu, _topk_gpu, topk_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList

alias DEBUG_BENCH = False
alias PRINT_OUTPUT = False


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
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[mut=True, type, rank]
    ) capturing [_] -> None,
    out_idx_type: DType = DType.index,
    rank: Int = 2,
](ctx: DeviceContext, test_case: TestCase) raises:
    # Fetch arguments
    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    alias largest = test_case.largest
    alias sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K

    var in_buffer = HostNDBuffer[type, rank](DimList(batch_size, N))
    var topk_vals = HostNDBuffer[type, rank](DimList(batch_size, K))
    var topk_idxs = HostNDBuffer[out_idx_type, rank](
        DimList(batch_size, out_idx_len)
    )

    # Fill the buffer with consecutive values
    fill_fn(in_buffer.tensor)

    # Move data to device
    var device_in = DeviceNDBuffer[type, rank](DimList(batch_size, N), ctx=ctx)
    var device_out_vals = DeviceNDBuffer[type, rank](
        DimList(batch_size, K), ctx=ctx
    )
    var device_out_idxs = DeviceNDBuffer[out_idx_type, rank](
        DimList(batch_size, out_idx_len), ctx=ctx
    )

    var num_blocks_per_input_: Int = ceildiv(
        N, block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    var device_local_topk_vals = DeviceNDBuffer[type, rank](
        DimList(batch_size, num_blocks_per_input_ * K), ctx=ctx
    )
    var device_local_topk_idxs = DeviceNDBuffer[out_idx_type, rank](
        DimList(batch_size, num_blocks_per_input_ * K), ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)
    ctx.synchronize()

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            _topk_gpu[sampling=sampling, largest=largest](
                ctx,
                K,
                device_in.tensor,
                device_local_topk_vals.tensor,
                device_local_topk_idxs.tensor,
                device_out_vals.tensor,
                device_out_idxs.tensor,
                block_size=block_size,
                num_blocks_per_input=num_blocks_per_input,
            )
            ctx.enqueue_copy(topk_vals.tensor.data, device_out_vals.buffer)
            ctx.enqueue_copy(topk_idxs.tensor.data, device_out_idxs.buffer)
            ctx.synchronize()

        alias msg = "tk-smpl-gpu" if sampling else String("tk-gpu")
        time_kernel[run_func](m, ctx, msg)

    _topk_gpu[sampling=sampling, largest=largest](
        ctx,
        K,
        device_in.tensor,
        device_local_topk_vals.tensor,
        device_local_topk_idxs.tensor,
        device_out_vals.tensor,
        device_out_idxs.tensor,
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    var _msg1: String = "Top-K values: "
    var _msg2 = "Sample token index: " if sampling else StaticString(
        "Top K indices: "
    )

    @parameter
    if sampling and PRINT_OUTPUT:
        print(_msg2, topk_idxs.tensor)
        print(_msg1, topk_vals.tensor)
    elif PRINT_OUTPUT:
        print(_msg1, topk_vals.tensor)
        print(_msg2, topk_idxs.tensor)

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu = HostNDBuffer[type, rank](DimList(batch_size, K))
        var topk_idxs_cpu = HostNDBuffer[DType.int64, rank](
            DimList(batch_size, K)
        )

        @parameter
        if DEBUG_BENCH:

            @always_inline
            @parameter
            fn run_func_cpu(ctx: DeviceContext) raises:
                _top_k_cpu[rank=rank, type=type, largest=largest](
                    in_buffer.tensor,
                    K,
                    rank - 1,
                    topk_vals_cpu.tensor,
                    topk_idxs_cpu.tensor,
                    1,
                    True,
                )

            time_kernel[run_func_cpu](m, ctx, "topk-cpu")

        _top_k_cpu[rank=rank, type=type, largest=largest](
            in_buffer.tensor,
            K,
            rank - 1,
            topk_vals_cpu.tensor,
            topk_idxs_cpu.tensor,
            1,
            True,
        )

        for i in range(topk_vals.tensor.num_elements()):
            assert_almost_equal(
                topk_vals.tensor.data[i],
                topk_vals_cpu.tensor.data[i],
            )

            @parameter
            if type == DType.float32:
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

    @parameter
    if DEBUG_BENCH:
        m.dump_report()


fn test_case_multi_rank[
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[mut=True, type, rank]
    ) capturing [_] -> None,
    rank: Int,
    out_idx_type: DType = DType.index,
](ctx: DeviceContext, test_case: TestCaseMultiRank[rank=rank, *_]) raises:
    # Fetch arguments
    var input_shape = test_case.input_shape
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    alias largest = test_case.largest
    alias sampling = test_case.sampling
    # Instantiate data in host memory
    var out_idx_len = 1 if sampling else K
    var out_vals_shape = input_shape
    out_vals_shape[rank - 1] = K
    var out_idxs_shape = input_shape
    out_idxs_shape[rank - 1] = out_idx_len

    var in_buffer = HostNDBuffer[type, rank](input_shape)
    var topk_vals = HostNDBuffer[type, rank](out_vals_shape)
    var topk_idxs = HostNDBuffer[out_idx_type, rank](out_idxs_shape)

    # Fill the buffer with consecutive values
    fill_fn(in_buffer.tensor)

    # Move data to device
    var device_in = DeviceNDBuffer[type, rank](input_shape, ctx=ctx)
    var device_out_vals = DeviceNDBuffer[type, rank](out_vals_shape, ctx=ctx)
    var device_out_idxs = DeviceNDBuffer[out_idx_type, rank](
        out_idxs_shape, ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)

    topk_gpu[sampling=sampling, largest=largest](
        ctx,
        K,
        device_in.tensor,
        device_out_vals.tensor,
        device_out_idxs.tensor,
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
    )

    # Copy results back to host
    ctx.enqueue_copy(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    # ASSERT equality with CPU topk kernel reference
    @parameter
    if not sampling:
        var topk_vals_cpu = HostNDBuffer[type, rank](out_vals_shape)
        var topk_idxs_cpu = HostNDBuffer[DType.int64, rank](out_idxs_shape)

        _top_k_cpu[rank=rank, type=type, largest=largest](
            in_buffer.tensor,
            K,
            rank - 1,
            topk_vals_cpu.tensor,
            topk_idxs_cpu.tensor,
            1,
            True,
        )

        for i in range(topk_vals.tensor.num_elements()):
            assert_almost_equal(
                topk_vals.tensor.data[i],
                topk_vals_cpu.tensor.data[i],
            )

            @parameter
            if type == DType.float32:
                assert_equal(
                    topk_idxs.tensor.data[i],
                    topk_idxs_cpu.tensor.data[i].cast[out_idx_type](),
                )


@parameter
fn fill_random[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    alias min_val = -1e9
    alias max_val = 1e9
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[rank: Int, type: DType](mut buf: NDBuffer[mut=True, type, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


@value
struct TestCase[_sampling: Bool, _largest: Bool = True]:
    alias sampling = _sampling
    alias largest = _largest
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
    ):
        self.N = N
        self.K = K
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_blocks_per_input = None


@value
struct TestCaseMultiRank[_sampling: Bool, rank: Int, _largest: Bool = True]:
    alias sampling = _sampling
    alias largest = _largest
    var input_shape: IndexList[rank]
    var K: Int
    var block_size: OptionalReg[Int]
    var num_blocks_per_input: OptionalReg[Int]

    fn __init__(
        out self,
        input_shape: IndexList[rank],
        K: Int,
        block_size: OptionalReg[Int] = None,
        num_blocks_per_input: OptionalReg[Int] = None,
    ):
        self.input_shape = input_shape
        self.K = K
        self.block_size = block_size
        self.num_blocks_per_input = num_blocks_per_input


fn print_test_case(test_case: TestCase):
    var num_blocks_per_in_msg = String("auto")
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
    var num_blocks_per_in_msg = String("auto")
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    var block_size_msg = String("auto")
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


fn test_min_topk[type: DType](ctx: DeviceContext) raises:
    alias llama3_vocab_size = 128256

    alias test_case0 = TestCase[_sampling=False, _largest=False](
        N=1024,
        K=1,
        block_size=256,
        batch_size=1,
    )
    print_test_case(test_case0)
    test_case_batched[
        type,
        fill_iota,
        out_idx_type = DType.uint64,
    ](ctx, test_case0)

    alias test_case1 = TestCase[_sampling=False, _largest=False](
        N=32000,
        K=5,
        block_size=512,
        batch_size=16,
        num_blocks_per_input=8,
    )
    print_test_case(test_case1)
    test_case_batched[
        type,
        fill_iota,
    ](ctx, test_case1)

    alias test_case2 = TestCase[_sampling=False, _largest=False](
        N=llama3_vocab_size,
        K=10,
        block_size=1024,
        batch_size=64,
        num_blocks_per_input=6,
    )
    print_test_case(test_case2)
    test_case_batched[
        type,
        fill_random,
    ](ctx, test_case2)


fn test_multi_rank[type: DType, sampling: Bool](ctx: DeviceContext) raises:
    alias llama3_vocab_size = 128256

    alias test_case_multi_rank1 = TestCaseMultiRank[
        _sampling=sampling, rank=1, _largest=True
    ](
        input_shape=IndexList[1](4096),
        K=10,
        block_size=256,
    )
    print_test_case(test_case_multi_rank1)
    test_case_multi_rank[type, fill_iota](ctx, test_case_multi_rank1)

    alias test_case_multi_rank2 = TestCaseMultiRank[
        _sampling=sampling, rank=2, _largest=True
    ](
        input_shape=IndexList[2](10, 1024),
        K=5,
        block_size=512,
    )
    print_test_case(test_case_multi_rank2)
    test_case_multi_rank[type, fill_iota](ctx, test_case_multi_rank2)

    alias test_case_multi_rank3 = TestCaseMultiRank[
        _sampling=sampling, rank=3, _largest=True
    ](
        input_shape=IndexList[3](2, 128, llama3_vocab_size),
        K=5,
        num_blocks_per_input=2,
    )
    print_test_case(test_case_multi_rank3)
    test_case_multi_rank[type, fill_iota](ctx, test_case_multi_rank3)


fn main() raises:
    alias llama3_vocab_size = 128256
    with DeviceContext() as ctx:
        alias type = DType.float32
        alias bf16_type = DType.bfloat16
        var N: Int
        var K: Int
        var block_size: Int
        var batch_size: Int
        var sampling: Bool
        var num_blocks_per_input: Int

        # var test_cases: [TestCase] = []
        # var N_values = [1024, 32000, 128256]
        # var K_values = [1, 5, 10]
        # var block_size_values = [256, 512, 1024]
        # var batch_size_values = [1, 16, 64, 256]
        # var _samplingvalues = [False, True]

        alias test_case1 = TestCase[_sampling=False](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case1)
        test_case_batched[
            type,
            fill_iota,
            out_idx_type = DType.uint64,
        ](ctx, test_case1)

        alias test_case2 = TestCase[_sampling=False](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case2)
        test_case_batched[type, fill_iota](ctx, test_case2)

        alias test_case3 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case3)
        test_case_batched[type, fill_random](ctx, test_case3)

        alias test_case4 = TestCase[_sampling=True](
            N=1024,
            K=1,
            block_size=256,
            batch_size=1,
        )
        print_test_case(test_case4)
        test_case_batched[
            type,
            fill_iota,
        ](ctx, test_case4)

        alias test_case5 = TestCase[_sampling=True](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case5)
        test_case_batched[type, fill_iota](ctx, test_case5)

        alias test_case6 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case6)
        test_case_batched[
            type,
            fill_random,
            out_idx_type = DType.int32,
        ](ctx, test_case6)

        alias test_case7 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=16,
        )
        print_test_case(test_case7)
        test_case_batched[type, fill_iota](ctx, test_case7)

        alias test_case8 = TestCase[_sampling=False](
            N=32000,
            K=25,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=2,
        )
        print_test_case(test_case8)
        test_case_batched[type, fill_iota](ctx, test_case8)

        alias test_case9 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=1024,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case9)
        test_case_batched[type, fill_iota](ctx, test_case9)

        alias test_case10 = TestCase[_sampling=True](
            N=1024,
            K=10,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case10)
        test_case_batched[type, fill_iota](ctx, test_case10)

        alias test_case11 = TestCase[_sampling=True](
            N=32000,
            K=1,
            block_size=512,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case11)
        test_case_batched[bf16_type, fill_random](ctx, test_case11)

        alias test_case12 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case12)
        test_case_batched[bf16_type, fill_random](ctx, test_case12)

        alias test_case13 = TestCase[_sampling=False](
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

        alias test_case14 = TestCase[_sampling=False](
            N=32000,
            K=1,
            block_size=512,
            batch_size=1,
        )
        print_test_case(test_case14)
        test_case_batched[bf16_type, fill_random](ctx, test_case14)

        alias test_case15 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case15)
        test_case_batched[bf16_type, fill_iota](ctx, test_case15)

        alias test_case16 = TestCase[_sampling=True](
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

        alias test_case17 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=16,
        )
        print_test_case(test_case17)
        test_case_batched[bf16_type, fill_random](ctx, test_case17)

        alias test_case18 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case18)
        test_case_batched[bf16_type, fill_random](ctx, test_case18)

        alias test_case19 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case19)
        test_case_batched[bf16_type, fill_random](ctx, test_case19)

        # Run minimum top-k tests
        test_min_topk[type](ctx)

        # Run multi-rank tests
        test_multi_rank[type, False](ctx)
        test_multi_rank[type, True](ctx)
        test_multi_rank[bf16_type, False](ctx)
        test_multi_rank[bf16_type, True](ctx)
