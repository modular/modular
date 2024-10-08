# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s


from math import ceildiv, iota
from os import abort
from random import random_float64

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from collections import OptionalReg
from internal_utils import HostNDBuffer, DeviceNDBuffer
from random import random_float64
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from memory import UnsafePointer
from nn.topk_gpu import _topk_gpu

from utils import IndexList

alias idx_t = DType.index  # bad practice (matches the idx_t in the topk_gpu kernel)
alias DEBUG_BENCH = False


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](inout m: Bench, ctx: DeviceContext, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(inout m: Bencher):
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
    fill_fn: fn[rank: Int, type: DType] (inout NDBuffer[type, rank]) capturing [
        _
    ] -> None,
    sampling: Bool = True,
    rank: Int = 2,
](ctx: DeviceContext, test_case: TestCase) raises:
    # Fetch arguments
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    var block_size = test_case.block_size
    var num_blocks_per_input = test_case.num_blocks_per_input
    # Instantiate data in host memory
    var in_buffer = HostNDBuffer[type, rank](DimList(batch_size, N))
    var topk_vals = HostNDBuffer[type, rank](DimList(batch_size, K))
    var topk_idxs = HostNDBuffer[idx_t, rank](DimList(batch_size, K))

    # Fill the buffer with consecutive values
    fill_fn(in_buffer.tensor)
    # print("Input buffer: ", in_buffer.tensor)

    # Run the Top-K kernel
    alias topk_kernel = _topk_gpu[type, rank=rank, sampling=sampling]

    # Move data to device
    var device_in = DeviceNDBuffer[type, rank](DimList(batch_size, N), ctx=ctx)
    var device_out_vals = DeviceNDBuffer[type, rank](
        DimList(batch_size, K), ctx=ctx
    )
    var device_out_idxs = DeviceNDBuffer[idx_t, rank](
        DimList(batch_size, K), ctx=ctx
    )

    var num_blocks_stg1 = ceildiv(in_buffer.tensor.num_elements(), block_size)
    var device_local_topk_vals = DeviceNDBuffer[type, rank](
        DimList(batch_size, num_blocks_stg1 * K), ctx=ctx
    )
    var device_local_topk_idxs = DeviceNDBuffer[idx_t, rank](
        DimList(batch_size, num_blocks_stg1 * K), ctx=ctx
    )

    ctx.enqueue_copy_to_device(device_in.buffer, in_buffer.tensor.data)
    ctx.synchronize()

    @parameter
    if DEBUG_BENCH:
        var m = Bench()

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            topk_kernel(
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
            ctx.synchronize()

        alias msg = "tk-smpl" if sampling else "tk"
        time_kernel[run_func](m, ctx, msg)
        m.dump_report()

    topk_kernel(
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
    ctx.synchronize()

    # Copy results back to host
    ctx.enqueue_copy_from_device(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy_from_device(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    var _msg1: String = "Probability of chosen logit: " if sampling else "Top-K values: "
    var _msg2 = "Sample token index: " if sampling else "Top K indices: "

    @parameter
    if sampling:
        print("Sample token index: ", end="")
        for i in range(batch_size):
            print(topk_idxs.tensor.data[i * K], " ", end="")
            break  # Since all batch vectors are the same
        print()

        # @parameter
        # if DEBUG:
        #     print("Max logit prob: ", end="")
        #     for k in range(batch_size):
        #         for i in range(K):
        #             print(topk_vals.tensor.data[K * k + i], "  ", end="")
        #         print()
        #     print()
    else:
        print(_msg1, topk_vals.tensor)
        print(_msg2, topk_idxs.tensor)

    _ = topk_vals
    _ = topk_idxs
    _ = in_buffer
    _ = device_in
    _ = device_local_topk_vals
    _ = device_local_topk_idxs
    _ = device_out_vals
    _ = device_out_idxs


@parameter
fn fill_random[
    rank: Int, dtype: DType
](inout buffer: NDBuffer[dtype, rank],):
    alias min_val = 0.0
    alias max_val = 100000.0
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[rank: Int, type: DType](inout buf: NDBuffer[type, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


@value
struct TestCase[_sampling: Bool]:
    alias sampling = _sampling
    var N: Int
    var K: Int
    var block_size: Int
    var batch_size: Int
    var num_blocks_per_input: OptionalReg[Int]

    fn __init__(
        inout self,
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


fn print_test_case(test_case: TestCase):
    var num_blocks_per_in_msg = str("auto")
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = str(test_case.num_blocks_per_input.value())
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


fn main() raises:
    alias llama3_vocab_size = 128256
    with DeviceContext() as ctx:
        alias type = DType.float32
        var N: Int
        var K: Int
        var block_size: Int
        var batch_size: Int
        var sampling: Bool
        var num_blocks_per_input: Int = 0

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
            sampling = test_case1.sampling,
        ](ctx, test_case1)

        alias test_case2 = TestCase[_sampling=False](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case2)
        test_case_batched[type, fill_iota, sampling = test_case2.sampling](
            ctx, test_case2
        )

        alias test_case3 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case3)
        test_case_batched[type, fill_iota, sampling = test_case3.sampling](
            ctx, test_case3
        )

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
            sampling = test_case4.sampling,
        ](ctx, test_case4)

        alias test_case5 = TestCase[_sampling=True](
            N=32000,
            K=5,
            block_size=512,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case5)
        test_case_batched[type, fill_iota, sampling = test_case5.sampling](
            ctx, test_case5
        )

        alias test_case6 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=10,
            block_size=1024,
            batch_size=64,
            num_blocks_per_input=6,
        )
        print_test_case(test_case6)
        test_case_batched[type, fill_iota, sampling = test_case6.sampling](
            ctx, test_case6
        )

        alias test_case7 = TestCase[_sampling=False](
            N=1024,
            K=5,
            block_size=256,
            batch_size=16,
        )
        print_test_case(test_case7)
        test_case_batched[type, fill_iota, sampling = test_case7.sampling](
            ctx, test_case7
        )

        alias test_case8 = TestCase[_sampling=False](
            N=32000,
            K=10,
            block_size=512,
            batch_size=64,
            num_blocks_per_input=4,
        )
        print_test_case(test_case8)
        test_case_batched[type, fill_iota, sampling = test_case8.sampling](
            ctx, test_case8
        )

        alias test_case9 = TestCase[_sampling=False](
            N=llama3_vocab_size,
            K=1,
            block_size=1024,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case9)
        test_case_batched[type, fill_iota, sampling = test_case9.sampling](
            ctx, test_case9
        )

        alias test_case10 = TestCase[_sampling=True](
            N=1024,
            K=10,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case10)
        test_case_batched[type, fill_iota, sampling = test_case10.sampling](
            ctx, test_case10
        )

        alias test_case11 = TestCase[_sampling=True](
            N=32000,
            K=1,
            block_size=512,
            batch_size=256,
            num_blocks_per_input=8,
        )
        print_test_case(test_case11)
        test_case_batched[type, fill_iota, sampling = test_case11.sampling](
            ctx, test_case11
        )

        alias test_case12 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=1,
        )
        print_test_case(test_case12)
        test_case_batched[type, fill_iota, sampling = test_case12.sampling](
            ctx, test_case12
        )

        alias test_case13 = TestCase[_sampling=False](
            N=1024,
            K=10,
            block_size=1024,
            batch_size=256,
        )
        print_test_case(test_case13)
        test_case_batched[type, fill_iota, sampling = test_case13.sampling](
            ctx, test_case13
        )

        alias test_case14 = TestCase[_sampling=False](
            N=32000,
            K=1,
            block_size=512,
            batch_size=1,
        )
        print_test_case(test_case14)
        test_case_batched[type, fill_iota, sampling = test_case14.sampling](
            ctx, test_case14
        )

        alias test_case15 = TestCase[_sampling=True](
            N=llama3_vocab_size,
            K=5,
            block_size=1024,
            batch_size=16,
            num_blocks_per_input=8,
        )
        print_test_case(test_case15)
        test_case_batched[type, fill_iota, sampling = test_case15.sampling](
            ctx, test_case15
        )

        alias test_case16 = TestCase[_sampling=True](
            N=1024,
            K=5,
            block_size=256,
            batch_size=64,
        )
        print_test_case(test_case16)
        test_case_batched[type, fill_iota, sampling = test_case16.sampling](
            ctx, test_case16
        )
