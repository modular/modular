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

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx, warp_id, lane_id
from gpu import WARP_SIZE, barrier, warp
from gpu.memory import AddressSpace, external_memory
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from layout.tensor_core_async import tile_layout_k_major
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import Layout, LayoutTensor
from layout.int_tuple import IntTuple
from layout.layout_tensor import LayoutTensorIter
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
    PipelineState,
)
from linalg import vendor_blas
from math import ceildiv
from random import rand
from sys import size_of, argv
from utils import StaticTuple
from utils.index import Index
from utils.numerics import get_accum_type


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
fn gemv_tma_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    BLOCK_SIZE_M: UInt,
    BLOCK_SIZE_K: UInt,
    ROWS_PER_WARP: UInt,
    NUM_PIPELINE_STAGES: UInt,
](
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
    a_tma_op: TMATensorTile[dtype, a_layout, a_desc_layout],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    M: UInt,
    N: UInt,
    K: UInt,
):
    var tid = thread_idx.x
    var bidx = block_idx.x
    var block_row = bidx * BLOCK_SIZE_M

    var warp_row_offset = warp_id() * ROWS_PER_WARP
    var global_row_idx = block_row + warp_row_offset

    alias accum_type = get_accum_type[dtype]()

    alias a_smem_layout = tile_layout_k_major[
        dtype,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
    ]()

    alias b_smem_layout = Layout(IntTuple(BLOCK_SIZE_K, 1), IntTuple(1, 1))

    var a_smem_base = rebind[
        UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.SHARED, alignment2=128
        ]
    ](
        external_memory[
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_A_dynamic_shared_memory",
        ]()
    )

    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()

    var a_smem = LayoutTensorIter[
        dtype,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ](
        a_smem_base,
        a_size * NUM_PIPELINE_STAGES,
    )

    alias a_smem_tile_t = LayoutTensor[
        dtype,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias b_smem_tile_t = LayoutTensor[
        dtype,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    var b_smem = (a_smem_base + NUM_PIPELINE_STAGES * a_size).bitcast[
        Scalar[dtype]
    ]()
    var b_smem_tile = b_smem_tile_t(b_smem)

    alias a_expected_bytes = a_size * size_of[dtype]()

    var tma_mbar_ptr = (
        (b_smem + b_size)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[alignment=8]()
    )
    var tma_mbar = UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment2=8
    ](tma_mbar_ptr)

    # Initialize dot products for all rows before column processing.
    var dot_products = InlineArray[Scalar[accum_type], ROWS_PER_WARP](fill=0)

    if thread_idx.x == 0:

        @parameter
        for i in range(NUM_PIPELINE_STAGES):
            tma_mbar[i].init()

    barrier()

    # Double buffering.
    var consumer_phase = PipelineState[NUM_PIPELINE_STAGES]()
    var producer_phase = PipelineState[NUM_PIPELINE_STAGES](0, 1, 0)

    for col_offset in range(0, K, BLOCK_SIZE_K):
        var current_block_size = min(BLOCK_SIZE_K, K - col_offset)

        # Producer: Thread 0 loads data.
        if thread_idx.x == 0:
            var stage = producer_phase.index()
            tma_mbar[stage].expect_bytes(
                BLOCK_SIZE_M * current_block_size * size_of[dtype]()
            )
            a_tma_op.async_copy(
                a_smem.next(stage)[],
                tma_mbar[stage],
                (UInt(col_offset), block_row),
            )
            producer_phase.step()

        # Consumer: All threads wait and process.
        var stage = consumer_phase.index()
        var phase = consumer_phase.phase()

        tma_mbar[stage].wait(phase)
        barrier()

        # Load B vector.
        if tid < UInt(current_block_size):
            b_smem_tile[tid, 0] = b[col_offset + tid, 0]
        barrier()

        # Process current buffer.
        var current_tile = a_smem.next_unsafe(Int(stage))[]

        @parameter
        for i in range(ROWS_PER_WARP):
            var global_row = global_row_idx + i
            if global_row < M:
                # Each thread processes strided columns.
                for j in range(lane_id(), current_block_size, WARP_SIZE):
                    var a_val = current_tile[warp_row_offset + i, j]
                    var b_val = b_smem_tile[j, 0]
                    dot_products[i] += rebind[__type_of(dot_products[i])](
                        a_val.cast[accum_type]() * b_val.cast[accum_type]()
                    )
        barrier()

        consumer_phase.step()

    @parameter
    for i in range(ROWS_PER_WARP):
        var global_row = global_row_idx + i
        if global_row < M:
            var final_dot_product = warp.sum(dot_products[i])
            if lane_id() == 0:
                c[global_row, 0] = Scalar[dtype](final_dot_product)


def gemv_tma[
    dtype: DType,
    rank_c: Int,
    rank_a: Int,
    rank_b: Int,
    *,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    c_device: NDBuffer[dtype, rank_c, _, _],
    a_device: NDBuffer[dtype, rank_a, _, _],
    b_device: NDBuffer[dtype, rank_b, _, _],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
):
    # TODO: Tune further.
    alias THREAD_NUM = 1024
    alias BLOCK_SIZE_M = 64
    alias BLOCK_SIZE_K = UInt(256) if dtype == DType.bfloat16 else UInt(128)
    # Number of warps per block for 128 threads.
    alias WARPS_PER_BLOCK = THREAD_NUM // WARP_SIZE
    alias ROWS_PER_WARP = UInt(BLOCK_SIZE_M // WARPS_PER_BLOCK)
    alias NUM_PIPELINE_STAGES = 1

    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    constrained[c.rank == 2]()
    constrained[a.rank == 2]()
    constrained[b.rank == 2 and b.shape[1]() == 1 or b.rank == 1]()

    a_tma_op = create_tma_tile[
        dtype,
        2,
        Index(BLOCK_SIZE_M, BLOCK_SIZE_K),
    ](ctx, a)

    # Shared memory needed for NUM_PIPELINE_STAGES*A and B working tiles.
    # +8 bytes for each of NUM_PIPELINE_STAGES barriers.
    alias smem_use = (
        NUM_PIPELINE_STAGES * BLOCK_SIZE_M * BLOCK_SIZE_K * size_of[dtype]()
        + BLOCK_SIZE_K * size_of[dtype]()
        + 8 * NUM_PIPELINE_STAGES
    )

    alias kernel = gemv_tma_kernel[
        dtype,
        a_tma_op.layout,
        b.layout,
        c.layout,
        a_tma_op.desc_layout,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        ROWS_PER_WARP,
        NUM_PIPELINE_STAGES,
    ]

    ctx.enqueue_function[kernel](
        c,
        a_tma_op,
        b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_SIZE_M)),
        block_dim=(THREAD_NUM),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


def test_gemv_tma[
    dtype: DType
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    benchmark: Bool = False,
):
    var M = m.value
    var N = n.value
    var K = k.value

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(k.dim, n.dim)
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(k.value, n.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[dtype, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[dtype, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[dtype, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[dtype, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[dtype, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[dtype, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[dtype, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[dtype, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var at = a_host.tensor
    var bt = b_host.tensor
    rand[dtype](at.data, M * K)
    rand[dtype](bt.data, K * N)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    gemv_tma(
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    if benchmark:
        alias num_runs = 50
        alias num_warmup = 10

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            gemv_tma(
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                M,
                N,
                K,
                ctx,
            )

        for _ in range(num_warmup):
            run_func(ctx)
        ctx.synchronize()

        var nstime = ctx.execution_time[run_func](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        # Round TFLOPS to two decimal places for cleaner output
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 3)
        print(
            String(M, "x", N, "x", K, ": DTYPE=", dtype),
            sectime * 1000,
            tflops_rounded,
        )
    else:
        # Compare with vendor BLAS for correctness.
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=False,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()

        alias rtol = 1e-2
        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=0.0001,
            rtol=rtol,
        )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    with DeviceContext() as ctx:
        var benchmark = is_benchmark()
        test_gemv_tma[DType.bfloat16](
            ctx, dynamic(256), static[1](), static[256](), benchmark=benchmark
        )
        test_gemv_tma[DType.bfloat16](
            ctx, dynamic(4096), static[1](), static[4096](), benchmark=benchmark
        )

        test_gemv_tma[DType.float32](
            ctx, dynamic(256), static[1](), static[256](), benchmark=benchmark
        )
        test_gemv_tma[DType.float32](
            ctx, dynamic(4096), static[1](), static[4096](), benchmark=benchmark
        )
