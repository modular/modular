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
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.host._nvidia_cuda import (
    TMADescriptor,
    create_tma_descriptor,
)
from gpu.id import block_idx, thread_idx, warp_id, lane_id
from gpu import WARP_SIZE, barrier, warp
from gpu.memory import (
    AddressSpace,
    external_memory,
    cp_async_bulk_tensor_shared_cluster_global,
)
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import Layout, LayoutTensor
from layout.int_tuple import IntTuple
from layout.layout_tensor import LayoutTensorIter
from layout.tma_async import (
    SharedMemBarrier,
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


@__llvm_arg_metadata(descriptor_a, `nvvm.grid_constant`)
@__llvm_arg_metadata(descriptor_b, `nvvm.grid_constant`)
fn gemv_tma_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BLOCK_SIZE_M: UInt,
    BLOCK_SIZE_K: UInt,
    ROWS_PER_WARP: UInt,
    NUM_PIPELINE_STAGES: UInt,
](
    descriptor_a: TMADescriptor,
    descriptor_b: TMADescriptor,
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
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

    alias a_smem_layout = Layout.row_major(BLOCK_SIZE_M, BLOCK_SIZE_K)

    alias b_smem_layout = Layout.row_major(BLOCK_SIZE_K)

    var descriptor_a_ptr = UnsafePointer(to=descriptor_a).bitcast[NoneType]()
    var descriptor_b_ptr = UnsafePointer(to=descriptor_b).bitcast[NoneType]()

    var a_smem_base = rebind[
        UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED]
    ](
        external_memory[
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_A_dynamic_shared_memory",
        ]()
    )

    alias a_size = a_smem_layout.size()

    var b_smem_base = (a_smem_base + NUM_PIPELINE_STAGES * a_size).bitcast[
        Scalar[dtype]
    ]()

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

    var b_smem = LayoutTensorIter[
        dtype,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ](
        b_smem_base,
        b_size * NUM_PIPELINE_STAGES,
    )

    var tma_mbar_ptr = (b_smem_base + b_size * NUM_PIPELINE_STAGES).bitcast[
        SharedMemBarrier
    ]()
    var tma_mbar = UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
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
                + current_block_size * size_of[dtype]()
            )

            cp_async_bulk_tensor_shared_cluster_global[
                Scalar[dtype],
                SharedMemBarrier,
                2,
            ](
                a_smem.next(stage)[].ptr,
                descriptor_a_ptr,
                UnsafePointer(to=tma_mbar[stage]),
                Index(UInt(col_offset), block_row),
            )
            cp_async_bulk_tensor_shared_cluster_global[
                Scalar[dtype],
                SharedMemBarrier,
                1,
            ](
                b_smem.next(stage)[].ptr,
                descriptor_b_ptr,
                UnsafePointer(to=tma_mbar[stage]),
                Index(UInt(col_offset)),
            )
            producer_phase.step()

        # Consumer: All threads wait and process.
        var stage = consumer_phase.index()
        var phase = consumer_phase.phase()

        tma_mbar[stage].wait(phase)

        # Process current buffer.
        var current_a_tile = a_smem.next_unsafe(Int(stage))[]
        var current_b_tile = b_smem.next_unsafe(Int(stage))[]

        for k_idx in range(0, current_block_size, WARP_SIZE):
            if k_idx + lane_id() < current_block_size:
                var col_idx = k_idx + lane_id()
                var b_val = current_b_tile[col_idx]

                @parameter
                for i in range(ROWS_PER_WARP):
                    var row_idx = warp_row_offset + i
                    if global_row_idx + i < M:
                        var a_val = current_a_tile[row_idx, col_idx]
                        dot_products[i] += rebind[__type_of(dot_products[i])](
                            a_val.cast[accum_type]() * b_val.cast[accum_type]()
                        )

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
](
    c_device_buffer: DeviceNDBuffer[dtype, rank_c, _],
    a_device_buffer: DeviceNDBuffer[dtype, rank_a, _],
    b_device_buffer: DeviceNDBuffer[dtype, rank_b, _],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
):
    var c_device = c_device_buffer.tensor
    var a_device = a_device_buffer.tensor
    var b_device = b_device_buffer.tensor

    # TODO: Tune further.
    alias THREAD_NUM = 1024
    alias BLOCK_SIZE_M = 64
    alias BLOCK_SIZE_K = UInt(256)
    # Number of warps per block for 128 threads.
    alias WARPS_PER_BLOCK = THREAD_NUM // WARP_SIZE
    alias ROWS_PER_WARP = UInt(BLOCK_SIZE_M // WARPS_PER_BLOCK)
    alias NUM_PIPELINE_STAGES = 1

    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    constrained[c.rank == 2]()
    constrained[a.rank == 2]()
    constrained[b.rank == 1]()

    var tma_desc_a = create_tma_descriptor[dtype, 2](
        a_device_buffer.buffer,
        (M, K),
        (K, 1),
        Index(BLOCK_SIZE_M, BLOCK_SIZE_K),
    )
    var tma_desc_b = create_tma_descriptor[dtype, 1](
        b_device_buffer.buffer,
        (K),
        (1),
        Index(BLOCK_SIZE_K),
    )
    # Shared memory needed for NUM_PIPELINE_STAGES A and B working tiles.
    # +8 bytes for each of NUM_PIPELINE_STAGES barriers.
    alias smem_use = (
        NUM_PIPELINE_STAGES * BLOCK_SIZE_M * BLOCK_SIZE_K * size_of[dtype]()
        + NUM_PIPELINE_STAGES * BLOCK_SIZE_K * size_of[dtype]()
        + 8 * NUM_PIPELINE_STAGES
    )

    alias kernel = gemv_tma_kernel[
        dtype,
        a.layout,
        b.layout,
        c.layout,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        ROWS_PER_WARP,
        NUM_PIPELINE_STAGES,
    ]

    ctx.enqueue_function[kernel](
        tma_desc_a,
        tma_desc_b,
        c,
        a,
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
    alias static_b_shape = DimList(k.dim)
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[dtype, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[dtype, 1, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[dtype, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[dtype, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[dtype, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[dtype, 1, static_b_shape](
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
        c_device,
        a_device,
        b_device,
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
                c_device,
                a_device,
                b_device,
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
        # Round TFLOPS to two decimal places for cleaner output.
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 3)
        print(
            String(M, "x", N, "x", K, ": DTYPE=", dtype),
            sectime * 1000,
            tflops_rounded,
        )
    else:
        # Compare with vendor BLAS for correctness.
        var b_2d = NDBuffer[dtype, 2](
            b_device.buffer._unsafe_ptr(),
            Index(K, 1),
            Index(1, K),
        )
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_2d,
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
