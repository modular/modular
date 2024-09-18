# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from sys import env_get_string, env_get_int, alignof, sizeof, simdwidthof
from buffer import NDBuffer
from buffer.dimlist import DimList
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    BenchConfig,
    ThroughputMeasure,
)
from gpu import WARP_SIZE, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
    cublas_matmul,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.event import time_function
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_sram,
    copy_sram_to_dram,
)
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from linalg.cublas import cublas_matmul
from linalg._multistage_gemm_gpu import multistage_mma
from linalg.utils_gpu import block_swizzle
from memory.reference import _GPUAddressSpace as AddressSpace
from memory import UnsafePointer
from random import rand
from testing import assert_almost_equal

from utils.index import Index, StaticIntTuple


@always_inline
fn identity[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return tid


@always_inline
fn args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    @parameter
    if swap:
        return Tuple(arg_1, arg_0)
    else:
        return Tuple(arg_0, arg_1)


fn multistage_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        (a_type is DType.float32 or a_type is DType.bfloat16)
        and a_type == b_type == c_type,
        "Pipeline gemm only supports tf32 or BF16 mma",
    ]()

    constrained[
        (BK == 16 and a_type is DType.float32)
        or (BK == 32 and a_type is DType.bfloat16),
        "Pipeline gemm only supports BK = 16 w/ FP32 and BK = 32 w/ BF16.",
    ]()

    alias simd_size = simdwidthof[c_type]()

    var M = c.shape[0]()
    var N = c.shape[1]()
    var K = a.shape[1]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == num_threads // WARP_SIZE,
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle(
        Index(BlockIdx.x(), BlockIdx.y()), Index(N // BN, M // BM)
    ) if swizzle_block else Index(BlockIdx.x(), BlockIdx.y())

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    # Prepare circular shared memory buffer for A and B.
    # Each pipeline stage has its own buffer.
    var a_smem = external_memory[
        Scalar[a_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[a_type, simd_size]](),
    ]()
    alias a_smem_size = num_pipeline_stages * BM * BK
    var a_smem_iter = LayoutTensorIter[
        a_type, Layout.row_major(BM, BK), AddressSpace.SHARED, circular=True
    ](a_smem, a_smem_size)

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[b_type]]()
    alias b_smem_size = num_pipeline_stages * BK * BN
    alias BD_0 = BN if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b_smem_iter = LayoutTensorIter[
        b_type, b_smem_layout, AddressSpace.SHARED, circular=True
    ](b_smem, b_smem_size)

    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = args_to_tuple[transpose_b](0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias c_frag_size = frag_size[2]

    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0)

    var num_k_tiles = ceildiv(K, BK)

    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
        transpose_b,
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        num_k_tiles,
    )

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](block_idx[1], block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](int(warp_y), int(warp_x))

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if a_type in (DType.bfloat16, DType.float16):
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            accum_type,
            Layout.row_major(BM, BN),
            address_space = AddressSpace.SHARED,
        ](a_smem.bitcast[Scalar[accum_type]]())
        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            int(warp_y), int(warp_x)
        )
        copy_local_to_sram[thread_layout = Layout.row_major(8, 4)](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_threads * simd_size // BN, BN // simd_size
            )
        ](
            c_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )

    # Store FP32 results to FP32 buffer in global memory.
    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            c_gmem_warp_tile.vectorize[1, 2](),
            c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
        )


fn bench_gemm[
    type: DType, params: ParamConfig
](inout m: Bench, ctx: DeviceContext) raises:
    print("bench multistage gemm", str(params))

    alias M = params.M
    alias N = params.N
    alias K = params.K
    alias BM = params.BM
    alias BN = params.BN
    alias BK = params.BK
    alias WM = params.WM
    alias WN = params.WN
    alias num_pipeline_stages = params.num_pipeline_stages
    alias transpose_b = params.transpose_b

    alias num_threads = (BM // WM) * (BN // WN) * WARP_SIZE
    alias shared_mem_bytes = 80 * 1024

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var b_trans_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[type]].alloc(M * N)

    rand[type](a_host, M * K)
    rand[type](b_host, K * N)

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](K * N)
    var c_device = ctx.create_buffer[type](M * N)
    var c_device_ref = ctx.create_buffer[type](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device.ptr)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device.ptr)

    var c_tensor = LayoutTensor[type, c_layout](c_device.ptr)
    var a_tensor = LayoutTensor[type, a_layout](a_device.ptr)
    var b_tensor = LayoutTensor[type, b_layout](b_device.ptr)

    alias gemm = multistage_gemm[
        type,  # c_type
        c_layout,
        type,  # a_type
        a_layout,
        type,  # b_type
        b_layout,
        transpose_b,
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
    ]
    # TODO: The cache config doesn't really help here, see #38391.
    var func = ctx.compile_function[
        gemm,
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_ptx=Path("./pipeline-gemm.ptx"),
    ](
        threads_per_block=num_threads,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            shared_mem_bytes
        ),
    )

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func,
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=shared_mem_bytes,
            )

        b.iter_custom[kernel_launch](ctx)

    var flops = int(2.0 * M * N * K)
    var name = "multistage_gemm" + "/dtype=" + str(type) + "/M=" + str(
        M
    ) + "/N=" + str(N) + "/K=" + str(K)
    m.bench_function[bench_func](
        BenchId(name), ThroughputMeasure(BenchMetric.flops, flops)
    )

    ctx.enqueue_function(
        func,
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=shared_mem_bytes,
    )

    ctx.enqueue_copy_from_device(c_host, c_device)

    var c_buffer_ref = NDBuffer[type, 2, DimList(M, N)](c_device_ref.ptr)

    var handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
    check_cublas_error(
        cublas_matmul(
            handle,
            c_buffer_ref,
            a_buffer,
            b_buffer,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    )
    check_cublas_error(cublasDestroy(handle))

    ctx.enqueue_copy_from_device(c_host_ref, c_device_ref)

    alias rtol = 1e-3 if type == DType.float32 else 1e-4
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=rtol):
            print(
                i,
                c_host[i],
                c_host_ref[i],
                abs((c_host[i] - c_host_ref[i]) / c_host_ref[i]),
            )
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=rtol)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func^


@value
struct ParamConfig:
    var M: Int
    var N: Int
    var K: Int
    var BM: Int
    var BN: Int
    var BK: Int
    var WM: Int
    var WN: Int
    var num_pipeline_stages: Int
    var transpose_b: Bool

    fn __str__(self) -> String:
        return (
            "/M="
            + str(self.M)
            + "/N="
            + str(self.N)
            + "/K="
            + str(self.K)
            + "/BM="
            + str(self.BM)
            + "/BN="
            + str(self.BN)
            + "/BK="
            + str(self.BK)
        )


def main():
    var m = Bench(
        BenchConfig(max_iters=1, max_batch_size=1, min_warmuptime_secs=0)
    )

    # input params
    alias dtype = DType.bfloat16
    alias M = env_get_int["M", 8192]()
    alias N = env_get_int["N", 8192]()
    alias K = env_get_int["K", 128]()
    alias BM = env_get_int["BM", 128]()
    alias BN = env_get_int["BN", 128]()
    alias WM = env_get_int["WM", 32]()
    alias WN = env_get_int["WN", 32]()
    alias num_pipeline_stages = env_get_int["NUM_STAGES", 4]()
    alias transpose_b = True if env_get_string[
        "TRANSPOSE_B", "True"
    ]() == "True" else False

    # set params these based on input
    alias BK = 64 // sizeof[dtype]()
    alias params = ParamConfig(
        M, N, K, BM, BN, BK, WM, WN, num_pipeline_stages, transpose_b
    )
    print(str(params))

    with DeviceContext() as ctx:
        bench_gemm[dtype, params](m, ctx)
    m.dump_report()
