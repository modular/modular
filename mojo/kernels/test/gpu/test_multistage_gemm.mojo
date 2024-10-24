# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from random import rand
from sys import alignof, argv, simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu import WARP_SIZE, BlockIdx, GridDim, ThreadIdx, barrier, lane_id
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.event import time_function
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from layout.int_tuple import IntTuple
from layout.layout import *
from layout import RuntimeLayout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_sram,
    copy_sram_to_dram,
    copy_local_to_local,
)
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from linalg.cublas import cublas_matmul
from linalg.utils_gpu import block_swizzle
from memory import UnsafePointer

from utils.index import Index, IndexList
from internal_utils._utils import ValOrDim, dynamic, static

from layout.nd_buffer_stub import from_ndbuffer_row_major
from layout.swizzle import Swizzle, make_swizzle
from linalg.utils_gpu import (
    block_swizzle,
    MatmulConfig,
    MatmulKernels,
    select_config,
)
from layout.tensor_builder import LayoutTensorBuild as tb

from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_equal,
    fill,
    linspace,
    random,
    zero,
)


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


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


@always_inline
fn multistage_mma[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout,
    # Hack:
    /,
    *,
    swizzle_a: Bool = True,
    static_num_iters: Dim = Dim(),
    prefetch_init: Bool = True,
    continue_prefetch_b: Bool = False,
    transpose_b_next: Bool = False,
    b_next_gmem_layout: Layout = Layout(),
    b_next_smem_layout: Layout = Layout(),
    next_op_b_iter_alignment: Int = alignof[b_type](),
](
    c: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL],
    a_iter_arg: LayoutTensorIter[_, a_layout, **_],
    b_iter_arg: LayoutTensorIter[b_type, b_layout, **_],
    a_smem_iter_arg: LayoutTensorIter[
        a_type, a_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    inout b_smem_iter: LayoutTensorIter[
        b_type, b_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    num_iters: Int,
    /,
    *,
    num_a_rows: OptionalReg[Int] = None,
    num_b_rows: OptionalReg[Int] = None,
    next_op_b_iter: LayoutTensorIter[
        b_type, b_next_gmem_layout, alignment=next_op_b_iter_alignment
    ] = LayoutTensorIter[
        b_type, b_next_gmem_layout, alignment=next_op_b_iter_alignment
    ](),
):
    alias simd_size = simdwidthof[a_type]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var a_smem_iter = a_smem_iter_arg
    # work around inout argument can't have default value.
    var next_b_iter = next_op_b_iter

    alias async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias async_copy_b_layout = Layout.row_major(
        num_threads * simd_size // b_smem_layout.stride[0].value(),
        b_smem_layout.stride[0].value() // simd_size,
    )
    alias swizzle_b = transpose_b or b_type.is_half_float()

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    if prefetch_init:

        @parameter
        for stage in range(num_pipeline_stages - 1):

            @parameter
            if a_iter.address_space == AddressSpace.GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(stage)[]

                copy_dram_to_sram_async[
                    thread_layout=async_copy_a_layout,
                    swizzle=swizzle_a,
                ](
                    a_smem_tile.vectorize[1, simd_size](),
                    a_iter[]
                    .bitcast[a_type, address_space = AddressSpace.GENERIC]()
                    .vectorize[1, simd_size](),
                )

                a_iter._incr()

            @parameter
            if b_iter.address_space == AddressSpace.GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(stage)[]

                copy_dram_to_sram_async[
                    thread_layout=async_copy_b_layout,
                    swizzle=swizzle_b,
                ](
                    b_smem_tile.vectorize[1, simd_size](),
                    b_iter[]
                    .bitcast[b_type, address_space = AddressSpace.GENERIC]()
                    .vectorize[1, simd_size](),
                )

                b_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(num_pipeline_stages - 2)
        barrier()

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    # Register tiles.
    var a_reg_tiles = tb[a_type]().row_major[
        2 * num_m_mmas, a_frag_size
    ]().local().alloc().split[2]()

    var b_reg_tiles = tb[b_type]().row_major[
        2 * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)

    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    mma_op.load_a[swizzle_a](
        a_warp_tile, a_reg_tiles[0].vectorize[1, a_frag_size]()
    )

    mma_op.load_b(b_warp_tile, b_reg_tiles[0], warp_tile_coordn=int(warp_x))

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                a_smem_iter._incr()
                b_smem_iter._incr()

                a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
                b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                    b_wtile_coord0, b_wtile_coord1
                )

            mma_op.load_a[swizzle_a](
                a_warp_tile,
                a_reg_tiles[next].vectorize[1, a_frag_size](),
                (k_mma + 1) % num_k_mmas,
            )

            mma_op.load_b(
                b_warp_tile,
                b_reg_tiles[next],
                (k_mma + 1) % num_k_mmas,
                int(warp_x),
            )

            mma_op.mma(
                a_reg_tiles[current].vectorize[1, a_frag_size](),
                b_reg_tiles[current],
                c.vectorize[1, c_frag_size](),
            )

            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_iters:

                    @parameter
                    if a_iter.address_space == AddressSpace.GENERIC:
                        var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]

                        copy_dram_to_sram_async[
                            thread_layout=async_copy_a_layout,
                            swizzle=swizzle_a,
                        ](
                            a_smem_prefetch_tile.vectorize[1, simd_size](),
                            a_iter[]
                            .bitcast[
                                a_type, address_space = AddressSpace.GENERIC
                            ]()
                            .vectorize[1, simd_size](),
                        )

                        a_iter._incr()

                    @parameter
                    if b_iter.address_space == AddressSpace.GENERIC:
                        var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]

                        copy_dram_to_sram_async[
                            thread_layout=async_copy_b_layout,
                            swizzle=swizzle_b,
                        ](
                            b_smem_prefetch_tile.vectorize[1, simd_size](),
                            b_iter[]
                            .bitcast[
                                b_type, address_space = AddressSpace.GENERIC
                            ]()
                            .vectorize[1, simd_size](),
                        )

                        b_iter._incr()
                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()


fn multistage_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    __homogeneous_tile: Bool = True,
](
    c: LayoutTensor[
        c_type, c_layout, __experimental_non_homogeneous_tile=__homogeneous_tile
    ],
    a: LayoutTensor[
        a_type, a_layout, __experimental_non_homogeneous_tile=__homogeneous_tile
    ],
    b: LayoutTensor[
        b_type, b_layout, __experimental_non_homogeneous_tile=__homogeneous_tile
    ],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        a_type in (DType.float32, DType.bfloat16) and a_type == b_type,
        "Pipeline gemm only supports tf32 or BF16 mma",
    ]()

    alias simd_size = simdwidthof[c_type]()

    var M: UInt = c.dim(0)
    var N: UInt = b.dim(0) if transpose_b else b.dim(1)
    var K: UInt = b.dim(1) if transpose_b else b.dim(0)

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias num_pipeline_stages = config.num_pipeline_stages

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()

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

    constrained[
        num_warps_m * num_warps_n == num_threads // WARP_SIZE,
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle(
        (int(BlockIdx.x()), int(BlockIdx.y())),
        (int(GridDim.x()), int(GridDim.y())),
    ) if swizzle_block else Index(int(BlockIdx.x()), int(BlockIdx.y()))

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
        a_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = a_smem.alignment,
        circular=True,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    a_type,
                    Layout.row_major(BM, BK),
                    address_space = AddressSpace.SHARED,
                    alignment = a_smem.alignment,
                    circular=True,
                ]().ptr
            )
        ](a_smem),
        a_smem_size,
    )

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[b_type]]()
    alias b_smem_size = num_pipeline_stages * BK * BN
    alias BD_0 = BN if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)

    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
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

    var c_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)

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
    if c_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = tb[accum_type]().row_major[
            WM, WN
        ]().shared().view(a_smem.bitcast[accum_type]() + int(warp_id * WM * WN))

        copy_local_to_sram[
            thread_layout = Layout.row_major(8, 4),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                WARP_SIZE * simd_size // WN, WN // simd_size
            ),
            swizzle=swizzle,
        ](
            c_gmem_warp_tile.vectorize[1, simd_size](),
            accum_smem_warp_tile.vectorize[1, simd_size](),
        )

    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            c_gmem_warp_tile.vectorize[1, 2](),
            c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
            c_gmem_warp_tile.distance(c.ptr),
            M,
            N,
        )


fn test[
    type: DType, transpose_b: Bool
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    print("test multistage matmul")
    alias static_M = m.dim.get()
    alias static_N = n.dim.get()
    alias static_K = k.dim.get()

    var M = m.value
    var N = n.value
    var K = k.value

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)

    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    alias c_layout = Layout.row_major[c_device.rank](c_device.shape)
    alias a_layout = Layout.row_major[c_device.rank](a_device.shape)
    alias b_layout = Layout.row_major[c_device.rank](b_device.shape)

    var c_tensor = from_ndbuffer_row_major(c_device.tensor)
    var a_tensor = from_ndbuffer_row_major(a_device.tensor)
    var b_tensor = from_ndbuffer_row_major(b_device.tensor)

    alias kernels = MatmulKernels[type, type, type, transpose_b]()
    alias config = kernels.ampere_128x128_4

    alias gemm = multistage_gemm[
        type,  # c_type
        c_tensor.layout,
        type,  # a_type
        a_tensor.layout,
        type,  # b_type
        b_tensor.layout,
        transpose_b,
        config,
        __homogeneous_tile=True,
    ]

    var func = ctx.compile_function[
        gemm,
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_ptx=Path("./pipeline-gemm.ptx"),
    ](
        threads_per_block=int(config.num_threads()),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func,
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(int(config.num_threads()), 1, 1),
                shared_mem_bytes=config.shared_mem_usage(),
            )

        # Warmup
        for _ in range(nwarmup):
            ctx.enqueue_function(
                func,
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(int(config.num_threads()), 1, 1),
                shared_mem_bytes=config.shared_mem_usage(),
            )

        var nstime = ctx.execution_time[run_func](nrun) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(
            "Tranpose B ",
            transpose_b,
            nrun,
            " runs avg(s)",
            sectime,
            "TFlops/s",
            TFlop / sectime,
        )

    ctx.enqueue_function(
        func,
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(int(config.num_threads()), 1, 1),
        shared_mem_bytes=config.shared_mem_usage(),
    )

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)

    var handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
    check_cublas_error(
        cublas_matmul(
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    )
    check_cublas_error(cublasDestroy(handle))

    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    alias rtol = 1e-3 if type == DType.float32 else 1e-2
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

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor

    _ = func^


def main():
    with DeviceContext() as ctx:
        test[DType.bfloat16, True](
            ctx, static[482](), static[6144](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, static[482](), static[4096](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, static[482](), static[28672](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, static[482](), static[4096](), static[14336]()
        )
        test[DType.bfloat16, True](
            ctx, static[482](), static[128256](), static[4096]()
        )

        test[DType.bfloat16, True](
            ctx, dynamic(482), static[6144](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, dynamic(482), static[4096](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, dynamic(482), static[28672](), static[4096]()
        )
        test[DType.bfloat16, True](
            ctx, dynamic(482), static[4096](), static[14336]()
        )
        test[DType.bfloat16, True](
            ctx, dynamic(482), static[128256](), static[4096]()
        )
