# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from random import rand, seed
from sys import alignof, argv, simdwidthof, sizeof
from sys._assembly import inlined_assembly
from memory.unsafe import bitcast
from random import randint

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
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from gpu.intrinsics import lop
from gpu.host.info import DEFAULT_GPU_ARCH
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
from linalg.matmul_gpu import _matmul_gpu
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
fn multistage_mma_q[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
    group_size: Int,
    pack_factor: Int,
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout,
    scales_type: DType,
    scales_layout: Layout,
    scales_smem_layout: Layout,
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
    scales_smem_iter_arg: LayoutTensorIter[
        scales_type,
        scales_smem_layout,
        address_space = AddressSpace.SHARED, **_,
    ],
    scales_iter_arg: LayoutTensorIter[scales_type, scales_layout, **_],
    num_iters: Int,
    /,
    *,
    num_a_rows: OptionalReg[Int] = None,
    num_b_rows: OptionalReg[Int] = None,
):
    alias simd_size = simdwidthof[a_type]()
    alias simd_b_size = simdwidthof[b_type]()
    alias num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    alias repack_tile = Index(64, 16)

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_iter = a_iter_arg
    var b_iter = b_iter_arg
    var scales_iter = scales_iter_arg
    var a_smem_iter = a_smem_iter_arg
    var scales_smem_iter = scales_smem_iter_arg
    # work around inout argument can't have default value.
    alias async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias async_copy_b_layout = Layout.row_major(
        num_threads * simd_b_size // b_smem_layout.stride[0].value(),
        b_smem_layout.stride[0].value() // simd_b_size,
    )
    alias swizzle_b = transpose_b or b_type.is_half_float()

    alias async_copy_scales_layout = Layout.row_major(1, 32)

    alias smem_reg_scales_layout = Layout.row_major(8, 4)

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
                    swizzle=False,
                ](
                    b_smem_tile.vectorize[1, simd_b_size](),
                    b_iter[]
                    .bitcast[b_type, address_space = AddressSpace.GENERIC]()
                    .vectorize[1, simd_b_size](),
                )

                b_iter._incr()

            # Every group_size rows share a scale
            # Only load scales when necessary
            @parameter
            if scales_iter.address_space == AddressSpace.GENERIC:
                if stage % (group_size // BK) == 0:
                    alias scales_stage = stage // (group_size // BK)
                    var scales_smem_tile = scales_smem_iter.next_unsafe(
                        scales_stage
                    )[]

                    # We only need one warp for copying scales...
                    if tid < 32:
                        var src_fragments = scales_iter[].bitcast[
                            scales_type, address_space = AddressSpace.GENERIC
                        ]().vectorize[1, 4]().distribute[
                            async_copy_scales_layout
                        ](
                            int(tid)
                        )
                        var dst_fragments = scales_smem_tile.vectorize[
                            1, 4
                        ]().distribute[async_copy_scales_layout](int(tid))

                        dst_fragments.copy_from_async[](src_fragments)

                    scales_iter._incr()

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

    var b_reg_tiles = tb[a_type]().row_major[
        2 * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2]()

    var scales_reg_tiles = tb[scales_type]().row_major[
        num_n_mmas, 1
    ]().local().alloc().vectorize[1, 1]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)

    alias b_wtile_dim0 = WN // repack_tile[0] if transpose_b else (
        (BK * repack_tile[0]) // pack_factor
    )
    alias b_wtile_dim1 = (
        (BK * repack_tile[0]) // pack_factor
    ) if transpose_b else WN // repack_tile[0]
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )
    var scales_warp_tile = scales_smem_iter[].tile[ceildiv(BK, group_size), WN](
        0, int(warp_x)
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    mma_op.load_a[swizzle_a](
        a_warp_tile, a_reg_tiles[0].vectorize[1, a_frag_size]()
    )

    # load scales into regs
    # for thread 0-3, scales for col 0, 8, 16, ..., 56 are stored locally
    # thread 4-7 stores scales for col 1, 9, 17, ..., 57
    scales_reg_tiles.vectorize[simd_size, 1]().copy_from(
        scales_warp_tile.vectorize[1, simd_size]().distribute[
            smem_reg_scales_layout, axis=0
        ](int(lane_id))
    )

    mma_op.load_b(b_warp_tile, b_reg_tiles[0], scales_reg_tiles, 0)

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

                # prefecth scales into regs every (group_size) rows
                if (k_tile_id + 1) % (group_size // BK) == 0:
                    scales_smem_iter._incr()
                    scales_warp_tile = scales_smem_iter[].tile[
                        ceildiv(BK, group_size), WN
                    ](0, int(warp_x))
                    scales_reg_tiles.vectorize[simd_size, 1]().copy_from(
                        scales_warp_tile.vectorize[1, simd_size]().distribute[
                            smem_reg_scales_layout, axis=0
                        ](int(lane_id))
                    )

            mma_op.load_a[swizzle_a](
                a_warp_tile,
                a_reg_tiles[next].vectorize[1, a_frag_size](),
                (k_mma + 1) % num_k_mmas,
            )
            mma_op.load_b(
                b_warp_tile,
                b_reg_tiles[next],
                scales_reg_tiles,
                (k_mma + 1) % num_k_mmas,
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
                            swizzle=False,
                        ](
                            b_smem_prefetch_tile.vectorize[1, simd_b_size](),
                            b_iter[]
                            .bitcast[
                                b_type, address_space = AddressSpace.GENERIC
                            ]()
                            .vectorize[1, simd_b_size](),
                        )

                        b_iter._incr()

                    @parameter
                    if scales_iter.address_space == AddressSpace.GENERIC:
                        # Every group_size rows share a scale
                        # Only load scales when necessary
                        if (k_tile_id + num_pipeline_stages - 1) % (
                            group_size // BK
                        ) == 0:
                            var scales_smem_tile = scales_smem_iter.next_unsafe(
                                num_scales_stages - 1
                            )[]

                            # We only need one warp for copying scales...
                            if tid < 32:
                                var src_fragments = scales_iter[].bitcast[
                                    scales_type,
                                    address_space = AddressSpace.GENERIC,
                                ]().vectorize[1, 4]().distribute[
                                    async_copy_scales_layout
                                ](
                                    int(tid)
                                )
                                var dst_fragments = scales_smem_tile.vectorize[
                                    1, 4
                                ]().distribute[async_copy_scales_layout](
                                    int(tid)
                                )

                                dst_fragments.copy_from_async[](
                                    src_fragments, base_offset=0
                                )

                            scales_iter._incr()

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()


fn multistage_gemm_q[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    scales_type: DType,
    scales_layout: Layout,
    group_size: Int,
    pack_factor: Int,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
    scales: LayoutTensor[
        scales_type,
        scales_layout,
    ],
):
    alias simd_size = simdwidthof[c_type]()

    alias repack_tile = Index(64, 16)

    var M: UInt = c.dim(0)
    var N: UInt = c.dim(1)
    var K: UInt = a.dim(1)

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
    # constrained[
    #    (a_type is DType.float32 or a_type is DType.bfloat16)
    #    and a_type == b_type == c_type,
    #    "Pipeline gemm only supports tf32 or BF16 mma",
    # ]()

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
    alias b_smem_size = num_pipeline_stages * BK * BN // pack_factor
    alias BD_0 = BN // repack_tile[0]
    alias BD_1 = (BK * repack_tile[0]) // pack_factor
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)

    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_smem_size)

    # multiple stages may share the same scales
    alias num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * BK, group_size
    ) + 1
    var scales_smem = (b_smem + b_smem_size).bitcast[Scalar[scales_type]]()
    alias scales_smem_size = num_scales_stages * BN * ceildiv(BK, group_size)
    alias scales_smem_layout = Layout.row_major(ceildiv(BK, group_size), BN)

    var scales_smem_iter = LayoutTensorIter[
        scales_type,
        scales_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](scales_smem, scales_smem_size)

    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = args_to_tuple[transpose_b](0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )
    var scales_gmem_iter = scales.tiled_iterator[
        ceildiv(BK, group_size), BN, axis=0
    ](b_tile_coords[1], b_tile_coords[0])

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

    multistage_mma_q[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
        transpose_b,
        group_size,
        pack_factor,
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        scales_smem_iter,
        scales_gmem_iter,
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

        if M % BM == 0:
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
            copy_sram_to_dram[
                thread_layout = Layout.row_major(
                    WARP_SIZE * simd_size // WN, WN // simd_size
                ),
                swizzle=swizzle,
            ](
                c_gmem_warp_tile.vectorize[1, simd_size](),
                accum_smem_warp_tile.vectorize[1, simd_size](),
                c_gmem_warp_tile.distance(c.ptr),
                M,
                N,
            )

    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            c_gmem_warp_tile.vectorize[1, 2](),
            c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
            c_gmem_warp_tile.distance(c.ptr),
            M,
            N,
        )


# this kernel dequantizes a repacked INT4 matrix into bf16 format
# Assuming a 64x16 (nxk) packing scheme
# Tile [i, j] stores part of the original matrix [i*64:(i+1)*64, j*16:(j+1)*16]
# Within each tile, weights are repacked similarly to the Marlin kernel.
# The memory address for tile [i, j] is (i * (N//64) + j) * tile_size,
# where tile_size is 64 * 16 * 4 / pack_factor = 512 Bytes.
fn create_ref_b[
    type_q: DType,
    type_b: DType,
    scales_type: DType,
    b_q_layout: Layout,
    b_layout: Layout,
    scales_layout: Layout,
    group_size: Int,
    pack_factor: Int,
](
    b_q: LayoutTensor[type_q, b_q_layout],
    b_out: LayoutTensor[type_b, b_layout],
    scales: LayoutTensor[scales_type, scales_layout],
):
    alias WARP_SIZE = 32
    alias BLOCK_N = 128
    alias BLOCK_K = 32
    alias repack_tile = Index(64, 16)
    alias TILE_N = 64
    alias TILE_K = 16
    alias num_k_warps = BLOCK_K // repack_tile[1]

    var tid: UInt = ThreadIdx.x()
    var warp_id: UInt = tid // WARP_SIZE
    var lane_id: UInt = tid % WARP_SIZE
    var block_idx = Index(int(BlockIdx.x()), int(BlockIdx.y()))
    var warp_x: UInt = warp_id // num_k_warps
    var warp_y: UInt = warp_id % num_k_warps

    var b_q_gmem_tile = b_q.tile[
        BLOCK_N // repack_tile[0], (BLOCK_K * repack_tile[0]) // pack_factor
    ](block_idx[0], block_idx[1])
    var warp_q_tile = b_q_gmem_tile.tile[
        1, (repack_tile[0] * repack_tile[1]) // pack_factor
    ](warp_x, warp_y)

    var scales_tile = scales.tile[ceildiv(BLOCK_K, group_size), BLOCK_N](
        (block_idx[1] * BLOCK_K) // group_size, block_idx[0]
    )
    var warp_scales_tile = scales_tile.tile[
        ceildiv(BLOCK_K, group_size), repack_tile[0]
    ](0, warp_x)
    alias smem_reg_scales_layout = Layout.row_major(8, 4)
    var scales_reg_tiles = tb[scales_type]().row_major[
        repack_tile[0] // 8, 1
    ]().local().alloc().vectorize[1, 1]()
    # load scales
    scales_reg_tiles.vectorize[8, 1]().copy_from(
        warp_scales_tile.vectorize[1, 8]().distribute[
            smem_reg_scales_layout, axis=0
        ](int(lane_id))
    )

    var b_out_tile = b_out.tile[BLOCK_N, BLOCK_K](block_idx[0], block_idx[1])
    var warp_out_tile = b_out_tile.tile[repack_tile[0], repack_tile[1]](
        warp_x, warp_y
    )
    var mma_tile_iter_1 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 0)
    var mma_tile_iter_2 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 1)

    var vec = bitcast[DType.int32, 4](warp_q_tile.vectorize[1, 4]()[lane_id])

    @always_inline
    fn int4tobf16(
        i4: Int32, scale: SIMD[DType.bfloat16, 1]
    ) -> SIMD[DType.bfloat16, 2]:
        alias MASK: Int32 = 0x000F000F
        alias I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300
        alias lut: Int32 = (0xF0 & 0xCC) | 0xAA
        var BF16_BIAS = SIMD[DType.bfloat16, 2](-136, -136)
        var BF16_SCALE = SIMD[DType.bfloat16, 2](scale, scale)
        var BF16_ZERO = SIMD[DType.bfloat16, 2](0, 0)
        var BF16_ONE = SIMD[DType.bfloat16, 2](1, 1)

        var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

        var v = bitcast[DType.bfloat16, 2](t).fma(BF16_ONE, BF16_BIAS).fma(
            BF16_SCALE, BF16_ZERO
        )
        return v

    alias write_back_layout = Layout.row_major(1, 32)
    alias write_back_type = __type_of(mma_tile_iter_1[].vectorize[1, 2]()[0])

    @parameter
    for i in range(0, TILE_N // 8, 2):
        var q_int = vec[i // 2]

        var v1 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        var v2 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v2)
        q_int >>= 4
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()

        v1 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        v2 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v2)
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()


fn test_quantized[
    type: DType
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    # quantization configs
    alias group_size = 128
    alias bit_width = 4
    alias has_zero_point = False
    alias pack_factor = 8

    alias repack_tile = Index(64, 16)

    print("test multistage matmul")
    alias static_M = m.dim.get()
    alias static_N = n.dim.get()
    alias static_K = k.dim.get()
    alias a_type = DType.bfloat16

    var M = m.value
    var N = n.value
    var K = k.value

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(
        n.dim // repack_tile[0], (k.dim * repack_tile[0]) // pack_factor
    )
    alias static_b_ref_shape = DimList(n.dim, k.dim)
    alias static_c_shape = DimList(m.dim, n.dim)
    alias static_scales_shape = DimList(k.dim // group_size, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(
        n.dim // repack_tile[0], (k.dim * repack_tile[0]) // pack_factor
    )
    var dynamic_b_ref_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)
    var dynamic_scales_shape = DimList(k.value // group_size, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)
    var scales_host = HostNDBuffer[a_type, 2, static_scales_shape](
        dynamic_scales_shape
    )
    var c_host_ref = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)

    zero(c_host.tensor)
    random(a_host.tensor)

    # elements of b matrix is between [-1, 1]
    random(scales_host.tensor, 0, 0.125)
    randint(
        b_host.tensor.data,
        n.value * (k.value // pack_factor),
        UInt.MIN,
        UInt.MAX,
    )

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var scales_device = DeviceNDBuffer[a_type, 2, static_scales_shape](
        dynamic_scales_shape, ctx=ctx
    )
    var b_device_ref = DeviceNDBuffer[a_type, 2, static_b_ref_shape](
        dynamic_b_ref_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[a_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy_to_device(scales_device.buffer, scales_host.tensor.data)

    alias c_layout = Layout.row_major[c_device.rank](c_device.shape)
    alias a_layout = Layout.row_major[c_device.rank](a_device.shape)
    alias b_layout = Layout.row_major[c_device.rank](b_device.shape)
    alias scales_layout = Layout.row_major[scales_device.rank](
        scales_device.shape
    )
    alias b_ref_layout = Layout.row_major[b_device_ref.rank](b_device_ref.shape)

    var c_tensor = LayoutTensor[a_type, c_layout,](
        c_device.buffer.ptr,
        RuntimeLayout[c_layout].row_major(c_device.tensor.dynamic_shape),
    )
    var a_tensor = LayoutTensor[a_type, a_layout,](
        a_device.buffer.ptr,
        RuntimeLayout[a_layout].row_major(a_device.tensor.dynamic_shape),
    )
    var b_tensor = LayoutTensor[type, b_layout,](
        b_device.buffer.ptr,
        RuntimeLayout[b_layout].row_major(b_device.tensor.dynamic_shape),
    )
    var scales_tensor = LayoutTensor[a_type, scales_layout,](
        scales_device.buffer.ptr,
        RuntimeLayout[scales_layout].row_major(
            scales_device.tensor.dynamic_shape
        ),
    )
    var b_ref_tensor = LayoutTensor[a_type, b_ref_layout,](
        b_device_ref.buffer.ptr,
        RuntimeLayout[b_ref_layout].row_major(
            b_device_ref.tensor.dynamic_shape
        ),
    )

    var c_device_ref = DeviceNDBuffer[a_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    alias kernels = MatmulKernels[a_type, type, a_type, True]()
    alias config = kernels.ampere_128x128_4

    # estimate smem usage
    var smem_usage: Int = 0
    var block_mnk = config.block_tile_shape
    var num_pipeline_stages = config.num_pipeline_stages
    var a_usage = block_mnk[0] * block_mnk[2] * num_pipeline_stages * sizeof[
        a_type
    ]()
    var b_usage = block_mnk[1] * block_mnk[2] * num_pipeline_stages * sizeof[
        type
    ]() // pack_factor
    var c_usage = block_mnk[0] * block_mnk[1] * sizeof[DType.float32]()
    var num_scales_stages = ceildiv(
        (num_pipeline_stages - 1) * block_mnk[2], group_size
    ) + 1
    var scales_usage = block_mnk[1] * ceildiv(
        block_mnk[2], group_size
    ) * num_scales_stages * sizeof[a_type]()
    smem_usage = a_usage + b_usage + scales_usage
    smem_usage = max(c_usage, smem_usage)

    alias gemm = multistage_gemm_q[
        a_type,  # c_type
        c_tensor.layout,
        a_type,  # a_type
        a_tensor.layout,
        type,  # b_type
        b_tensor.layout,
        a_type,
        scales_tensor.layout,
        group_size,
        pack_factor,
        True,
        config,
    ]

    var func = ctx.compile_function[
        gemm,
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm-2.ptx"),
    ](
        threads_per_block=int(config.num_threads()),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            smem_usage  # config.shared_mem_usage()
        ),
    )

    alias dequan = create_ref_b[
        type,
        a_type,
        a_type,
        b_tensor.layout,
        b_ref_tensor.layout,
        scales_tensor.layout,
        group_size,
        pack_factor,
    ]

    var func_dequan = ctx.compile_function[
        dequan,
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm-2.ptx"),
    ](
        threads_per_block=int(128),
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
                scales_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(int(config.num_threads()), 1, 1),
                shared_mem_bytes=smem_usage,
            )

        # Warmup
        for _ in range(nwarmup):
            ctx.enqueue_function(
                func,
                c_tensor,
                a_tensor,
                b_tensor,
                scales_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(int(config.num_threads()), 1, 1),
                shared_mem_bytes=smem_usage,
            )

        var nstime = ctx.execution_time[run_func](nrun) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(
            "Tranpose B ",
            "True",
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
        scales_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(int(config.num_threads()), 1, 1),
        shared_mem_bytes=smem_usage,  # config.shared_mem_usage(),
    )

    ctx.enqueue_function(
        func_dequan,
        b_tensor,
        b_ref_tensor,
        scales_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
    )

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)

    alias kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    alias config_ref = kernels_ref.ampere_128x128_4
    _matmul_gpu[
        target=DEFAULT_GPU_ARCH,
        use_tensor_core=True,
        transpose_b=True,
        config=config_ref,
    ](
        c_device_ref.tensor,
        a_device.tensor,
        b_device_ref.tensor,
        ctx,
    )

    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)

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
    _ = c_host
    _ = c_host_ref
    _ = scales_host
    _ = a_device
    _ = b_device
    _ = scales_device

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor
    _ = scales_tensor

    _ = func^
    _ = func_dequan^


def main():
    with DeviceContext() as ctx:
        test_quantized[DType.uint32](
            ctx, static[482](), static[6144](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, static[482](), static[4096](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, static[482](), static[28672](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, static[482](), static[4096](), static[14336]()
        )
        test_quantized[DType.uint32](
            ctx, static[482](), static[128256](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, dynamic(482), static[6144](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, dynamic(482), static[4096](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, dynamic(482), static[28672](), static[4096]()
        )
        test_quantized[DType.uint32](
            ctx, dynamic(482), static[4096](), static[14336]()
        )
        test_quantized[DType.uint32](
            ctx, dynamic(482), static[128256](), static[4096]()
        )
