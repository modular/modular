# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from random import rand, randint, random_float64, seed
from sys import alignof, argv, simdwidthof, sizeof
from sys._assembly import inlined_assembly

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import WARP_SIZE, BlockIdx, GridDim, ThreadIdx, barrier, lane_id
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import DEFAULT_GPU_ARCH
from gpu.intrinsics import lop
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from layout import RuntimeLayout
from layout.int_tuple import IntTuple, to_int
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_local_to_sram,
    copy_sram_to_dram,
)
from layout.nd_buffer_stub import from_ndbuffer_row_major
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from linalg.cublas import cublas_matmul
from linalg.matmul_gpu import _matmul_gpu
from linalg.utils import GemmShape
from linalg.utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    block_swizzle,
    select_config,
)
from memory import UnsafePointer
from memory.unsafe import bitcast
from runtime.asyncrt import MojoCallContextPtr

from utils.index import Index, IndexList


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

    @always_inline
    @parameter
    fn _copy_tensor_to_sram[
        thread_layout: Layout, swizzle: Bool
    ](dst: LayoutTensor, src: LayoutTensor,):
        copy_dram_to_sram_async[thread_layout=thread_layout, swizzle=swizzle,](
            dst.vectorize[1, simd_size](),
            src.vectorize[1, simd_size](),
        )

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    if prefetch_init:

        @parameter
        for stage in range(num_pipeline_stages - 1):

            @parameter
            if a_iter.address_space == AddressSpace.GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(stage)[]

                _copy_tensor_to_sram[async_copy_a_layout, swizzle_a](
                    a_smem_tile, a_iter[]
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

                        _copy_tensor_to_sram[async_copy_a_layout, swizzle_a](
                            a_smem_prefetch_tile, a_iter[]
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
    b_packed_type: DType,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_packed_type, c_type, transpose_b],
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b_packed: LayoutTensor[b_packed_type, b_layout],
):
    alias simd_size = simdwidthof[c_type]()

    alias repack_tile = Index(64, 16)
    alias group_bytes = group_size // 2 + 2

    var M: UInt = c.dim(0)
    alias N = to_int(b_layout.shape[0])
    alias K = to_int(b_layout.shape[1]) // group_bytes * group_size

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias num_pipeline_stages = config.num_pipeline_stages

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()

    # Unpack quantized weights
    alias scales_type = DType.bfloat16
    alias b_type = DType.uint32
    alias b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b = LayoutTensor[b_type, b_weight_layout,](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    alias b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout,](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

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
        ceildiv(K, BK),
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

        var accum_smem_warp_tile = tb[c_type]().row_major[
            WM, WN
        ]().shared().view(
            a_smem.bitcast[Scalar[c_type]]() + int(warp_id * WM * WN)
        )

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
            c_reg_tile.vectorize[1, 2]().transpose(),
        )


# For a 4-bit weight matrix of shape (64 * TN, 16 * TK), we first
# repack elements within each 64x16 tile, and in memory every 8
# 4-bit elements are packed in one uint32.
#              K
#       0       16      32
#      0+-------+-------+-----+
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
# N   64+-------+-------+-----+   Matrix with 4-bit elements
#       | 64x16 | 64x16 | ... |
#       | tile  | tile  |     |
#    128+-------+-------+-----+
#
#
# Stored as uint32 tensor (64 * TN, 2 * TK):
#              K
#     0       2      4
#     0+------+------+-----+
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
# N  64+------+------+-----+    uint32 Matrix
#      | 64x2 | 64x2 | ... |
#      | tile | tile |     |
#   128+------+------+-----+
#
# Elements within each tile are stored continously
#                  K
#      0     1     2     3     4
#     0+-----+-----+-----+-----+
#      |  0  |  1  | 128 | 129 |
#     1+-----+-----+-----+-----+
#      |  2  |  3  | 130 | 131 |
#     2+-----+-----+-----+-----+
#      |  4  |  5  | 132 | 133 |
# N   3+-----+-----+-----+-----+
#      | ... | ... | ... | ... |
#    64+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128  |128+1| ... | ... |
#    65+-----+-----+-----+-----+
#      |TK*  |TK*  | ... | ... |
#      |128+2|128+3| ... | ... |
#    65+-----+-----+-----+-----+
# This data layout can be expressed by UInt32 LayoutTensor
# with shape = IntTuple(IntTuple(64, TN),IntTuple(2, TK))
# and stride = IntTuple(IntTuple(2, TK * 128),IntTuple(1, 128))
fn repack_Q4_0_for_sm8x[
    q_layout: Layout,
    repack_layout: Layout,
    scales_type: DType,
](
    q_weight: LayoutTensor[DType.uint8, q_layout],
    q_packed_weight: LayoutTensor[DType.uint8, repack_layout],
):
    alias group_size = 32
    alias group_bytes = sizeof[DType.float16]() + (group_size // 2)
    alias pack_factor = 8
    alias repack_tile = Index(64, 16)
    alias WARP_SIZE = 32
    alias BN = 128
    alias BK = 1024

    var tid: UInt = ThreadIdx.x()
    var warp_id: UInt = tid // WARP_SIZE
    alias num_warps_x = BN // repack_tile[0]
    var warp_x: UInt = warp_id % num_warps_x
    var warp_y: UInt = warp_id // num_warps_x
    var lane_id: Int = tid % WARP_SIZE
    var block_idx = Index(int(BlockIdx.x()), int(BlockIdx.y()))

    alias N = to_int(q_layout.shape[0])
    alias K = to_int(q_layout.shape[1]) // group_bytes * group_size

    alias K_groups = K // group_size
    alias BK_groups = BK // group_size

    alias uint_K = K // pack_factor
    alias uint_BK = BK // pack_factor

    @always_inline
    @parameter
    fn convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[DType.uint8, _]) -> SIMD[scales_type, 1]:
        var f32_values = bitcast[DType.float16, 1](input_bytes).cast[
            DType.float32
        ]()
        return bitcast[scales_type, 2](f32_values)[1]

    alias repacked_b_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, uint_K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * (uint_K // 2)),
            IntTuple(1, 128),
        ),
    )
    var repack_weights = LayoutTensor[DType.uint32, repacked_b_layout,](
        q_packed_weight.ptr.bitcast[Scalar[DType.uint32]](),
        RuntimeLayout[repacked_b_layout](),
    )

    alias b_scales_layout = Layout.row_major(K_groups, N)
    var b_scales_ptr = q_packed_weight.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, b_scales_layout,](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
        RuntimeLayout[b_scales_layout](),
    )

    # We keep 128x2 Q4_0 GGUF blocks in smem
    var smem = external_memory[
        Scalar[DType.uint8],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[DType.uint8, 1]](),
    ]()
    var qb_smem = LayoutTensor[
        DType.uint8,
        Layout.row_major(BN, 2 * group_bytes),
        address_space = AddressSpace.SHARED,
    ](smem.bitcast[Scalar[DType.uint8]]())

    var q_gmem_tile = q_weight.tile[BN, BK_groups * group_bytes](
        block_idx[0], block_idx[1]
    )
    var q_gmem_iter = q_gmem_tile.tiled_iterator[BN, 2 * group_bytes, axis=1](
        0, 0
    )

    var repacked_gmem_tile = repack_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var repacked_gemm_iter = repacked_gmem_tile.tiled_iterator[
        BN, 2 * group_size // pack_factor, axis=1
    ](0, 0)

    var scales_gmem_tile = repack_scales.tile[BK_groups, BN](
        block_idx[1], block_idx[0]
    )
    var scales_gmem_iter = scales_gmem_tile.tiled_iterator[2, BN, axis=0](0, 0)

    # local regs store frags of repacked 64x16 tiles
    var repack_reg_tile = tb[DType.uint32]().row_major[
        2, 4
    ]().local().alloc().fill(0)

    # We load 128x2 Q4_0 GGUF blocks to smem.
    # Each warp repacks 64x1 Q4_0 GGUF blocks, which are
    # 64x32 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # two tile for each warp.
    # repack_reg_tile[0] stores frags of the first 64x16 tile,
    # repack_reg_tile[1] stores frags of the second,
    for i in range(ceildiv(BK_groups, 2)):
        barrier()
        copy_dram_to_sram[thread_layout = Layout.row_major(128, 1),](
            qb_smem.vectorize[1, 4](),
            q_gmem_iter[]
            .bitcast[DType.uint8, address_space = AddressSpace.GENERIC]()
            .vectorize[1, 4](),
        )
        q_gmem_iter._incr()
        barrier()
        q_warp_tile = qb_smem.tile[repack_tile[0], group_bytes](warp_x, warp_y)

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var warp_mma_iter = q_warp_tile.tiled_iterator[
                8, group_bytes, axis=0
            ]()
            alias thd_mma_layout = Layout.row_major(8, 4)

            @parameter
            for i_mma_tile in range(0, 8, 2):
                var frag: SIMD[DType.uint32, 2] = 0

                # The first 2 Bytes is the scale for this Q4_0 block
                # GGUF pack elements 0-15 in the lower 4-bit of the 16 Bytes,
                # and elments 16-31 in the higher 4-bit of the 16 Bytes.
                #
                # This gets elements 0, 1, 8, 9, 16, 17, 24, 25 for
                # thread 0.
                var tmp = (
                    warp_mma_iter[]
                    .slice[:, 2:]()
                    .vectorize[1, 2]()
                    .distribute[thd_mma_layout](lane_id)
                )
                warp_mma_iter._incr()

                for ind in range(2):
                    frag[ind] |= tmp[0, 0][0].cast[DType.uint32]() & 0xF
                    frag[ind] |= (tmp[0, 0][1].cast[DType.uint32]() & 0xF) << 16
                    frag[ind] |= (tmp[0, 1][0].cast[DType.uint32]() & 0xF) << 4
                    frag[ind] |= (tmp[0, 1][1].cast[DType.uint32]() & 0xF) << 20
                    tmp[0, 0] = tmp[0, 0] >> 4
                    tmp[0, 1] = tmp[0, 1] >> 4

                # the first 2 Bytes is the scale for this Q4_0 block
                tmp = (
                    warp_mma_iter[]
                    .slice[:, 2:]()
                    .vectorize[1, 2]()
                    .distribute[thd_mma_layout](lane_id)
                )
                warp_mma_iter._incr()

                for ind in range(2):
                    frag[ind] |= (tmp[0, 0][0].cast[DType.uint32]() & 0xF) << 8
                    frag[ind] |= (tmp[0, 0][1].cast[DType.uint32]() & 0xF) << 24
                    frag[ind] |= (tmp[0, 1][0].cast[DType.uint32]() & 0xF) << 12
                    frag[ind] |= (tmp[0, 1][1].cast[DType.uint32]() & 0xF) << 28
                    tmp[0, 0] = tmp[0, 0] >> 4
                    tmp[0, 1] = tmp[0, 1] >> 4

                repack_reg_tile[0, i_mma_tile // 2] = frag[0]
                repack_reg_tile[1, i_mma_tile // 2] = frag[1]

            var repack_warp_tile = repacked_gemm_iter[].tile[
                64, group_size // pack_factor
            ](warp_x, warp_y)
            alias write_back_type = __type_of(
                repack_warp_tile.vectorize[2, 2]()[0, 0]
            )
            repack_warp_tile.vectorize[2, 2]()[lane_id, 0] = rebind[
                write_back_type
            ](repack_reg_tile.vectorize[1, 4]()[0, 0])
            repack_warp_tile.vectorize[2, 2]()[lane_id, 1] = rebind[
                write_back_type
            ](repack_reg_tile.vectorize[1, 4]()[1, 0])
            repacked_gemm_iter._incr()

            alias scales_thread_layout = Layout(IntTuple(4, 8), IntTuple(16, 1))
            var rt_scales_thread_layout = RuntimeLayout[scales_thread_layout]()

            # cast scales to bf16 before storing back
            var scales_warp_tile = scales_gmem_iter[].tile[1, 64](
                warp_y, warp_x
            )

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    rt_scales_thread_layout(lane_id), 0
                ]
            )

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    rt_scales_thread_layout(lane_id) + 8, 0
                ]
            )

            scales_gmem_iter._incr()


# Tensors of GPTQ format are stored in a non-transposed way.
# Assume the original transposed matrix is of shape [N, K], the qweight shard
# will be a uint32 matrix of shape [K // 8, N], and scales will be of shape
# [K_groups, N]. The input is a uint8 tensor of shape
# [K_groups * group_bytes, N].
fn repack_GPTQ_for_sm8x[
    in_layout: Layout,
    out_layout: Layout,
    scales_type: DType,
    group_size: Int,
](
    in_tensor: LayoutTensor[DType.uint8, in_layout],
    out_tensor: LayoutTensor[DType.uint8, out_layout],
):
    alias raw_scales_type = DType.float16
    alias weights_bytes_per_group = group_size // 2
    alias group_bytes = sizeof[DType.float16]() + weights_bytes_per_group
    alias pack_factor = 8
    alias repack_tile = Index(64, 16)
    alias BN = 128
    alias BK = 1024

    var tid: UInt = ThreadIdx.x()
    var warp_id: UInt = tid // WARP_SIZE
    alias num_warps_x = BN // repack_tile[0]
    var warp_x: UInt = warp_id % num_warps_x
    var warp_y: UInt = warp_id // num_warps_x
    var lane_id: Int = tid % WARP_SIZE
    var block_idx = Index(int(BlockIdx.x()), int(BlockIdx.y()))

    alias N = to_int(in_layout.shape[1])
    alias K = to_int(in_layout.shape[0]) // group_bytes * group_size

    alias K_groups = K // group_size
    alias BK_groups = BK // group_size

    alias uint_K = K // pack_factor
    alias uint_BK = BK // pack_factor

    @always_inline
    @parameter
    fn convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[raw_scales_type, _]) -> SIMD[scales_type, 1]:
        var f32_values = bitcast[DType.float16, 1](input_bytes).cast[
            DType.float32
        ]()
        return bitcast[scales_type, 2](f32_values)[1]

    # Define 4-bit weights and scales for the raw input
    alias raw_weights_layout = Layout.row_major(uint_K, N)
    var raw_weights = LayoutTensor[DType.uint32, raw_weights_layout,](
        in_tensor.ptr.bitcast[Scalar[DType.uint32]](),
        RuntimeLayout[raw_weights_layout](),
    ).transpose()
    alias raw_scales_layout = Layout.row_major(K_groups, N)
    var raw_scales_ptr = in_tensor.ptr + N * K // 2
    var raw_scales = LayoutTensor[raw_scales_type, raw_scales_layout,](
        raw_scales_ptr.bitcast[Scalar[raw_scales_type]](),
        RuntimeLayout[raw_scales_layout](),
    ).transpose()

    # Define 4-bit weights and scales for the repacked buffer
    alias repacked_weights_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, uint_K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * (uint_K // 2)),
            IntTuple(1, 128),
        ),
    )
    var repack_weights = LayoutTensor[DType.uint32, repacked_weights_layout,](
        out_tensor.ptr.bitcast[Scalar[DType.uint32]](),
        RuntimeLayout[repacked_weights_layout](),
    )
    alias repacked_scales_layout = Layout.row_major(K_groups, N)
    var repacked_scales_ptr = out_tensor.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, repacked_scales_layout,](
        repacked_scales_ptr.bitcast[Scalar[scales_type]](),
        RuntimeLayout[repacked_scales_layout](),
    )

    # We keep 128x2 GPTQ blocks in smem
    var smem = external_memory[
        Scalar[DType.uint8],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[DType.uint8, 1]](),
    ]()
    var weights_smem = LayoutTensor[
        DType.uint8,
        Layout.row_major(BN, 2 * weights_bytes_per_group),
        address_space = AddressSpace.SHARED,
    ](smem.bitcast[Scalar[DType.uint8]]())
    var weights_smem_uint4 = LayoutTensor[
        DType.uint32,
        Layout.row_major(BN, 2 * group_size // pack_factor),
        address_space = AddressSpace.SHARED,
    ](smem.bitcast[Scalar[DType.uint32]]())

    var raw_weights_gmem_tile = raw_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var raw_weights_gmem_iter = raw_weights_gmem_tile.tiled_iterator[
        BN, 2 * weights_bytes_per_group // sizeof[DType.uint32](), axis=1
    ](0, 0)
    var raw_scales_gmem_tile = raw_scales.tile[BN, BK_groups](
        block_idx[0], block_idx[1]
    )
    var raw_scales_gmem_iter = raw_scales_gmem_tile.tiled_iterator[
        BN, 2, axis=1
    ](0, 0)

    var repacked_weights_gmem_tile = repack_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var repacked_weights_gmem_iter = repacked_weights_gmem_tile.tiled_iterator[
        BN, 2 * group_size // pack_factor, axis=1
    ](0, 0)

    var repacked_scales_gmem_tile = repack_scales.tile[BK_groups, BN](
        block_idx[1], block_idx[0]
    )
    var repacked_scales_gmem_iter = repacked_scales_gmem_tile.tiled_iterator[
        2, BN, axis=0
    ](0, 0)

    # local regs store frags of repacked 64x16 tiles
    var repack_reg_tile = tb[DType.uint32]().row_major[
        1, 4
    ]().local().alloc().fill(0)

    # We load 128x2 GPTQ blocks to smem.
    # Each warp repacks 64x1 GPTQ blocks, which are
    # 64xgroup_size 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # (group_size // 16) tiles for each warp.
    # repack_reg_tile[0] stores frags of the one 64x16 tile,
    for i in range(ceildiv(BK_groups, 2)):
        barrier()
        copy_dram_to_sram[thread_layout = Layout.row_major(128, 1),](
            weights_smem_uint4.vectorize[1, 1](),
            raw_weights_gmem_iter[].vectorize[1, 1](),
        )
        raw_weights_gmem_iter._incr()
        barrier()
        raw_weights_warp_tile = weights_smem.tile[
            repack_tile[0], weights_bytes_per_group
        ](warp_x, warp_y)

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var repacked_warp_tile = repacked_weights_gmem_iter[].tile[
                repack_tile[0], group_size // pack_factor
            ](warp_x, warp_y)

            @parameter
            for i_marlin_tile in range(group_size // repack_tile[1]):
                var warp_mma_iter = raw_weights_warp_tile.tiled_iterator[
                    8, repack_tile[1] // 2, axis=0
                ](0, i_marlin_tile)
                alias thd_mma_layout = Layout.row_major(8, 4)

                @parameter
                for i_mma_tile in range(0, 8, 2):
                    var frag: SIMD[DType.uint32, 1] = 0

                    # This gets elements 0, 1, 8, 9 for
                    # thread 0.
                    var tmp = (
                        warp_mma_iter[].distribute[thd_mma_layout](lane_id)
                    )
                    warp_mma_iter._incr()
                    frag[0] |= tmp[0, 0][0].cast[DType.uint32]() & 0x0F
                    frag[0] |= (tmp[0, 0][0].cast[DType.uint32]() & 0xF0) << 12
                    frag[0] |= (tmp[0, 1][0].cast[DType.uint32]() & 0x0F) << 4
                    frag[0] |= (tmp[0, 1][0].cast[DType.uint32]() & 0xF0) << 16

                    tmp = warp_mma_iter[].distribute[thd_mma_layout](lane_id)
                    warp_mma_iter._incr()
                    frag[0] |= (tmp[0, 0][0].cast[DType.uint32]() & 0x0F) << 8
                    frag[0] |= (tmp[0, 0][0].cast[DType.uint32]() & 0xF0) << 20
                    frag[0] |= (tmp[0, 1][0].cast[DType.uint32]() & 0x0F) << 12
                    frag[0] |= (tmp[0, 1][0].cast[DType.uint32]() & 0xF0) << 24

                    repack_reg_tile[0, i_mma_tile // 2] = frag[0]

                var repacked_marlin_tile = repacked_warp_tile.tile[
                    repack_tile[0], repack_tile[1] // pack_factor
                ](0, i_marlin_tile)
                alias write_back_type = __type_of(
                    repacked_marlin_tile.vectorize[2, 2]()[0, 0]
                )
                repacked_marlin_tile.vectorize[2, 2]()[lane_id, 0] = rebind[
                    write_back_type
                ](repack_reg_tile.vectorize[1, 4]()[0, 0])

            repacked_weights_gmem_iter._incr()

            alias scales_thread_layout = Layout(IntTuple(4, 8), IntTuple(16, 1))
            var rt_scales_thread_layout = RuntimeLayout[scales_thread_layout]()

            # cast scales to bf16 before storing back
            var scales_warp_tile = repacked_scales_gmem_iter[].tile[1, 64](
                warp_y, warp_x
            )
            var raw_scales_warp_tile = raw_scales_gmem_iter[].tile[64, 1](
                warp_x, warp_y
            )

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](raw_scales_warp_tile[rt_scales_thread_layout(lane_id), 0])

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](raw_scales_warp_tile[rt_scales_thread_layout(lane_id) + 8, 0])

            repacked_scales_gmem_iter._incr()
            raw_scales_gmem_iter._incr()


@always_inline
fn matmul_gpu_qint4[
    c_type: DType,
    a_type: DType,
    b_shape: DimList, //,
    group_size: Int,
    target: StringLiteral,
](
    c: NDBuffer[c_type, 2, _],
    a: NDBuffer[a_type, 2, _],
    b: NDBuffer[DType.uint8, 2, b_shape],
    ctx: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    constrained["cuda" in target, "unsupported target"]()
    var cuda_ctx = ctx.get_device_context()

    alias pack_factor = 8

    alias a_shape = a.shape
    alias c_shape = c.shape
    var shape = GemmShape.get[transpose_b=True](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    var tensor_c = from_ndbuffer_row_major(
        rebind[NDBuffer[c_type, 2, c_shape]](c)
    )
    var tensor_a = from_ndbuffer_row_major(
        rebind[NDBuffer[a_type, 2, a_shape]](a)
    )
    var tensor_b = from_ndbuffer_row_major(
        rebind[NDBuffer[DType.uint8, 2, b_shape]](b)
    )

    alias kernels = MatmulKernels[a_type, DType.uint8, c_type, True]()
    alias config = kernels.ampere_128x128_4

    # estimate smem usage
    var smem_usage: Int = 0
    var block_mnk = config.block_tile_shape
    var num_pipeline_stages = config.num_pipeline_stages
    var a_usage = block_mnk[0] * block_mnk[2] * num_pipeline_stages * sizeof[
        a_type
    ]()
    var b_usage = block_mnk[1] * block_mnk[2] * num_pipeline_stages * sizeof[
        DType.uint32
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
        c_type,  # c_type
        tensor_c.layout,
        a_type,  # a_type
        tensor_a.layout,
        DType.uint8,  # b_type
        tensor_b.layout,
        group_size,
        pack_factor,
        True,
        config,
    ]

    var func = cuda_ctx.compile_function[gemm,](
        threads_per_block=int(config.num_threads()),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]

    cuda_ctx.enqueue_function(
        func,
        tensor_c,
        tensor_a,
        tensor_b,
        grid_dim=(ceildiv(n, BN), ceildiv(m, BM), 1),
        block_dim=(int(config.num_threads()), 1, 1),
        shared_mem_bytes=smem_usage,
    )


@always_inline
fn gpu_qint4_repack_Q4_0[
    b_shape: DimList, //,
    target: StringLiteral,
](
    b: NDBuffer[DType.uint8, 2, b_shape],
    b_packed: NDBuffer[DType.uint8, 2, b_shape],
    ctx: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    constrained["cuda" in target, "unsupported target"]()
    var cuda_ctx = ctx.get_device_context()

    alias pack_factor = 8
    alias group_size = 32
    alias group_bytes = 2 + (group_size // 2)
    alias BN = 128
    alias BK = 1024

    alias N = b.shape.get[0]()
    alias K = b.shape.get[1]() // group_bytes * group_size

    var tensor_b = from_ndbuffer_row_major(
        rebind[NDBuffer[DType.uint8, 2, b_shape]](b)
    )
    var tensor_packed_b = from_ndbuffer_row_major(
        rebind[NDBuffer[DType.uint8, 2, b_shape]](b_packed)
    )

    var smem_usage: Int = BN * 2 * group_bytes

    alias repack = repack_Q4_0_for_sm8x[
        tensor_b.layout,
        tensor_packed_b.layout,
        DType.bfloat16,
    ]

    var repack_func = cuda_ctx.compile_function[repack,](
        threads_per_block=int(128),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    cuda_ctx.enqueue_function(
        repack_func,
        tensor_b,
        tensor_packed_b,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
    )


@always_inline
fn gpu_qint4_repack_GPTQ[
    b_shape: DimList,
    b_packed_shape: DimList, //,
    group_size: Int,
    target: StringLiteral,
](
    b: NDBuffer[DType.uint8, 2, b_shape],
    b_packed: NDBuffer[DType.uint8, 2, b_packed_shape],
    ctx: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    constrained["cuda" in target, "unsupported target"]()
    var cuda_ctx = ctx.get_device_context()

    alias pack_factor = 8
    alias group_bytes = 2 + (group_size // 2)
    alias BN = 128
    alias BK = 1024

    alias N = b.shape.get[1]()
    alias K = b.shape.get[0]() // group_bytes * group_size

    constrained[
        N == b_packed.shape.get[0](),
        "qmatmul: Mismatched input/output dimension.",
    ]()
    constrained[
        K == (b_packed.shape.get[1]() // group_bytes * group_size),
        "qmatmul: Mismatched input/output dimension.",
    ]()

    var tensor_b = from_ndbuffer_row_major(
        rebind[NDBuffer[DType.uint8, 2, b_shape]](b)
    )
    var tensor_packed_b = from_ndbuffer_row_major(
        rebind[NDBuffer[DType.uint8, 2, b_packed_shape]](b_packed)
    )

    var smem_usage: Int = BN * 2 * group_bytes

    alias repack = repack_GPTQ_for_sm8x[
        tensor_b.layout,
        tensor_packed_b.layout,
        DType.bfloat16,
        group_size,
    ]

    var repack_func = cuda_ctx.compile_function[repack,](
        threads_per_block=int(128),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    cuda_ctx.enqueue_function(
        repack_func,
        tensor_b,
        tensor_packed_b,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
    )
