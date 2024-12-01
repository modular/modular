# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

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
from internal_utils._utils import ValOrDim, dynamic, static
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
from linalg.utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    block_swizzle,
    select_config,
)
from memory import UnsafePointer
from memory.unsafe import bitcast
from quantization import Q4sym

from utils.index import Index, IndexList


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


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
    mut b_smem_iter: LayoutTensorIter[
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

    var tid: UInt32 = ThreadIdx.x
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
    # work around mut argument can't have default value.
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

    var tid: UInt32 = ThreadIdx.x
    var warp_id = tid // WARP_SIZE

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle(
        (int(BlockIdx.x), int(BlockIdx.y)),
        (int(GridDim.x), int(GridDim.y)),
    ) if swizzle_block else Index(int(BlockIdx.x), int(BlockIdx.y))

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

    var tid: UInt = ThreadIdx.x
    var warp_id: UInt = tid // WARP_SIZE
    alias num_warps_x = BN // repack_tile[0]
    var warp_x: UInt = warp_id % num_warps_x
    var warp_y: UInt = warp_id // num_warps_x
    var lane_id: Int = tid % WARP_SIZE
    var block_idx = Index(int(BlockIdx.x), int(BlockIdx.y))

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

            # The repack_warp_tile is of shape [64, (2, 2)]. In this case,
            # elements [0, 0], [0, 1], [1, 0] and [1, 1] are stored continously
            # in the memory. We need to use a element shape of [2, 2] to
            # correctly vectorize this tenosr.
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

            alias scales_thread_layout = Layout(
                IntTuple(4, 8),
                IntTuple(16, 1),
            )
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


# this kernel dequantizes a repacked INT4 matrix into bf16 format
# Assuming a 64x16 (nxk) packing scheme
# Tile [i, j] stores part of the original matrix [i*64:(i+1)*64, j*16:(j+1)*16]
# Within each tile, weights are repacked similarly to the Marlin kernel.
# The memory address for tile [i, j] is (i * (N//64) + j) * tile_size,
# where tile_size is 64 * 16 * 4 / pack_factor = 512 Bytes.
fn create_ref_b[
    type_q: DType,
    type_b: DType,
    b_q_layout: Layout,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
](
    b_packed: LayoutTensor[type_q, b_q_layout],
    b_out: LayoutTensor[type_b, b_layout],
):
    alias WARP_SIZE = 32
    alias BLOCK_N = 128
    alias BLOCK_K = 32
    alias repack_tile = Index(64, 16)
    alias TILE_N = 64
    alias TILE_K = 16
    alias num_k_warps = BLOCK_K // repack_tile[1]

    var tid: UInt = ThreadIdx.x
    var warp_id: UInt = tid // WARP_SIZE
    var lane_id: UInt = tid % WARP_SIZE
    var block_idx = Index(int(BlockIdx.x), int(BlockIdx.y))
    var warp_x: UInt = warp_id // num_k_warps
    var warp_y: UInt = warp_id % num_k_warps

    alias group_bytes = group_size // 2 + 2
    alias N = to_int(b_q_layout.shape[0])
    alias K = to_int(b_q_layout.shape[1]) // group_bytes * group_size

    # Unpack quantized weights
    alias scales_type = DType.bfloat16
    alias b_type = DType.uint32
    alias b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b_q = LayoutTensor[b_type, b_weight_layout,](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    alias b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout,](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

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


fn random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return (
        random_float64(min=min, max=max)
        .cast[DType.float32]()
        .cast[DType.float16]()
    )


struct _block_Q4_0:
    alias group_size = 32

    var base_scale: Float16
    var q_bits: InlineArray[UInt8, Self.group_size // 2]


fn test_repack_Q4_0_for_sm8x(
    ctx: DeviceContext, n: ValOrDim, k: ValOrDim
) raises:
    print("test repack_Q4_0_for_sm8x")

    fn fill_random[type: DType](array: InlineArray[Scalar[type]]):
        rand(array.unsafe_ptr(), len(array), min=0, max=255)

    fn build_b_buffer(N: Int, K: Int, b_ptr: UnsafePointer[UInt8]):
        var k_groups = ceildiv(K, 32)
        var block_ptr = b_ptr.bitcast[_block_Q4_0]()

        for n in range(N):
            for k in range(k_groups):
                block_ptr[].base_scale = random_float16()
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

    alias group_size = 32
    alias pack_factor = 8
    var N = n.value
    var K = k.value
    alias BN = 128
    alias BK = 1024
    alias group_bytes = 2 + (group_size // 2)

    alias static_gguf_b_shape = DimList(
        n.dim, (k.dim // group_size) * group_bytes
    )
    alias static_repacked_b_shape = DimList(
        n.dim, (k.dim // group_size) * group_bytes
    )
    alias static_dequan_shape = DimList(k.dim, n.dim)

    var dynamic_gguf_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_repacked_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_dequan_shape = DimList(k.value, n.value)

    var gguf_b_host = HostNDBuffer[DType.uint8, 2](dynamic_gguf_b_shape)
    var repacked_b_host = HostNDBuffer[DType.uint8, 2, static_repacked_b_shape](
        dynamic_repacked_b_shape
    )
    var gguf_dequan_ref_host = HostNDBuffer[DType.bfloat16, 2](
        dynamic_dequan_shape
    )
    var repacked_dequan_host = HostNDBuffer[DType.bfloat16, 2](
        dynamic_dequan_shape
    )

    zero(repacked_b_host.tensor)
    build_b_buffer(N, K, gguf_b_host.tensor.data)
    Q4sym[group_size, DType.bfloat16].dequantize_and_write_to_tensor(
        gguf_b_host.tensor,
        gguf_dequan_ref_host.tensor,
        gguf_dequan_ref_host.tensor.get_shape(),
    )
    zero(repacked_dequan_host.tensor)

    var gguf_b_device = DeviceNDBuffer[DType.uint8, 2, static_gguf_b_shape](
        dynamic_gguf_b_shape, ctx=ctx
    )
    var repacked_b_device = DeviceNDBuffer[
        DType.uint8, 2, static_repacked_b_shape
    ](dynamic_repacked_b_shape, ctx=ctx)
    var repacked_dequan_device = DeviceNDBuffer[
        DType.bfloat16, 2, static_dequan_shape
    ](dynamic_dequan_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(gguf_b_device.buffer, gguf_b_host.tensor.data)
    ctx.enqueue_copy_to_device(
        repacked_b_device.buffer, repacked_b_host.tensor.data
    )

    alias gguf_b_layout = Layout.row_major[gguf_b_device.rank](
        gguf_b_device.shape
    )
    alias repacked_b_layout = Layout.row_major[repacked_b_device.rank](
        repacked_b_device.shape
    )
    alias repack_dequan_layout = Layout.row_major[repacked_dequan_device.rank](
        repacked_dequan_device.shape
    )
    alias repacked_b_old_layout = Layout.row_major(
        int(n.dim) // 64,
        int(k.dim) * 64 // pack_factor,
    )

    var gguf_b_tensor = LayoutTensor[DType.uint8, gguf_b_layout,](
        gguf_b_device.buffer.ptr,
        RuntimeLayout[gguf_b_layout].row_major(
            gguf_b_device.tensor.dynamic_shape
        ),
    )
    var repacked_b_tensor = LayoutTensor[DType.uint8, repacked_b_layout,](
        repacked_b_device.buffer.ptr,
        RuntimeLayout[repacked_b_layout](),
    )
    var repacked_dequan_tensor = LayoutTensor[
        DType.bfloat16,
        repack_dequan_layout,
    ](
        repacked_dequan_device.buffer.ptr,
        RuntimeLayout[repack_dequan_layout].row_major(
            repacked_dequan_device.tensor.dynamic_shape
        ),
    )

    var smem_usage: Int = BN * 2 * group_bytes

    alias repack = repack_Q4_0_for_sm8x[
        gguf_b_tensor.layout,
        repacked_b_tensor.layout,
        DType.bfloat16,
    ]

    var repack_func = ctx.compile_function[repack,](
        threads_per_block=int(128),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    alias dequan = create_ref_b[
        DType.uint8,
        DType.bfloat16,
        repacked_b_tensor.layout,
        repacked_dequan_tensor.layout,
        group_size,
        pack_factor,
    ]

    var dequan_func = ctx.compile_function[dequan,](
        threads_per_block=int(128),
    )

    ctx.enqueue_function(
        repack_func,
        gguf_b_tensor,
        repacked_b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
    )

    ctx.enqueue_function(
        dequan_func,
        repacked_b_tensor,
        repacked_dequan_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
    )

    ctx.enqueue_copy_from_device(
        repacked_b_host.tensor.data, repacked_b_device.buffer
    )
    ctx.enqueue_copy_from_device(
        repacked_dequan_host.tensor.data, repacked_dequan_device.buffer
    )

    ctx.synchronize()

    alias rtol = 2e-2
    assert_almost_equal(
        gguf_dequan_ref_host.tensor,
        repacked_dequan_host.tensor,
        atol=0.0001,
        rtol=rtol,
    )


fn test_quantized[
    type: DType
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    # quantization configs
    alias group_size = 128
    alias has_zero_point = False
    alias pack_factor = 8
    alias group_bytes = group_size // 2 + 2

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
    alias static_b_shape = DimList(n.dim, (k.dim // group_size) * group_bytes)
    alias static_b_ref_shape = DimList(n.dim, k.dim)
    alias static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_b_ref_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)

    zero(c_host.tensor)
    random(a_host.tensor)

    var b_scales_ptr = (b_host.tensor.data + N * K // 2).bitcast[
        Scalar[a_type]
    ]()
    var b_scales_view = NDBuffer[
        a_type, 2, DimList(k.dim // group_size, n.dim)
    ](b_scales_ptr)
    # elements of b matrix is between [-1, 1]
    random(b_scales_view, 0, 0.125)
    randint(
        b_host.tensor.data.bitcast[Scalar[DType.uint32]](),
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
    var b_device_ref = DeviceNDBuffer[a_type, 2, static_b_ref_shape](
        dynamic_b_ref_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[a_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    alias c_layout = Layout.row_major[c_device.rank](c_device.shape)
    alias a_layout = Layout.row_major[c_device.rank](a_device.shape)
    alias b_layout = Layout.row_major[c_device.rank](b_device.shape)
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
        b_tensor.layout,
        b_ref_tensor.layout,
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
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(int(config.num_threads()), 1, 1),
        shared_mem_bytes=smem_usage,  # config.shared_mem_usage(),
    )

    ctx.enqueue_function(
        func_dequan,
        b_tensor,
        b_ref_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
    )

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)

    alias kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    alias config_ref = kernels_ref.ampere_128x128_4
    _matmul_gpu[use_tensor_core=True, transpose_b=True, config=config_ref](
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
    _ = a_device
    _ = b_device

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor

    _ = func^
    _ = func_dequan^


def main():
    with DeviceContext() as ctx:
        test_repack_Q4_0_for_sm8x(
            ctx,
            static[4096](),
            static[4096](),
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[6144](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[4096](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[28672](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[4096](), static[14336]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[128256](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[6144](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[4096](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[28672](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[4096](), static[14336]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[128256](), static[4096]()
        )
