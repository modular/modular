# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceildiv
from sys import alignof, simdwidthof, sizeof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import (
    WARP_SIZE,
    BlockIdx,
    GridDim,
    ThreadIdx,
    barrier,
    lane_id,
    warp_broadcast,
)
from gpu.host import Context, FuncAttribute, Function, Stream, synchronize
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
    Fill,
    CacheEviction,
)
from gpu.mma import ld_matrix, mma
from layout.int_tuple import UNKNOWN_VALUE, IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_local_to_sram,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace as AddressSpace

from utils.index import Index, IndexList

from .matmul_gpu import matmul_kernel_naive
from .utils import apply_epilogue, elementwise_epilogue_type
from .utils_gpu import MatmulConfig, MatmulKernels, block_swizzle


@always_inline
fn distance[
    type: DType, //
](arg0: UnsafePointer[Scalar[type]], arg1: UnsafePointer[Scalar[type]]) -> Int:
    return (int(arg0) - int(arg1)) // sizeof[arg1.type]()


@always_inline
fn multistage_mma[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    a_smem_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    b_smem_layout: Layout, //,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    transpose_b: Bool,
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
    k_group_size: UInt = 1,
](
    c: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL, **_],
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
    var warp_id = warp_broadcast(tid // WARP_SIZE)

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

    @always_inline
    @parameter
    fn _copy_dram_to_sram_async_a[
        tile_layout: Layout, //,
        *,
        masked: Bool = False,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        a_tile: LayoutTensor[
            a_type,
            tile_layout,
            address_space = AddressSpace.SHARED,
            element_layout=_, **_,
        ],
        num_rows: Int = UNKNOWN_VALUE,
    ):
        copy_dram_to_sram_async[
            thread_layout=async_copy_a_layout,
            swizzle=swizzle_a,
            masked=masked,
            fill=fill,
            eviction_policy=eviction_policy,
        ](
            a_tile.vectorize[1, simd_size](),
            a_iter[].bitcast[a_type]().vectorize[1, simd_size](),
            num_rows,
        )

    @always_inline
    @parameter
    fn _copy_dram_to_sram_async_b[
        tile_layout: Layout, //,
        *,
        masked: Bool = False,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        b_tile: LayoutTensor[
            b_type,
            tile_layout,
            address_space = AddressSpace.SHARED,
            element_layout=_, **_,
        ],
        num_rows: Int = UNKNOWN_VALUE,
    ):
        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout,
            swizzle=swizzle_b,
            masked=masked,
            fill=fill,
            eviction_policy=eviction_policy,
        ](
            b_tile.vectorize[1, simd_size](),
            b_iter[].bitcast[b_type]().vectorize[1, simd_size](),
            num_rows,
        )

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    if prefetch_init:

        @parameter
        for stage in range(num_pipeline_stages - 1):

            @parameter
            if a_iter.address_space == AddressSpace.GENERIC:
                var a_smem_tile = a_smem_iter.next_unsafe(stage)[]

                if num_a_rows:
                    _copy_dram_to_sram_async_a[masked=True](
                        a_smem_tile, num_a_rows.value()
                    )
                else:
                    _copy_dram_to_sram_async_a(a_smem_tile)

                a_iter._incr()

            @parameter
            if b_iter.address_space == AddressSpace.GENERIC:
                var b_smem_tile = b_smem_iter.next_unsafe(stage)[]

                if num_b_rows:
                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                        0, num_b_rows.value() - stage * BK
                    )
                    _copy_dram_to_sram_async_b[masked=True](
                        b_smem_tile, num_rows_bound
                    )
                else:
                    _copy_dram_to_sram_async_b(b_smem_tile)

                b_iter._incr()

            async_copy_commit_group()

        # Guard stage 0.
        async_copy_wait_group(num_pipeline_stages - 2)
        barrier()

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas: UInt = BK // MMA_K
    alias num_k_mma_iters: UInt = num_k_mmas // k_group_size
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    constrained[
        num_k_mmas % (2 * k_group_size) == 0,
        "num_k_mmas must be an integer multiple of 2*k_group_size",
    ]()

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    alias num_reg_tiles = 2 * k_group_size
    # Register tiles.
    var a_reg_tiles = tb[a_type]().row_major[
        2 * k_group_size * num_m_mmas, a_frag_size
    ]().local().alloc().split[2 * k_group_size]()

    var b_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)

    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    @parameter
    for i in range(int(k_group_size)):

        @parameter
        if a_iter.address_space == AddressSpace.LOCAL:
            # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
            # Need to cast address space because it's not known at parse time to be LOCAL.
            copy_local_to_local(a_reg_tiles[i], a_iter[])
            a_iter._incr()
        else:
            mma_op.load_a[swizzle_a](
                a_warp_tile, a_reg_tiles[i].vectorize[1, a_frag_size](), i
            )

        mma_op.load_b(b_warp_tile, b_reg_tiles[i], i, int(warp_x))

    @parameter
    if a_iter.address_space == AddressSpace.LOCAL:
        constrained[
            static_num_iters.has_value(),
            "Using input in registers requires static iteration bound.\n",
        ]()

        @parameter
        for k_tile_id in range(static_num_iters.get()):
            var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                b_wtile_coord0,
                b_wtile_coord1,
            )

            # Perform prefetch registers and mma until current shared memory tile's
            # data has all been loaded to registers.
            @parameter
            for k_mma0 in range(int(num_k_mma_iters)):

                @parameter
                for k_mma1 in range(int(k_group_size)):
                    alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    alias current = k_mma % num_reg_tiles
                    alias k_mma_next = k_mma + k_group_size
                    alias next = int(k_mma_next % num_reg_tiles)

                    @parameter
                    if k_mma_next == num_k_mmas:
                        alias prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                        # Prefetch one k tile (if valid) from global memory to current
                        # shared memory buffer.
                        @parameter
                        if prefetch_tile_id < static_num_iters.get():

                            @parameter
                            if b_iter.address_space == AddressSpace.GENERIC:
                                var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                    num_pipeline_stages - 1
                                )[]

                                if num_b_rows:
                                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                        0,
                                        num_b_rows.value()
                                        - prefetch_tile_id * BK,
                                    )
                                    _copy_dram_to_sram_async_b[masked=True](
                                        b_smem_prefetch_tile, num_rows_bound
                                    )
                                else:
                                    _copy_dram_to_sram_async_b(
                                        b_smem_prefetch_tile
                                    )

                                b_iter._incr()

                        async_copy_commit_group()

                        # Guard the next k tile's shared memory buffer.
                        async_copy_wait_group(num_pipeline_stages - 2)
                        barrier()

                        b_smem_iter._incr()

                        a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
                        b_warp_tile = b_smem_iter[].tile[
                            b_wtile_dim0, b_wtile_dim1
                        ](b_wtile_coord0, b_wtile_coord1)

                    # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
                    copy_local_to_local(a_reg_tiles[int(next)], a_iter[])
                    a_iter._incr()

                    alias kidx = k_mma_next % num_k_mmas
                    mma_op.load_b(
                        b_warp_tile,
                        b_reg_tiles[int(next)],
                        int(kidx),
                        int(warp_x),
                    )

                @parameter
                for k_mma1 in range(int(k_group_size)):
                    alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                    alias current = k_mma % num_reg_tiles
                    mma_op.mma(
                        a_reg_tiles[int(current)].vectorize[1, a_frag_size](),
                        b_reg_tiles[int(current)],
                        c.vectorize[1, c_frag_size](),
                    )

        return

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma0 in range(int(num_k_mma_iters)):

            @parameter
            for k_mma1 in range(int(k_group_size)):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                alias k_mma_next = k_mma + k_group_size
                alias next = int(k_mma_next % num_reg_tiles)

                @parameter
                if k_mma_next == num_k_mmas:
                    var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                    # Prefetch one k tile (if valid) from global memory to current
                    # shared memory buffer.
                    if prefetch_tile_id < num_iters:

                        @parameter
                        if a_iter.address_space == AddressSpace.GENERIC:
                            var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]

                            if num_a_rows:
                                _copy_dram_to_sram_async_a[masked=True](
                                    a_smem_prefetch_tile, num_a_rows.value()
                                )
                            else:
                                _copy_dram_to_sram_async_a(a_smem_prefetch_tile)

                            a_iter._incr()

                        @parameter
                        if b_iter.address_space == AddressSpace.GENERIC:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[]

                            if num_b_rows:
                                var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                    0,
                                    num_b_rows.value() - prefetch_tile_id * BK,
                                )
                                _copy_dram_to_sram_async_b[masked=True](
                                    b_smem_prefetch_tile, num_rows_bound
                                )
                            else:
                                _copy_dram_to_sram_async_b(b_smem_prefetch_tile)

                            b_iter._incr()
                    else:

                        @parameter
                        if continue_prefetch_b:
                            var b_smem_prefetch_tile = b_smem_iter.next_unsafe(
                                num_pipeline_stages - 1
                            )[].reshape[b_next_smem_layout]()

                            alias row_size = b_next_smem_layout.stride[
                                0
                            ].value()

                            if not num_b_rows:
                                copy_dram_to_sram_async[
                                    thread_layout = Layout.row_major(
                                        num_threads * simd_size // row_size,
                                        row_size // simd_size,
                                    ),
                                    swizzle = transpose_b_next
                                    or b_type.is_half_float(),
                                ](
                                    b_smem_prefetch_tile.vectorize[
                                        1, simd_size
                                    ](),
                                    next_b_iter[]
                                    .bitcast[b_type]()
                                    .vectorize[1, simd_size](),
                                )

                            else:
                                # TODO: can we guard at compile time num_b_rows is set here?
                                var num_rows_bound = num_b_rows.value() if transpose_b_next else max(
                                    0,
                                    num_b_rows.value()
                                    - (prefetch_tile_id - num_iters) * BK,
                                )

                                copy_dram_to_sram_async[
                                    thread_layout = Layout.row_major(
                                        num_threads * simd_size // row_size,
                                        row_size // simd_size,
                                    ),
                                    swizzle = transpose_b_next
                                    or b_type.is_half_float(),
                                    masked=True,
                                ](
                                    b_smem_prefetch_tile.vectorize[
                                        1, simd_size
                                    ](),
                                    next_b_iter[]
                                    .bitcast[b_type]()
                                    .vectorize[1, simd_size](),
                                    num_rows_bound,
                                )

                            next_b_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(num_pipeline_stages - 2)
                    barrier()

                    a_smem_iter._incr()
                    b_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
                    b_warp_tile = b_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                alias kidx = int(k_mma_next % num_k_mmas)
                mma_op.load_a[swizzle_a](
                    a_warp_tile,
                    a_reg_tiles[next].vectorize[1, a_frag_size](),
                    kidx,
                )

                mma_op.load_b(
                    b_warp_tile,
                    b_reg_tiles[next],
                    kidx,
                    int(warp_x),
                )

            @parameter
            for k_mma1 in range(int(k_group_size)):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                mma_op.mma(
                    a_reg_tiles[int(current)].vectorize[1, a_frag_size](),
                    b_reg_tiles[int(current)],
                    c.vectorize[1, c_frag_size](),
                )


fn multistage_gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
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

    var tid = ThreadIdx.x()
    var ln_id = lane_id()
    var warp_id = warp_broadcast(tid // WARP_SIZE)

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx = block_swizzle(
        Index[element_bitwidth=32, unsigned=True](BlockIdx.x(), BlockIdx.y()),
        Index[element_bitwidth=32, unsigned=True](GridDim.x(), GridDim.y()),
    ) if swizzle_block else Index[element_bitwidth=32, unsigned=True](
        BlockIdx.x(), BlockIdx.y()
    )

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(warp_id, num_warps_n)

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
        address_space = a_smem.address_space,
        alignment = a_smem.alignment,
        circular=True,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    a_type,
                    Layout.row_major(BM, BK),
                    address_space = a_smem.address_space,
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

    # create input layout tensors A and Bv
    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = (block_idx[0], 0) if transpose_b else (0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    # Compute MMA config
    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    var num_rows = min(BM, int(M) - block_idx[1] * BM)

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias c_frag_size = frag_size[2]
    var c_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)

    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
        transpose_b,
        k_group_size = config.k_group_size,
    ](
        c_reg_tile,
        a_gmem_iter,
        b_gmem_iter,
        a_smem_iter,
        b_smem_iter,
        ceildiv(K, BK),
        num_a_rows=num_rows,
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
        ]().shared().view(a_smem.bitcast[c_type]() + warp_id * WM * WN)

        copy_local_to_sram[
            thread_layout = Layout.row_major(8, 4),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            alias warp_layout = Layout.row_major(
                WARP_SIZE * simd_size // WN, WN // simd_size
            )
            var c_gmem_frag = c_gmem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](ThreadIdx.x())
            var c_smem_frag = accum_smem_warp_tile.vectorize[
                1, simd_size
            ]().distribute[warp_layout](ThreadIdx.x())
            var thread_offset = c_gmem_frag.distance(c.ptr)
            alias num_stores_per_thread = __type_of(c_gmem_frag).layout.size()

            var c_smem_frag_offset = c_smem_frag.distance(
                accum_smem_warp_tile.ptr
            )

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = __type_of(c_smem_frag).layout(i)
                alias src_idx_base = src_idx % swizzle.size()
                alias src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle(
                    c_smem_frag_offset + src_idx_base
                ) + src_idx_diff

                alias dst_static_idx = __type_of(c_gmem_frag).layout(i)
                var dst_idx = 0

                @parameter
                if c_layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = c_gmem_frag.runtime_layout(i)

                var m = int((thread_offset + dst_idx) // N)
                var n = int((thread_offset + dst_idx) % N)
                alias alignment = alignof[SIMD[c_type, simd_size]]()
                if m < M and n < N:
                    epilogue[alignment=alignment](
                        (m, n),
                        accum_smem_warp_tile.ptr.load[
                            width=simd_size, alignment=alignment
                        ](swizzled_idx).cast[c_type](),
                    )
        else:
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

    # Store FP32 results to FP32 buffer in global memory.
    else:

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](ln_id)
            var c_reg_frag = c_reg_tile.vectorize[1, 2]().transpose()
            var thread_offset = c_gmem_frag.distance(c.ptr)

            @parameter
            for i in range(__type_of(c_gmem_frag).layout.size()):
                alias src_idx = c_reg_frag.layout(i)
                alias dst_static_idx: UInt = __type_of(c_gmem_frag).layout(i)
                var dst_idx = 0

                @parameter
                if c_layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = c_gmem_frag.runtime_layout(i)

                alias alignment = alignof[SIMD[c_type, 2]]()
                var m = int((thread_offset + dst_idx) // N)
                var n = int((thread_offset + dst_idx) % N)
                if m < M and n < N:
                    var vec = c_reg_frag.ptr.offset(src_idx).load[
                        width=2, alignment = alignof[SIMD[c_type, 2]]()
                    ]()
                    epilogue[alignment=alignment]((m, n), vec)

        else:
            copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
                c_gmem_warp_tile.vectorize[1, 2](),
                c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
                c_gmem_warp_tile.distance(c.ptr),
                M,
                N,
            )


fn multistage_gemm_split_k_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    work_space_type: DType,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
    work_space: NDBuffer[work_space_type, 3],
    num_partitions: UInt,
):
    var M = c.dim(0)
    var N = c.dim(1)

    alias BK = config.block_tile_shape[2]

    # If K is not divisible by num_partitions, the first num_partitions-1 parts
    # will be rounded up to multiple of BK.
    var a_part = a.split[axis=1, alignment=BK](num_partitions, BlockIdx.z())
    var b_part = b.split[axis= 1 if transpose_b else 0, alignment=BK](
        num_partitions, BlockIdx.z()
    )
    var work_space_part = LayoutTensor[work_space_type, c_layout](
        work_space.data + BlockIdx.z() * M * N,
        RuntimeLayout[c_layout].row_major(IndexList[2](M, N)),
    )

    alias k_partition_config = MatmulConfig[
        a_type, b_type, work_space_type, transpose_b
    ](
        config.block_tile_shape,
        config.warp_tile_shape,
        config.num_pipeline_stages,
    )

    multistage_gemm_kernel[
        work_space_type,
        work_space_part.layout,
        a_type,
        a_part.layout,
        b_type,
        b_part.layout,
        transpose_b,
        k_partition_config,
    ](work_space_part, a_part, b_part)
