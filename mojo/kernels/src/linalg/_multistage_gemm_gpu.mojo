# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from collections.optional import OptionalReg
from math import ceildiv
from sys import alignof, simdwidthof, sizeof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import WARP_SIZE, BlockIdx, GridDim, ThreadIdx, barrier, lane_id
from gpu.host import Context, FuncAttribute, Function, Stream, synchronize
from gpu.host.memory import _copy_device_to_host, _copy_host_to_device
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
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
from layout.nd_buffer_stub import (
    copy_from_nd_buffer,
    copy_from_nd_buffer_async,
    copy_to_nd_buffer,
    distribute,
    vectorize,
)
from layout.swizzle import Swizzle
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from memory import UnsafePointer
from memory.reference import _GPUAddressSpace as AddressSpace

from utils.index import Index

from .matmul_gpu import matmul_kernel_naive
from .utils import apply_epilogue, elementwise_epilogue_type
from .utils_gpu import block_swizzle


@always_inline
fn distance[
    type: DType, //
](arg0: UnsafePointer[Scalar[type]], arg1: UnsafePointer[Scalar[type]]) -> Int:
    return (int(arg0) - int(arg1)) // sizeof[arg1.type]()


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
](
    c: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL],
    a_iter_arg: LayoutTensorIter[_, a_layout, address_space=_, circular=_],
    b_iter_arg: LayoutTensorIter[b_type, b_layout],
    a_smem_iter_arg: LayoutTensorIter[
        a_type, a_smem_layout, address_space = AddressSpace.SHARED, circular=_
    ],
    inout b_smem_iter: LayoutTensorIter[
        b_type, b_smem_layout, address_space = AddressSpace.SHARED, circular=_
    ],
    num_iters: Int,
    num_a_rows: Optional[Int] = None,
    num_b_rows: Optional[Int] = None,
    next_op_b_iter: LayoutTensorIter[
        b_type, b_next_gmem_layout
    ] = LayoutTensorIter[b_type, b_next_gmem_layout](),
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
                var a_smem_tile = a_smem_iter.next(stage)[]

                if num_a_rows:
                    copy_dram_to_sram_async[
                        thread_layout=async_copy_a_layout,
                        swizzle=swizzle_a,
                        masked=True,
                    ](
                        a_smem_tile.vectorize[1, simd_size](),
                        a_iter[]
                        .bitcast[a_type, address_space = AddressSpace.GENERIC]()
                        .vectorize[1, simd_size](),
                        num_a_rows.value(),
                    )
                else:
                    copy_dram_to_sram_async[
                        thread_layout=async_copy_a_layout,
                        swizzle=swizzle_a,
                    ](
                        a_smem_tile.vectorize[1, simd_size](),
                        a_iter[]
                        .bitcast[a_type, address_space = AddressSpace.GENERIC]()
                        .vectorize[1, simd_size](),
                    )

                a_iter += 1

            @parameter
            if b_iter.address_space == AddressSpace.GENERIC:
                var b_smem_tile = b_smem_iter.next(stage)[]

                if num_b_rows:
                    var num_rows_bound = num_b_rows.value() if transpose_b else max(
                        0, num_b_rows.value() - stage * BK
                    )
                    copy_dram_to_sram_async[
                        thread_layout=async_copy_b_layout,
                        swizzle=swizzle_b,
                        masked=True,
                    ](
                        b_smem_tile.vectorize[1, simd_size](),
                        b_iter[]
                        .bitcast[b_type, address_space = AddressSpace.GENERIC]()
                        .vectorize[1, simd_size](),
                        num_rows_bound,
                    )
                else:
                    copy_dram_to_sram_async[
                        thread_layout=async_copy_b_layout,
                        swizzle=swizzle_b,
                    ](
                        b_smem_tile.vectorize[1, simd_size](),
                        b_iter[]
                        .bitcast[b_type, address_space = AddressSpace.GENERIC]()
                        .vectorize[1, simd_size](),
                    )

                b_iter += 1

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
    var a_reg_tiles = LayoutTensor[
        a_type,
        Layout.row_major(2 * num_m_mmas, a_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().split[2]()
    # ].stack_allocation().vectorize[1, a_frag_size]().split[2]()
    var b_reg_tiles = LayoutTensor[
        b_type,
        Layout.row_major(2 * num_n_mmas, b_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().vectorize[1, b_frag_size]().split[2]()

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
    if a_iter.address_space == AddressSpace.LOCAL:
        # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
        # Need to cast address space because it's not known at parse time to be LOCAL.
        copy_local_to_local(a_reg_tiles[0], a_iter[])
        a_iter += 1
    else:
        mma_op.load_a[swizzle_a](
            a_warp_tile, a_reg_tiles[0].vectorize[1, a_frag_size]()
        )

    mma_op.load_b(b_warp_tile, b_reg_tiles[0], warp_tile_coordn=int(warp_x))

    @parameter
    if a_iter.address_space == AddressSpace.LOCAL:
        constrained[
            static_num_iters.has_value(),
            "Using input in registers requires static iteration bound.\n",
        ]()

        @parameter
        for k_tile_id in range(static_num_iters.get()):
            var a_smem_iter_tmp = a_smem_iter.next(k_tile_id)
            var b_smem_iter_tmp = b_smem_iter.next(k_tile_id)

            var a_warp_tile = a_smem_iter_tmp[].tile[WM, BK](int(warp_y), 0)
            var b_warp_tile = b_smem_iter_tmp[].tile[
                b_wtile_dim0, b_wtile_dim1
            ](
                b_wtile_coord0,
                b_wtile_coord1,
            )

            # Perform prefetch registers and mma until current shared memory tile's
            # data has all been loaded to registers.
            @parameter
            for k_mma in range(num_k_mmas):
                var current = k_mma % 2
                var next = (k_mma + 1) % 2

                @parameter
                if k_mma == num_k_mmas - 1:
                    var a_smem_next_tile = a_smem_iter_tmp.next()[]
                    var b_smem_next_tile = b_smem_iter_tmp.next()[]

                    a_warp_tile = a_smem_next_tile.tile[WM, BK](int(warp_y), 0)
                    b_warp_tile = b_smem_next_tile.tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                # Assume input is the 16x8 output of 16x8x16 or 16x8x8 mma.
                copy_local_to_local(a_reg_tiles[next], a_iter[])
                a_iter += 1

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

                @parameter
                if k_mma + 2 == num_k_mmas:
                    var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                    # Prefetch one k tile (if valid) from global memory to current
                    # shared memory buffer.
                    if prefetch_tile_id < num_iters:

                        @parameter
                        if b_iter.address_space == AddressSpace.GENERIC:
                            var b_smem_prefetch_tile = b_smem_iter_tmp.next(
                                num_pipeline_stages - 1
                            )[]

                            if num_b_rows:
                                var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                    0,
                                    num_b_rows.value() - prefetch_tile_id * BK,
                                )
                                copy_dram_to_sram_async[
                                    thread_layout=async_copy_b_layout,
                                    swizzle=swizzle_b,
                                    masked=True,
                                ](
                                    b_smem_prefetch_tile.vectorize[
                                        1, simd_size
                                    ](),
                                    b_iter[]
                                    .bitcast[
                                        b_type,
                                        address_space = AddressSpace.GENERIC,
                                    ]()
                                    .vectorize[1, simd_size](),
                                    num_rows_bound,
                                )
                            else:
                                copy_dram_to_sram_async[
                                    thread_layout=async_copy_b_layout,
                                    swizzle=swizzle_b,
                                ](
                                    b_smem_prefetch_tile.vectorize[
                                        1, simd_size
                                    ](),
                                    b_iter[]
                                    .bitcast[
                                        b_type,
                                        address_space = AddressSpace.GENERIC,
                                    ]()
                                    .vectorize[1, simd_size](),
                                )
                            b_iter += 1

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(num_pipeline_stages - 2)
                    barrier()

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
        for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                a_smem_iter += 1
                b_smem_iter += 1

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
                        var a_smem_prefetch_tile = a_smem_iter.next(
                            num_pipeline_stages - 1
                        )[]

                        if num_a_rows:
                            copy_dram_to_sram_async[
                                thread_layout=async_copy_a_layout,
                                swizzle=swizzle_a,
                                masked=True,
                            ](
                                a_smem_prefetch_tile.vectorize[1, simd_size](),
                                a_iter[]
                                .bitcast[
                                    a_type, address_space = AddressSpace.GENERIC
                                ]()
                                .vectorize[1, simd_size](),
                                num_a_rows.value(),
                            )

                        else:
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

                        a_iter += 1

                    @parameter
                    if b_iter.address_space == AddressSpace.GENERIC:
                        var b_smem_prefetch_tile = b_smem_iter.next(
                            num_pipeline_stages - 1
                        )[]

                        if num_b_rows:
                            var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                0,
                                num_b_rows.value() - prefetch_tile_id * BK,
                            )
                            copy_dram_to_sram_async[
                                thread_layout=async_copy_b_layout,
                                swizzle=swizzle_b,
                                masked=True,
                            ](
                                b_smem_prefetch_tile.vectorize[1, simd_size](),
                                b_iter[]
                                .bitcast[
                                    b_type, address_space = AddressSpace.GENERIC
                                ]()
                                .vectorize[1, simd_size](),
                                num_rows_bound,
                            )
                        else:
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

                        b_iter += 1
                else:

                    @parameter
                    if continue_prefetch_b:
                        var b_smem_prefetch_tile = b_smem_iter.next(
                            num_pipeline_stages - 1
                        )[].reshape[b_next_smem_layout]()

                        # TODO: can we guard at compile time num_b_rows is set here?
                        var num_rows_bound = num_b_rows.value() if transpose_b_next else max(
                            0,
                            num_b_rows.value()
                            - (prefetch_tile_id - num_iters) * BK,
                        )

                        alias row_size = b_next_smem_layout.stride[0].value()

                        copy_dram_to_sram_async[
                            thread_layout = Layout.row_major(
                                num_threads * simd_size // row_size,
                                row_size // simd_size,
                            ),
                            swizzle = transpose_b_next
                            or b_type.is_half_float(),
                            masked=True,
                        ](
                            b_smem_prefetch_tile.vectorize[1, simd_size](),
                            next_b_iter[]
                            .bitcast[
                                b_type, address_space = AddressSpace.GENERIC
                            ]()
                            .vectorize[1, simd_size](),
                            num_rows_bound,
                        )

                        next_b_iter += 1

                    pass

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()


fn multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    transpose_b: Bool,
    BM: UInt,
    BN: UInt,
    BK: UInt,
    WM: UInt,
    WN: UInt,
    num_threads: UInt,
    num_pipeline_stages: UInt,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        a_type in (DType.float32, DType.bfloat16)
        and a_type == b_type == c_type,
        "Pipeline gemm only supports tf32 or BF16 mma",
    ]()

    constrained[
        (BK == 16 and a_type == DType.float32)
        or (BK == 32 and a_type == DType.bfloat16),
        "Pipeline gemm only supports BK = 16 w/ FP32 and BK = 32 w/ BF16.",
    ]()

    alias simd_size = simdwidthof[c_type]()

    var M: UInt = c.dim[0]()
    alias N: UInt = b_shape.get[0]() if transpose_b else b_shape.get[1]()
    alias K: UInt = b_shape.get[1]() if transpose_b else b_shape.get[0]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == num_threads // WARP_SIZE,
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid = ThreadIdx.x()
    var ln_id = lane_id()

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    # NOTE: the condition ( not (N // BN & 1)) is for a temporary solution
    # for solving mismatches in some shapes
    var block_idx = block_swizzle(
        (int(BlockIdx.x()), int(BlockIdx.y())),
        (int(GridDim.x()), int(GridDim.y())),
    ) if swizzle_block else Index(int(BlockIdx.x()), int(BlockIdx.y()))

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(tid // WARP_SIZE, num_warps_n)

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

    # create input layout tensors A and B
    var a_gmem_slice = LayoutTensor[
        a_type,
        Layout.row_major(BM, K),
    ](a.data.offset(BM * block_idx[1] * K))

    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b_gmem_tensor = LayoutTensor[
        b_type,
        b_layout,
    ](b.data)

    # global memory iterator
    var a_gmem_iter = a_gmem_slice.tiled_iterator[BM, BK, axis=1](0, 0)
    var b_tile_coords = (block_idx[0], 0) if transpose_b else (0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b_gmem_tensor.tiled_iterator[
        BD_0, BD_1, axis=b_tile_axis
    ](b_tile_coords[0], b_tile_coords[1])

    alias async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias async_copy_b_layout = Layout.row_major(
        num_threads * simd_size // BD_1, BD_1 // simd_size
    )

    alias swizzle_b = transpose_b or b_type.is_half_float()

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    for stage in range(int(num_pipeline_stages - 1)):
        var a_smem_tile = a_smem_iter.next(stage)[]
        var b_smem_tile = b_smem_iter.next(stage)[]

        if M % BM == 0:
            copy_dram_to_sram_async[
                thread_layout=async_copy_a_layout,
                swizzle=True,  # xor_2bits_per8T,
            ](
                a_smem_tile.vectorize[1, simd_size](),
                a_gmem_iter[].vectorize[1, simd_size](),
            )
        else:
            copy_dram_to_sram_async[
                thread_layout=async_copy_a_layout,
                swizzle=True,
            ](
                a_smem_tile.vectorize[1, simd_size](),
                a_gmem_iter[].vectorize[1, simd_size](),
                a_gmem_iter[].distance(a.data),
                M,
                K,
            )

        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout, swizzle=swizzle_b
        ](
            b_smem_tile.vectorize[1, simd_size](),
            b_gmem_iter[].vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        a_gmem_iter += 1
        b_gmem_iter += 1

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
    # TODO: parameterize fragment size based on data type.
    var a_reg_tiles = LayoutTensor[
        a_type,
        Layout.row_major(2 * num_m_mmas, a_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().vectorize[1, a_frag_size]().split[2]()
    var b_reg_tiles = LayoutTensor[
        b_type,
        Layout.row_major(2 * num_n_mmas, b_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().vectorize[1, b_frag_size]().split[2]()
    var c_reg_tile = LayoutTensor[
        accum_type, Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size)
    ].stack_allocation().fill(0)

    var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)

    # TODO: warp the following in the tile method, maybe tile[shape: IntTuple].
    # I can't use b_warp_tile = ... if transpose_b else ... because the operands
    # are deduced as different types since their layout are different.
    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    mma_op.load_a(a_warp_tile, a_reg_tiles[0])
    mma_op.load_b(b_warp_tile, b_reg_tiles[0])

    var num_k_tiles = ceildiv(K, BK)

    for k_tile_id in range(num_k_tiles):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma_0 in range(int(num_k_mmas)):
            alias k_mma = UInt(k_mma_0)
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                a_smem_iter += 1
                b_smem_iter += 1

                a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
                b_warp_tile = b_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
                    b_wtile_coord0, b_wtile_coord1
                )

            mma_op.load_a(
                a_warp_tile, a_reg_tiles[next], (k_mma + 1) % num_k_mmas
            )
            mma_op.load_b(
                b_warp_tile, b_reg_tiles[next], (k_mma + 1) % num_k_mmas
            )

            mma_op.mma(
                a_reg_tiles[current],
                b_reg_tiles[current],
                c_reg_tile.vectorize[1, c_frag_size](),
            )

            @parameter
            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_k_tiles:
                    # fmt: off
                    var a_smem_prefetch_tile = a_smem_iter.next(num_pipeline_stages - 1)[]
                    var b_smem_prefetch_tile = b_smem_iter.next(num_pipeline_stages - 1)[]
                    # fmt: on

                    # TODO: Extend the async copy instrinsic to creat dummy copies. The
                    # prefetch for the three two iterations should be dummy.
                    if M % BM == 0:
                        copy_dram_to_sram_async[
                            thread_layout=async_copy_a_layout,
                            swizzle=True,
                        ](
                            a_smem_prefetch_tile.vectorize[1, simd_size](),
                            a_gmem_iter[].vectorize[1, simd_size](),
                        )
                    else:
                        copy_dram_to_sram_async[
                            thread_layout=async_copy_a_layout,
                            swizzle=True,
                        ](
                            a_smem_prefetch_tile.vectorize[1, simd_size](),
                            a_gmem_iter[].vectorize[1, simd_size](),
                            a_gmem_iter[].distance(a.data),
                            M,
                            K,
                        )

                    copy_dram_to_sram_async[
                        thread_layout=async_copy_b_layout,
                        swizzle=swizzle_b,
                    ](
                        b_smem_prefetch_tile.vectorize[1, simd_size](),
                        b_gmem_iter[].vectorize[1, simd_size](),
                    )

                    a_gmem_iter += 1
                    b_gmem_iter += 1

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()

    # Map global memory tile down to thread.
    var c_gmem_slice = LayoutTensor[c_type, Layout.row_major(BM, N)](
        c.data.offset(block_idx[1] * BM * N)
    )
    var c_gmem_tile = c_gmem_slice.tile[BM, BN](0, block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](int(warp_y), int(warp_x))

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if c_type.is_half_float():
        # Stage fragments in shared memory. Reuse a_smem for c tile in smem
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
        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            alias c_store_layout = Layout.row_major(
                num_threads * simd_size // BN, BN // simd_size
            )
            var c_gmem_frag = c_gmem_tile.vectorize[1, simd_size]().distribute[
                c_store_layout
            ](ThreadIdx.x())
            var c_smem_frag = accum_smem_tile.vectorize[
                1, simd_size
            ]().distribute[c_store_layout](ThreadIdx.x())
            var thread_offset = c_gmem_frag.distance(c.data)
            alias num_stores_per_thread = c_gmem_frag.layout.size()
            alias src_align = alignof[
                SIMD[accum_type, simdwidthof[accum_type]()]
            ]()
            alias dst_align = alignof[SIMD[c_type, simd_size]]()

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = c_smem_frag.layout(i)
                alias dst_idx = c_gmem_frag.layout(i)
                var m = int((thread_offset + dst_idx) // N)
                var n = int((thread_offset + dst_idx) % N)
                if m < M and n < N:
                    epilogue(
                        (m, n),
                        c_smem_frag.ptr.load[
                            width=simd_size, alignment=src_align
                        ](src_idx).cast[c_type](),
                    )
        else:
            if M % BM == 0:
                copy_sram_to_dram[
                    thread_layout = Layout.row_major(
                        num_threads * simd_size // BN, BN // simd_size
                    )
                ](
                    c_gmem_tile.vectorize[1, simd_size](),
                    accum_smem_tile.vectorize[1, simd_size](),
                )
            else:
                copy_sram_to_dram[
                    thread_layout = Layout.row_major(
                        num_threads * simd_size // BN, BN // simd_size
                    )
                ](
                    c_gmem_tile.vectorize[1, simd_size](),
                    accum_smem_tile.vectorize[1, simd_size](),
                    c_gmem_tile.distance(c.data),
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
            var thread_offset = c_gmem_frag.distance(c.data)

            @parameter
            for i in range(c_gmem_frag.layout.size()):
                alias src_idx = c_reg_frag.layout(i)
                alias dst_idx: UInt = c_gmem_frag.layout(i)
                var m = int((thread_offset + dst_idx) // N)
                var n = int((thread_offset + dst_idx) % N)
                if m < M and n < N:
                    var vec = c_reg_frag.ptr.offset(src_idx).load[
                        width=2, alignment = alignof[SIMD[c_type, 2]]()
                    ]()
                    epilogue((m, n), vec)

        else:
            copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
                c_gmem_warp_tile.vectorize[1, 2](),
                c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
                c_gmem_warp_tile.distance(c.data),
                M,
                N,
            )
