# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from algorithm.functional import elementwise
from collections import OptionalReg
from math import ceildiv
from os import abort
from sys import argv, alignof, simdwidthof, sizeof

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
from gpu.host import (
    DeviceContext,
    FuncAttribute,
)
from gpu.host.info import A100
from gpu.host._compile import _get_gpu_target
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
    Fill,
    CacheEviction,
)
from gpu.mma import ld_matrix, mma
from layout.runtime_layout import RuntimeLayout

# from layout.fillers import arange, random
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
from layout._utils import ManagedLayoutGPUTensor, ManagedLayoutTensor
from math import exp2
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace as AddressSpace

from random import randn, rand

from utils.index import Index, IndexList
from utils import StaticTuple

from linalg.utils import apply_epilogue, elementwise_epilogue_type
from linalg.utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    block_swizzle,
    _bk_base,
)

from linalg.matmul_gpu import multistage_gemm
from linalg._multistage_gemm_gpu import distance, multistage_gemm_kernel

from testing import assert_almost_equal
from utils.numerics import FPUtils

import benchmark


# Perhaps we could take an approach where we try
# to reuse `multistage_mma`, but for now, it's
# simpler to be explicit, as this allows us
# to make sure we get things such as the swizzles correct
# w/ respect to the memory layouts.
# That is, if we want to keep smem continuous, we could
# use something like:
#
# if `tranpose_b`: # the same after coalesce
#   ((WN//2,2), WK) : ((K, K*WN//2), 1)
# else:
#   (WK, (WN//2,2)) : (WN//2, (1, K*WN//2))
#
# which will yield a different memory layout when
# `!transpose_b`.
#
# However, this would require special care, e.g. with swizzles,
# so for now, it is easier to simply manage the c reg tiles and
# b0 and b1 memory separately.
@always_inline
fn multistage_dual_mma[
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
    k_group_size: UInt = 1,
](
    c0: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL, **_],
    c1: LayoutTensor[c_type, c_layout, address_space = AddressSpace.LOCAL, **_],
    a_iter_arg: LayoutTensorIter[_, a_layout, **_],
    b0_iter_arg: LayoutTensorIter[b_type, b_layout, **_],
    b1_iter_arg: LayoutTensorIter[b_type, b_layout, **_],
    a_smem_iter_arg: LayoutTensorIter[
        a_type, a_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    inout b0_smem_iter: LayoutTensorIter[
        b_type, b_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    inout b1_smem_iter: LayoutTensorIter[
        b_type, b_smem_layout, address_space = AddressSpace.SHARED, **_
    ],
    num_iters: Int,
    /,
    *,
    num_b_rows: OptionalReg[Int] = None,
):
    constrained[
        b0_iter_arg.address_space == b1_iter_arg.address_space,
        "b0 and b1 should have the same address space",
    ]()
    alias simd_size = simdwidthof[a_type]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = warp_broadcast(tid // WARP_SIZE)

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_iter = a_iter_arg
    var b0_iter = b0_iter_arg
    var b1_iter = b1_iter_arg
    var a_smem_iter = a_smem_iter_arg

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
    fn _mask_tensor_row(
        tensor: LayoutTensor, num_rows: Int
    ) -> __type_of(tensor) as result:
        return __type_of(tensor)(
            tensor.ptr,
            RuntimeLayout(
                RuntimeTuple[tensor.layout.shape, unsigned=True](
                    num_rows, tensor.dim(1)
                ),
                tensor.runtime_layout.stride,
            ),
        )

    @always_inline
    @parameter
    fn _copy_single_tensor_to_sram(dst: LayoutTensor, src: LayoutTensor):
        copy_dram_to_sram_async[
            thread_layout=async_copy_a_layout,
            swizzle=swizzle_a,
        ](
            dst.vectorize[1, simd_size](),
            src.vectorize[1, simd_size](),
        )

    @always_inline
    @parameter
    fn _copy_dual_tensor_to_sram(
        b0_dst: LayoutTensor,
        b1_dst: LayoutTensor,
        b0_src: LayoutTensor,
        b1_src: LayoutTensor,
    ):
        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout,
            swizzle=swizzle_b,
        ](
            b0_dst.vectorize[1, simd_size](),
            b0_src.vectorize[1, simd_size](),
        )
        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout,
            swizzle=swizzle_b,
        ](
            b1_dst.vectorize[1, simd_size](),
            b1_src.vectorize[1, simd_size](),
        )

    # Prefetch (num_pipeline_stages - 1) stages.

    @parameter
    for stage in range(num_pipeline_stages - 1):
        var a_smem_tile = a_smem_iter.next_unsafe(stage)[]

        _copy_single_tensor_to_sram(a_smem_tile, a_iter[])

        a_iter._incr()

        var b0_smem_tile = b0_smem_iter.next_unsafe(stage)[]
        var b1_smem_tile = b1_smem_iter.next_unsafe(stage)[]
        if num_b_rows:
            var num_rows_bound = num_b_rows.value() if transpose_b else max(
                0, num_b_rows.value() - stage * BK
            )
            _copy_dual_tensor_to_sram(
                b0_smem_tile,
                b1_smem_tile,
                _mask_tensor_row(b0_iter[], num_rows_bound),
                _mask_tensor_row(b1_iter[], num_rows_bound),
            )
        else:
            _copy_dual_tensor_to_sram(
                b0_smem_tile, b1_smem_tile, b0_iter[], b1_iter[]
            )
        b0_iter._incr()
        b1_iter._incr()

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
    alias num_n_mmas = WN // (2 * MMA_N)
    constrained[
        num_k_mmas % (2 * k_group_size) == 0,
        "num_k_mmas must be an integer multiple of 2*k_group_size",
    ]()
    constrained[num_n_mmas % 2 == 0]()

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

    var b0_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()
    var b1_reg_tiles = tb[b_type]().row_major[
        2 * k_group_size * num_n_mmas, b_frag_size
    ]().local().alloc().vectorize[1, b_frag_size]().split[2 * k_group_size]()

    var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)

    alias b_wtile_dim0 = WN // 2 if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN // 2
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b0_warp_tile = b0_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )
    var b1_warp_tile = b1_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    @parameter
    for i in range(int(k_group_size)):
        mma_op.load_a[swizzle_a](
            a_warp_tile, a_reg_tiles[i].vectorize[1, a_frag_size](), i
        )
        mma_op.load_b(b0_warp_tile, b0_reg_tiles[i], i, int(warp_x))
        mma_op.load_b(b1_warp_tile, b1_reg_tiles[i], i, int(warp_x))

    for k_tile_id in range(num_iters):
        var a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
        var b0_warp_tile = b0_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )
        var b1_warp_tile = b1_smem_iter[].tile[b_wtile_dim0, b_wtile_dim1](
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
                        var a_smem_prefetch_tile = a_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        _copy_single_tensor_to_sram(
                            a_smem_prefetch_tile, a_iter[]
                        )
                        a_iter._incr()

                        var b0_smem_prefetch_tile = b0_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        var b1_smem_prefetch_tile = b1_smem_iter.next_unsafe(
                            num_pipeline_stages - 1
                        )[]
                        if num_b_rows:
                            var num_rows_bound = num_b_rows.value() if transpose_b else max(
                                0,
                                num_b_rows.value() - prefetch_tile_id * BK,
                            )
                            _copy_dual_tensor_to_sram(
                                b0_smem_prefetch_tile,
                                b1_smem_prefetch_tile,
                                _mask_tensor_row(b0_iter[], num_rows_bound),
                                _mask_tensor_row(b1_iter[], num_rows_bound),
                            )
                        else:
                            _copy_dual_tensor_to_sram(
                                b0_smem_prefetch_tile,
                                b1_smem_prefetch_tile,
                                b0_iter[],
                                b1_iter[],
                            )
                        b0_iter._incr()
                        b1_iter._incr()

                    async_copy_commit_group()

                    # Guard the next k tile's shared memory buffer.
                    async_copy_wait_group(num_pipeline_stages - 2)
                    barrier()

                    a_smem_iter._incr()
                    b0_smem_iter._incr()
                    b1_smem_iter._incr()

                    a_warp_tile = a_smem_iter[].tile[WM, BK](int(warp_y), 0)
                    b0_warp_tile = b0_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)
                    b1_warp_tile = b1_smem_iter[].tile[
                        b_wtile_dim0, b_wtile_dim1
                    ](b_wtile_coord0, b_wtile_coord1)

                alias kidx = int(k_mma_next % num_k_mmas)
                mma_op.load_a[swizzle_a](
                    a_warp_tile,
                    a_reg_tiles[next].vectorize[1, a_frag_size](),
                    kidx,
                )

                mma_op.load_b(
                    b0_warp_tile,
                    b0_reg_tiles[next],
                    kidx,
                    int(warp_x),
                )
                mma_op.load_b(
                    b1_warp_tile,
                    b1_reg_tiles[next],
                    kidx,
                    int(warp_x),
                )

            @parameter
            for k_mma1 in range(int(k_group_size)):
                alias k_mma = UInt32(k_mma0 * k_group_size + k_mma1)
                alias current = k_mma % num_reg_tiles
                mma_op.mma(
                    a_reg_tiles[int(current)].vectorize[1, a_frag_size](),
                    b0_reg_tiles[int(current)],
                    c0.vectorize[1, c_frag_size](),
                )
                mma_op.mma(
                    a_reg_tiles[int(current)].vectorize[1, a_frag_size](),
                    b1_reg_tiles[int(current)],
                    c1.vectorize[1, c_frag_size](),
                )


alias binary_fn_type = fn[type: DType, width: Int] (
    SIMD[type, width], SIMD[type, width]
) -> SIMD[type, width]


fn binary_mul[
    type: DType, width: Int
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x * y


fn binary_sub[
    type: DType, width: Int
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x - y


fn multistage_dual_gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    binary_lambda_fn: binary_fn_type = binary_mul,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b0: LayoutTensor[b_type, b_layout],
    b1: LayoutTensor[b_type, b_layout],
):
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        a_type in (DType.float32, DType.bfloat16) and a_type == b_type,
        "Pipeline gemm only supports tf32 or BF16 mma",
    ]()

    alias simd_size = simdwidthof[c_type]()

    var M: UInt = c.dim(0)
    var N: UInt = b0.dim(0) if transpose_b else b0.dim(1)
    var K: UInt = b0.dim(1) if transpose_b else b0.dim(0)
    # we require b0 and b1 to be of the same size

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
    alias b_smem_size = num_pipeline_stages * BK * BN // 2
    alias BD_0 = BN // 2 if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN // 2
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b0_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_smem_size)
    var b1_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](b_smem + b_smem_size, b_smem_size)

    # create input layout tensors A and Bv
    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = (block_idx[0], 0) if transpose_b else (0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b0_gmem_iter = b0.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )
    var b1_gmem_iter = b1.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    # Compute MMA config
    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // (2 * MMA_N)

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias c_frag_size = frag_size[2]
    var c0_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)
    var c1_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, c_frag_size
    ]().local().alloc().fill(0)

    multistage_dual_mma[
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
        c0_reg_tile,
        c1_reg_tile,
        a_gmem_iter,
        b0_gmem_iter,
        b1_gmem_iter,
        a_smem_iter,
        b0_smem_iter,
        b1_smem_iter,
        ceildiv(K, BK),
    )

    alias HWN = WN // 2
    # c_reg_tile is indeced in `mma` as follows:
    # c_frag[n_mma * num_m_mmas + m_mma, 0]
    # so, the first dim can be viewed as `row_major(num_n_mmas, num_m_mmas)`
    # we combine the second half to the first half
    var c0_vec = c0_reg_tile.vectorize[1, c_frag_size]()
    var c1_vec = c1_reg_tile.vectorize[1, c_frag_size]()

    @parameter
    for n_mma in range(c0_vec.layout.size()):
        c0_vec[n_mma, 0] = binary_lambda_fn(c0_vec[n_mma, 0], c1_vec[n_mma, 0])

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN // 2](block_idx[1], block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, HWN](int(warp_y), int(warp_x))

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if c_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=HWN, access_size=MMA_N
        ]()

        var accum_smem_warp_tile = tb[c_type]().row_major[
            WM, HWN
        ]().shared().view(a_smem.bitcast[Scalar[c_type]]() + warp_id * WM * HWN)

        copy_local_to_sram[
            thread_layout = Layout.row_major(8, 4),
            swizzle=swizzle,
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            c0_reg_tile.vectorize[1, 2]().transpose(),
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
                WARP_SIZE * simd_size // HWN, HWN // simd_size
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
                        WARP_SIZE * simd_size // HWN, HWN // simd_size
                    ),
                    swizzle=swizzle,
                ](
                    c_gmem_warp_tile.vectorize[1, simd_size](),
                    accum_smem_warp_tile.vectorize[1, simd_size](),
                )
            else:
                copy_sram_to_dram[
                    thread_layout = Layout.row_major(
                        WARP_SIZE * simd_size // HWN, HWN // simd_size
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
            var c_reg_frag = c0_reg_tile.vectorize[1, 2]().transpose()
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
                c0_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
                c_gmem_warp_tile.distance(c.ptr),
                M,
                N,
            )


fn multistage_dual_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    binary_lambda_fn: binary_fn_type = binary_mul,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b0: LayoutTensor[b_type, b_layout],
    b1: LayoutTensor[b_type, b_layout],
    ctx: DeviceContext,
):
    var M = c.dim(0)
    var N = c.dim(1)

    try:
        alias gemm_kernel_type = multistage_dual_gemm_kernel[
            c_type,
            c_layout,
            a_type,
            a_layout,
            b_type,
            b_layout,
            transpose_b,
            config,
            elementwise_lambda_fn,
            binary_lambda_fn,
        ]

        var gemm_kernel = ctx.compile_function[gemm_kernel_type,](
            threads_per_block=int(config.num_threads()),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                config.shared_mem_usage()
            ),
        )

        ctx.enqueue_function(
            gemm_kernel,
            c,
            a,
            b0,
            b1,
            grid_dim=config.grid_dim(M, 2 * N),
            block_dim=config.block_dim(),
            shared_mem_bytes=config.shared_mem_usage(),
        )
        ctx.synchronize()
    except e:
        abort(e)


fn multistage_gemm_simple[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    binary_lambda_fn: binary_fn_type = binary_mul,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
    ctx: DeviceContext,
) raises:
    var M = c.dim(0)
    var N = c.dim(1)

    # Dispatch w/o split K
    alias gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b,
        config,
        elementwise_lambda_fn,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type,](
        threads_per_block=int(config.num_threads()),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    ctx.enqueue_function(
        gemm_kernel,
        c,
        a,
        b,
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
    )


fn naive_dual_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    binary_lambda_fn: binary_fn_type = binary_mul,
](
    c01: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b01: LayoutTensor[b_type, b_layout],
    ctx: DeviceContext,
):
    try:
        multistage_gemm_simple[transpose_b=transpose_b, config=config](
            c01, a, b01, ctx
        )

        alias simd_width = simdwidthof[
            c_type, target = _get_gpu_target["sm_80"]()
        ]()
        alias align = alignof[SIMD[c_type, simd_width]]()

        var M = c01.dim(0)
        var N = c01.dim(1) // 2

        @always_inline
        @__copy_capture(c01, N)
        @parameter
        fn binary[simd_width: Int, rank: Int](idx0: IndexList[rank]):
            var m: Int = idx0[0]
            var n: Int = idx0[1]
            c01.vectorize[1, simd_width]()[m, n // simd_width] -= c01.vectorize[
                1, simd_width
            ]()[m, (N + n) // simd_width]

        ctx.synchronize()
        elementwise[binary, simd_width, target="gpu"](IndexList[2](M, N), ctx)
        ctx.synchronize()
    except e:
        abort(e)


fn runtime_row_major[
    cols: Int
](rows: Int) -> RuntimeLayout[
    Layout(IntTuple(UNKNOWN_VALUE, cols), IntTuple(cols, 1))
] as res:
    return __type_of(res).row_major(IndexList[2]((rows, cols)))


fn test_dual_matmul[
    transpose_b: Bool, N: Int = 512, K: Int = 512
](ctx: DeviceContext, M: Int = 512, do_benchmark: Bool = False) raises:
    alias dst_type = DType.float32
    alias src_type = DType.bfloat16
    alias warp_shape = Index(64, 64, _bk_base[src_type]())
    alias config = MatmulConfig[src_type, src_type, dst_type, transpose_b]()
    alias M128_N28672_K4096_config = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )
    alias M256_N28672_K4096_config_a100 = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(64, 256, 64),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )
    alias M256_N28672_K4096_config = M256_N28672_K4096_config_a100 if M256_N28672_K4096_config_a100.shared_mem_usage() <= ctx.device_info.shared_memory_per_multiprocessor else config
    alias M512_N28672_K4096_config = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )

    var layout_a = runtime_row_major[K](M)
    alias layout_b = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var layout_c = runtime_row_major[N](M)

    var mat_a = ManagedLayoutGPUTensor[src_type](layout_a)
    randn(mat_a.tensor.ptr, layout_a.size())
    var mat_b0 = ManagedLayoutGPUTensor[src_type, layout_b]()
    var mat_b1 = ManagedLayoutGPUTensor[src_type, layout_b]()

    rand(mat_b0.tensor.ptr, layout_b.size(), min=0.0, max=1.0)
    rand(mat_b1.tensor.ptr, layout_b.size(), min=-1.0, max=0.0)
    var mat_c = ManagedLayoutGPUTensor[dst_type](layout_c)

    @always_inline
    @parameter
    fn run_dual_gemm():
        if M <= 128:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M128_N28672_K4096_config,
            ](mat_c.tensor, mat_a.tensor, mat_b0.tensor, mat_b1.tensor, ctx)
        elif M <= 256:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M256_N28672_K4096_config,
            ](mat_c.tensor, mat_a.tensor, mat_b0.tensor, mat_b1.tensor, ctx)
        else:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M512_N28672_K4096_config,
            ](mat_c.tensor, mat_a.tensor, mat_b0.tensor, mat_b1.tensor, ctx)

    var dual_gemm_time: Float64 = 0.0
    if do_benchmark:
        dual_gemm_time = benchmark.run[run_dual_gemm](
            max_runtime_secs=5.0
        ).mean()
        print(
            "     DualGEMM[M=",
            M,
            ",N=",
            N,
            ",K=",
            K,
            "] took ",
            1e3 * dual_gemm_time,
            " ms",
            sep="",
        )
    else:
        run_dual_gemm()

    alias layout_b01 = Layout.row_major(
        2 * N, K
    ) if transpose_b else Layout.row_major(K, 2 * N)
    var mat_b01 = ManagedLayoutGPUTensor[src_type, layout_b01]()
    alias src_simd_width = simdwidthof[src_type]()

    var mat_b01v = mat_b01.tensor.vectorize[1, src_simd_width]()

    @parameter
    if transpose_b:
        for n in range(N):
            for k in range(K // src_simd_width):
                mat_b01v[n, k] = rebind[SIMD[src_type, mat_b01v.element_size]](
                    mat_b0.tensor.vectorize[1, src_simd_width]()[n, k]
                )
                mat_b01v[N + n, k] = rebind[
                    SIMD[src_type, mat_b01v.element_size]
                ](mat_b1.tensor.vectorize[1, src_simd_width]()[n, k])
    else:
        alias Niter = N // src_simd_width
        for k in range(K):
            for n in range(Niter):
                mat_b01v[k, n] = rebind[SIMD[src_type, mat_b01v.element_size]](
                    mat_b0.tensor.vectorize[1, src_simd_width]()[k, n]
                )
                mat_b01v[k, Niter + n] = rebind[
                    SIMD[src_type, mat_b01v.element_size]
                ](mat_b1.tensor.vectorize[1, src_simd_width]()[k, n])

    _ = mat_b0^
    _ = mat_b1^

    var layout_c01 = runtime_row_major[2 * N](M)
    var mat_c01 = ManagedLayoutGPUTensor[dst_type](layout_c01)

    @always_inline
    @parameter
    fn run_naive_dual_gemm():
        if M <= 128:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M128_N28672_K4096_config,
            ](
                mat_c01.tensor,
                mat_a.tensor,
                mat_b01.tensor,
                ctx,
            )
        elif M <= 256:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M256_N28672_K4096_config,
            ](
                mat_c01.tensor,
                mat_a.tensor,
                mat_b01.tensor,
                ctx,
            )
        else:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M512_N28672_K4096_config,
            ](
                mat_c01.tensor,
                mat_a.tensor,
                mat_b01.tensor,
                ctx,
            )

    if do_benchmark:
        var dgs = benchmark.run[run_naive_dual_gemm](
            max_runtime_secs=5.0
        ).mean()
        print(
            "NaiveDualGEMM[M=",
            M,
            ",N=",
            N,
            ",K=",
            K,
            "] took ",
            1e3 * dgs,
            " ms (",
            round(100 * dgs / dual_gemm_time, ndigits=5),
            "%)",
            sep="",
        )
    else:
        run_naive_dual_gemm()
    var mat_c_ref = mat_c01.tensor.split[axis=1](count=2, idx=0)
    _ = mat_a^
    _ = mat_b01^

    alias cbrt_eps = exp2(FPUtils[dst_type].mantissa_width() / -3).cast[
        dst_type
    ]()
    alias dst_simd_width = simdwidthof[dst_type]()
    # elementwise
    for m in range(M):
        for n in range(N // dst_simd_width):
            assert_almost_equal(
                rebind[SIMD[dst_type, dst_simd_width]](
                    mat_c.tensor.vectorize[1, dst_simd_width]()[m, n]
                ),
                rebind[SIMD[dst_type, dst_simd_width]](
                    mat_c_ref.vectorize[1, dst_simd_width]()[m, n]
                ),
                atol=cbrt_eps,
                rtol=cbrt_eps,
            )

    _ = mat_c^
    _ = mat_c01^


fn main() raises:
    var do_benchmark: Bool = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            do_benchmark = True
    with DeviceContext() as ctx:
        # test_dual_matmul[transpose_b=False](ctx, do_benchmark=do_benchmark)
        # test_dual_matmul[transpose_b=True](ctx, do_benchmark=do_benchmark)
        alias Ms = StaticTuple[Int, 3](128, 256, 1024)
        alias N = 14336
        alias K = 4096
        for m_idx in range(len(Ms)):
            var M = Ms[m_idx]
            # print("m_idx=", m_idx, "M=", M)
            # test_dual_matmul[transpose_b=False, N=N, K=K](ctx, M=M, do_benchmark=do_benchmark)
            test_dual_matmul[transpose_b=True, N=N, K=K](
                ctx, M=M, do_benchmark=do_benchmark
            )
