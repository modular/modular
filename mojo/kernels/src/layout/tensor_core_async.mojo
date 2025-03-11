# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Async Tensor Cores to perform asynchronous
matrix multiplication operations.
"""
from sys import sizeof

from gpu import WARP_SIZE, barrier
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import thread_idx
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout.layout import (
    is_row_major,
    tile_to_shape,
    upcast,
    MakeLayoutList,
    logical_divide,
)
from memory.unsafe_pointer import UnsafePointer

from utils import Index, IndexList

# ===-----------------------------------------------------------------------===#
# WGMMA shared memory layout                                                   #
# ===-----------------------------------------------------------------------===#
#
# TODO: add more context for WGMMA, core matrix. Assuming reader know them for now
#
#
# -------------------------------
# M/N x K, K-major, w/o swizzling
# -------------------------------
#
# Consider core matrix cm_M x cm_K, where cm_M = 8 and cm_K = 16 // sizeof[type]()
#
#    !!!! core matrix one contiguous chunk in row-major(cm_M, cm_K) !!!!
#
# E.g.
#                 | core matrix 00 | core matrix 01 |
#                 | core matrix 10 | core matrix 11 |
#
# The elements are stored in shared memory as
#
#       | core matrix 00 | core matrix 10 | core matrix 01 | core matrix 11 |
#
# and each core matrix ij is a contiguous 128B with row_major(cm_M, cn_K) layout.
#
# The share memory tile is logically mapped to a BM x BK sub-matrix in global memory,
# Without swizzling, the tile layout is
#
#     ((cm_M,  BM // cm_M), (cm_K, BK // cm_K))
#   : ((cm_K, cm_M * cm_K), (   1, BM  * cm_K))
#
# coalesceable to (but we use the above like cutlass)
#
#     (  BM, (cm_K, BK // cm_K))
#   : (cm_K, (   1, BM  * cm_K))
#
#
# WGMMA descriptor layout:
#
# B16 => no swizzle
#
# B16  : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1,LBO)) where x = 1
# B32  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 2
# B64  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 4
# B128 : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 8
#
# cm_M = cm_N = 8
# cm_K = T = 16 // sizeof[type]()
# When there is swizzle, there is the swizzle mode constraint:
# `swizzle_mode.bytes() = 16x = 16 xT/T = 16 xT/(16 / sizeof[type]()) = xT * sizeof[type]().
#
# Tiled descriptors:
#
# B16  : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2k)):((T,SBO),(1,LBO))
#
# When the layout is dense, and the core matrices are tiled in column major
# like above comment. We have `SBO = cm_M * T = cm_M * cm_K` and `LBO = (cm_M *
# m) * T = BM * T = BM * cm_K`. The minimal dense layout is then `BM = 8` and
# `BK = T` which is exactly the core matrix layout.
#
#
# B32  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 2
# B64  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 4
# B128 : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 8
#
# When the layout is dense, we have the unique solution `xT = T*2k = BK`, `SBO
# = cm_M * BK`. The minimal dense layout is then `m = 1` and `2k = x`.
#
# ----------------------------
# K x M/N, MN-major, siwzzling
# ----------------------------
#
# MN-major layouts are hard to reason. We port cutlass' three canonical
# layouts with some refactorization:
#
# B32   : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
# B64   : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
# B128  : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
#
# T = 16B // sizeof[type]()
# m = BM  // (2T or 4T or 8T)
# k = BK  // 8
#
# We simplify them to
#
# B32   : Swizzle<1,4,3> o smem_ptr o ((2T,m),(8,k)):((1,LBO),(2T,SBO))
# B64   : Swizzle<2,4,3> o smem_ptr o ((4T,m),(8,k)):((1,LBO),(4T,SBO))
# B128  : Swizzle<3,4,3> o smem_ptr o ((8T,m),(8,k)):((1,LBO),(8T,SBO))
#
# `2/4/8 * T` is generalized as `swizzle.bytes() // sizeof[type]()`.

alias supported_mma_shape = (
    Index(64, 8, 8),
    Index(64, 8, 16),
    Index(64, 16, 16),
    Index(64, 32, 16),
    Index(64, 64, 16),
    Index(64, 128, 16),
    Index(64, 256, 16),
)

# Core matrix dimensions
alias _CM_M = 8
alias _CM_N = 8
alias _CM_K_BYTES = 16
alias _CM_K_BITS = _CM_K_BYTES * 8
# TODO: unify by the following
alias _CM_NUM_ROWS = 8
alias _CM_ROW_BYTES = 16

alias WGMMA_K_BYTES = 32

alias _CM_LAYOUT_BITS = Layout.row_major(_CM_M, _CM_K_BITS)
alias _CM_TILE_STRIDE = IntTuple(1, _CM_K_BITS)


# constructs core matrix or "minimal dense" layout in bytes as described in file
# header.
fn _select_k_atom_bits[
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    return Layout.row_major(_CM_M, _CM_K_BITS * (swizzle_mode.bytes() // 16))


fn select_k_atom[
    type: DType,
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    alias a = _select_k_atom_bits[swizzle_mode]()
    return upcast(a, sizeof[type]() * 8)


fn _checked_tile_shape[
    type: DType,
    swizzle_mode: TensorMapSwizzle,
    BM: Int,
    BK: Int,
]() -> IntTuple:
    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        alias k_bytes = BK * sizeof[type]()
        constrained[
            k_bytes == swizzle_mode.bytes(),
            "K dim "
            + String(k_bytes)
            + " doesn't match "
            + String(swizzle_mode),
        ]()
        # swizzled WGMMA cannot be tiled in K if we constraint the layout to 2D.

    return IntTuple(BM, BK)


fn tile_layout_k_major[
    type: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    alias atom = select_k_atom[type, swizzle_mode]()
    alias new_shape = _checked_tile_shape[type, swizzle_mode, BM, BK]()
    return tile_to_shape(atom, new_shape)


fn tile_to_descriptor[
    type: DType,
    layout: Layout,
    is_k_major: Bool = True,
]() -> Layout:
    @parameter
    if is_k_major:
        # Tile a layout to ((8,m),(T,2)) shape to match the K-major wgmma descriptor
        alias T = _CM_K_BYTES // sizeof[type]()
        alias tiler = MakeLayoutList(Layout(8), Layout(T))
        return logical_divide(layout, tiler)
    else:
        # We are not using atom layout for MN-major layouts.
        return layout


fn tile_layout_mn_major[
    type: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    """Return the shared memory layout for mn-major input.

    This is the "unit" layout, the actual shared memory layout can be multiple
    of this unit.
    """
    constrained[
        swizzle_mode
        in (TensorMapSwizzle.SWIZZLE_NONE, TensorMapSwizzle.SWIZZLE_128B),
        "Only support 128B and no swizzle",
    ]()

    @parameter
    if swizzle_mode == TensorMapSwizzle.SWIZZLE_128B:
        # See comments in file header.
        alias row_len = swizzle_mode.bytes() // sizeof[type]()
        return Layout(
            IntTuple(
                IntTuple(row_len, mn_dim // row_len),
                IntTuple(_CM_NUM_ROWS, k_dim // _CM_NUM_ROWS),
            ),
            IntTuple(
                IntTuple(1, _CM_NUM_ROWS * row_len),
                IntTuple(row_len, _CM_NUM_ROWS * mn_dim),
            ),
        )

    # No swizzle
    # Number of elements per row in core matrix
    alias _CM_ROW_LEN = _CM_ROW_BYTES // sizeof[type]()
    return Layout(
        IntTuple(
            IntTuple(_CM_ROW_LEN, mn_dim // _CM_ROW_LEN),
            IntTuple(_CM_NUM_ROWS, k_dim // _CM_NUM_ROWS),
        ),
        IntTuple(
            IntTuple(1, _CM_NUM_ROWS * _CM_ROW_LEN),
            IntTuple(_CM_ROW_LEN, _CM_NUM_ROWS * mn_dim),
        ),
    )


fn _wgmma_descriptor[
    type: DType, //,
    layout: Layout,
    is_k_major: Bool = True,
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    addr: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED]
) -> WGMMADescriptor[type]:
    # Conform to canonical layout.
    constrained[
        layout.rank() == 2 and layout[0].rank() == 2 and layout[1].rank() == 2,
        "shared memory tile layout should have structure (rank-2, rank-2).",
    ]()

    alias shape00 = layout[0].shape[0].value()
    alias shape11 = layout[1].shape[1].value()
    alias stride01 = layout[0].stride[1].value()
    alias stride11 = layout[1].stride[1].value()

    @parameter
    if is_k_major:
        constrained[
            shape00 == 8 and shape11 % 2 == 0,
            "Tile shape must be ((8, _), (_, multiple of 2)), get "
            + String(layout),
        ]()

        # Ingore 4 LSB.
        alias SBO = (stride01 * sizeof[type]()) >> 4
        alias LBO = (stride11 * sizeof[type]()) >> 4

        return WGMMADescriptor.create[SBO, LBO, swizzle](addr)

    alias no_swizzle = swizzle == TensorMapSwizzle.SWIZZLE_NONE

    # Swizzle and non-swizzle modes switch SBO and LBO based on
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=bar%2520sync#asynchronous-warpgroup-level-majorness-supported-by-strides
    alias SBO = ((stride01 if no_swizzle else stride11) * sizeof[type]()) >> 4
    alias LBO = ((stride11 if no_swizzle else stride01) * sizeof[type]()) >> 4

    return WGMMADescriptor.create[SBO, LBO, swizzle](addr)


fn _lhs_descriptor[
    mma_shape: IndexList[3],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    alias flat_layout = __type_of(tensor).layout
    alias layout = tile_to_descriptor[tensor.dtype, flat_layout, True]()
    constrained[
        layout.rank() == 2 and layout[0].rank() == 2 and layout[1].rank() == 2,
        "shared memory tile layout should have structure (rank-2, rank-2). But"
        " got "
        + String(layout),
    ]()

    alias shape00 = layout[0].shape[0].value()
    alias shape11 = layout[1].shape[1].value()

    # General constraints for all swizzle types.
    constrained[
        shape00 == 8 and shape11 % 2 == 0,
        "Tile shape must be ((8, _), (_, multiple of 2)). But got "
        + String(layout),
    ]()

    alias type = __type_of(tensor).dtype
    alias stride10 = layout[1].stride[0].value()

    # Constraints for swizzle
    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        alias stride_bytes = stride10 * sizeof[type]()
        constrained[
            stride_bytes != swizzle_mode.bytes(),
            "Stride dim bytes "
            + String(stride_bytes)
            + " doesn't match "
            + String(swizzle_mode),
        ]()

    # Ingore 4 LSB.
    alias SBO = (layout[0].stride[1].value() * sizeof[type]()) >> 4
    alias LBO = (layout[1].stride[1].value() * sizeof[type]()) >> 4

    return WGMMADescriptor.create[SBO, LBO, swizzle_mode](tensor.ptr)


fn _rhs_descriptor[
    mma_shape: IndexList[3],
    transposed: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    # Transposed case is same to K-major A matrix.
    @parameter
    if transposed:
        return _lhs_descriptor[mma_shape, swizzle_mode](tensor)

    # Non-Transposed case is MN-major
    alias layout = tensor.layout
    alias stride01 = layout[0].stride[1].value()
    alias stride11 = layout[1].stride[1].value()

    alias type = tensor.dtype
    alias no_swizzle = swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE

    # Swizzle and non-swizzle modes switch SBO and LBO based on
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=bar%2520sync#asynchronous-warpgroup-level-majorness-supported-by-strides
    alias SBO = ((stride01 if no_swizzle else stride11) * sizeof[type]()) >> 4
    alias LBO = ((stride11 if no_swizzle else stride01) * sizeof[type]()) >> 4

    return WGMMADescriptor.create[SBO, LBO, swizzle_mode](tensor.ptr)


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _output_register_size[mma_shape: IndexList[3]]() -> Int:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()
    return mma_shape[0] * mma_shape[1] // 128


struct TensorCoreAsync[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    mma_shape: IndexList[3],
    /,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = False,
]:
    @always_inline
    fn __init__(out self):
        constrained[
            mma_shape in supported_mma_shape,
            String(
                "WGMMA operation of shape '", mma_shape, "' is not supported"
            ),
        ]()

    @staticmethod
    @always_inline
    fn wgmma[
        num_warp_groups: Int = 1
    ](
        a_smem_tile: LayoutTensor[
            a_type, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        b_smem_tile: LayoutTensor[
            b_type, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        c_reg_tile: LayoutTensor[
            c_type, _, address_space = AddressSpace.LOCAL, *_, **_
        ],
        wg_idx: Int = 0,
    ):
        alias a_smem_layout = a_smem_tile.layout
        alias b_smem_layout = b_smem_tile.layout

        # TODO: refactor once atom layout is simplified
        alias BM = a_smem_layout[0].size()
        alias BN = b_smem_layout[0].size()
        alias BK = a_smem_layout[1].size()

        # Canonical layouts conform to WGMMA's layout requirement e.g.
        # K-major layout requires BK = swizzle.bytes() // sizeof[T}().
        alias a_canonical_K = a_swizzle.bytes() // sizeof[
            a_type
        ]() if a_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        alias a_canonical_layout_flat = tile_layout_k_major[
            a_type, BM, a_canonical_K, a_swizzle
        ]()
        alias a_canonical_layout = tile_to_descriptor[
            a_type, a_canonical_layout_flat, True
        ]()
        alias b_canonical_K = b_swizzle.bytes() // sizeof[
            b_type
        ]() if b_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        alias b_canonical_layout_flat = tile_layout_k_major[
            b_type, BN, b_canonical_K, b_swizzle
        ]() if transpose_b else b_smem_layout
        alias b_canonical_layout = tile_to_descriptor[
            b_type, b_canonical_layout_flat, transpose_b
        ]()

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        # TODO: use layout algebra like `tile_to_shape` here.
        alias a_shape00 = a_canonical_layout[0].shape[0].value()
        alias a_stride01 = a_canonical_layout[0].stride[1].value()
        alias a_stride11 = a_canonical_layout[1].stride[1].value()
        alias b_shape00 = b_canonical_layout[0].shape[0].value()
        alias b_stride01 = b_canonical_layout[0].stride[1].value()
        alias b_stride11 = b_canonical_layout[1].stride[1].value()
        constrained[mma_shape[0] % a_shape00 == 0]()
        constrained[mma_shape[1] % b_shape00 == 0]()

        # fmt: off
        # Strides between WGMMA tiles
        alias a_m_stride = a_stride01 * (mma_shape[0] // a_shape00) * sizeof[a_type]()
        alias b_n_stride = b_stride01 * (mma_shape[1] // b_shape00) * sizeof[b_type]()
        # K dim is stepped by 2 core matrices.
        alias a_k_stride = a_stride11 * 2 * sizeof[a_type]()
        alias b_k_stride = b_stride11 * 2 * sizeof[b_type]()

        alias num_m_mmas = a_canonical_layout[0].size() // mma_shape[0] // num_warp_groups
        alias num_n_mmas = b_canonical_layout[0].size() // mma_shape[1]
        alias num_k_mmas = a_smem_layout[1].size() // mma_shape[2]

        # Number of wgmma per canonical layout. There can be multiple canonical layouts
        # per K dim e.g. BF16 128B swizzle has BK = 64 while input K = 128.
        alias a_num_k_mmas_per_tile = a_canonical_K // mma_shape[2]
        alias b_num_k_mmas_per_tile = b_canonical_K // mma_shape[2] if transpose_b else num_k_mmas
        # fmt: on

        # Vectorize each wgmma's fragment size.
        alias c_frag_size = mma_shape[0] * mma_shape[1] // 128
        c_frags = c_reg_tile.vectorize[1, c_frag_size]()
        constrained[
            __type_of(c_frags).layout.size() == num_m_mmas * num_n_mmas,
            String(
                "C fragments' size: ",
                __type_of(c_frags).layout.size(),
                " doesn't match the total number of wgmma: ",
                num_m_mmas * num_n_mmas,
                ".",
            ),
        ]()

        a_desc = _wgmma_descriptor[a_canonical_layout, True, a_swizzle](
            a_smem_tile.ptr
        )
        b_desc = _wgmma_descriptor[b_canonical_layout, transpose_b, b_swizzle](
            b_smem_tile.ptr
        )

        @parameter
        if num_warp_groups > 1:
            a_desc += a_m_stride * num_m_mmas * wg_idx

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                @parameter
                for k_mma in range(num_k_mmas):
                    # Offsets when K is multiple of canonical layouts.
                    alias a_offset_bytes = (
                        k_mma // a_num_k_mmas_per_tile
                    ) * a_canonical_layout.size() * sizeof[a_type]()
                    alias b_offset_bytes = (
                        k_mma // b_num_k_mmas_per_tile
                    ) * b_canonical_layout.size() * sizeof[
                        b_type
                    ]() if transpose_b else 0

                    alias a_k_mma = k_mma % a_num_k_mmas_per_tile
                    alias b_k_mma = k_mma % b_num_k_mmas_per_tile

                    alias a_offset = m_mma * a_m_stride + a_k_mma * a_k_stride + a_offset_bytes
                    alias b_offset = n_mma * b_n_stride + b_k_mma * b_k_stride + b_offset_bytes
                    a_desc_m = a_desc + a_offset
                    b_desc_n = b_desc + b_offset

                    c_frags[mma_id, 0] = wgmma_async[
                        mma_shape[0],
                        mma_shape[1],
                        mma_shape[2],
                        a_type=a_type,
                        b_type=b_type,
                        layout_b= "col" if transpose_b else "row",
                    ](a_desc_m, b_desc_n, c_frags[mma_id, 0])

    @staticmethod
    @always_inline
    fn wgmma(
        a_frag: LayoutTensor[
            a_type, _, address_space = AddressSpace.LOCAL, *_, **_
        ],
        b_smem_tile: LayoutTensor[
            b_type, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        c_reg_tile: LayoutTensor[
            c_type, _, address_space = AddressSpace.LOCAL, *_, **_
        ],
    ):
        # alias a_smem_layout = a_smem_tile.layout
        alias b_smem_layout = tile_to_descriptor[
            b_type, b_smem_tile.layout, transpose_b
        ]()

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        # TODO: use layout algebra like `tile_to_shape` here.
        alias b_shape00 = b_smem_layout[0].shape[0].value()
        alias b_stride01 = b_smem_layout[0].stride[1].value()
        alias b_stride11 = b_smem_layout[1].stride[1].value()
        # Strides between WGMMA tiles
        constrained[mma_shape[1] % b_shape00 == 0]()
        # fmt: off
        alias b_n_stride = b_stride01 * (mma_shape[1] // b_shape00) * sizeof[b_type]()
        # fmt: on
        # K dim is stepped by 2 core matrices.
        alias b_k_stride = b_stride11 * 2 * sizeof[b_type]()
        constrained[b_k_stride > 0]()

        alias num_n_mmas = b_smem_layout[0].size() // mma_shape[1]
        alias num_k_mmas = b_smem_layout[1].size() // mma_shape[2]
        alias num_m_mmas = a_frag.layout[0].shape[0].value() // num_k_mmas

        constrained[
            b_n_stride > 0 or (b_n_stride == 0 and num_n_mmas == 1),
            "b_smem_layout = " + String(b_smem_layout),
        ]()

        # Vectorize each wgmma's fragment size.
        alias a_frag_size = mma_shape[0] * mma_shape[2] // 128
        alias c_frag_size = mma_shape[0] * mma_shape[1] // 128
        a_frags = a_frag.vectorize[1, a_frag_size]()
        c_frags = c_reg_tile.vectorize[1, c_frag_size]()
        constrained[
            __type_of(c_frags).layout.size() == num_m_mmas * num_n_mmas,
            String(
                "C fragments' size: ",
                __type_of(c_frags).layout.size(),
                (
                    "\nDoesn't match the total number of wgmmas\n= num_m_mmas *"
                    " num_n_mmas: "
                ),
                num_m_mmas,
                " * ",
                num_n_mmas,
                ".\na_frag.layout[0].shape[0].value() = ",
                a_frag.layout[0].shape[0].value(),
                "\nnum_k_mmas = ",
                num_k_mmas,
            ),
        ]()

        b_desc = _rhs_descriptor[mma_shape, transpose_b, b_swizzle](b_smem_tile)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                @parameter
                for k_mma in range(num_k_mmas):
                    # a_desc_m = a_desc + m_mma * a_m_stride + k_mma * a_k_stride
                    alias offset = n_mma * b_n_stride + k_mma * b_k_stride
                    b_desc_n = b_desc + offset

                    c_frags[mma_id, 0] = wgmma_async[
                        mma_shape[0],
                        mma_shape[1],
                        mma_shape[2],
                        a_type=a_type,
                        b_type=b_type,
                        layout_b= "col" if transpose_b else "row",
                    ](
                        a_frags[m_mma + k_mma * num_m_mmas, 0],
                        b_desc_n,
                        c_frags[mma_id, 0],
                    )

    @staticmethod
    @always_inline
    fn arrive():
        wgmma_fence_aligned()

    @staticmethod
    @always_inline
    fn commit_group():
        wgmma_commit_group_sync()

    @staticmethod
    @always_inline
    fn wait_for_all():
        wgmma_wait_group_sync()
