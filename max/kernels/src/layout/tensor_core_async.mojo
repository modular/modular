# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Tensor Core Async Module.

This module provides high-performance abstractions for utilizing NVIDIA's Tensor Cores
to perform asynchronous matrix multiplication operations. It implements optimized memory
layouts and access patterns for efficient tensor core computations.

Key components:
- Layout creation functions for K-major and MN-major memory arrangements
- Swizzling support for improved memory access patterns
- WGMMA (Warp Group Matrix Multiply-Accumulate) descriptor generation
- TensorCoreAsync struct with methods for asynchronous matrix multiplication

The module supports various data types, matrix dimensions, and memory configurations,
enabling efficient implementation of deep learning primitives and other tensor operations
that can leverage hardware acceleration.

Performance features:
- Asynchronous execution model to overlap computation and memory access
- Support for different swizzling modes to optimize memory bandwidth
- Efficient register and shared memory utilization
- Support for multi-warp group execution

This implementation is specifically optimized for NVIDIA GPUs with Tensor Core support.
"""
from std.sys import size_of
from std.sys._assembly import inlined_assembly

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout.coord import Coord, CoordLike, Idx
from layout.tensor_engine import DefaultEngine, _layout_row_major
from layout import TileTensor
from layout.layout import (
    Layout,
    TensorLayout,
    row_major as tile_row_major,
)

from std.utils import StaticTuple
from std.utils.coord import Coord

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
# Consider core matrix cm_M x cm_K, where cm_M = 8 and cm_K = 16 // size_of[type]()
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
# cm_K = T = 16 // size_of[type]()
# When there is swizzle, there is the swizzle mode constraint:
# `swizzle_mode.bytes() = 16x = 16 xT/T = 16 xT/(16 / size_of[type]()) = xT * size_of[type]().
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
# T = 16B // size_of[type]()
# m = BM  // (2T or 4T or 8T)
# k = BK  // 8
#
# We simplify them to
#
# B32   : Swizzle<1,4,3> o smem_ptr o ((2T,m),(8,k)):((1,LBO),(2T,SBO))
# B64   : Swizzle<2,4,3> o smem_ptr o ((4T,m),(8,k)):((1,LBO),(4T,SBO))
# B128  : Swizzle<3,4,3> o smem_ptr o ((8T,m),(8,k)):((1,LBO),(8T,SBO))
#
# `2/4/8 * T` is generalized as `swizzle.bytes() // size_of[type]()`.


@inline(.always)
def _supported_mma_shape[
    mma_shape: Coord,
]() -> Bool:
    """Checks if a given MMA shape is supported for tensor core operations.

    This function validates the dimensions of the MMA shape against known tensor core
    operation constraints. It ensures that the shape is compatible with the expected
    dimensions for tensor core operations.

    Parameters:
        mma_shape: The shape of the MMA operation.

    Returns:
        `Bool` - True if the MMA shape is supported, False otherwise.
    """

    # Ideally this check should be input/output type dependent as mma_shape depends on input/output types
    # (https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmwgmmamma_async-nvvmwgmmammaasyncop).
    comptime match (Int(mma_shape[0]), Int(mma_shape[2])):
        case (64, 8):
            return (
                Int(mma_shape[1]) % 8 == 0
                and Int(mma_shape[1]) >= 8
                and Int(mma_shape[1]) <= 256
            )
        case (64, 16):
            return (
                Int(mma_shape[1]) % 8 == 0
                and Int(mma_shape[1]) >= 8
                and Int(mma_shape[1]) <= 256
            )
        case (64, 32):
            return (
                Int(mma_shape[1]) % 8 == 0
                and Int(mma_shape[1]) >= 8
                and Int(mma_shape[1]) <= 256
            )
        case _:
            return False


# Core matrix dimensions
# Each core matrix has 8 rows and 16 bytes per row.
comptime _CM_NUM_ROWS = 8


comptime _CM_ROW_BYTES = 16


@inline(.always)
def warpgroup_fence[
    accum_type: DType,
    //,
](
    accum: TileTensor[
        accum_type,
        address_space=.LOCAL,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
) where (accum.rank == accum.flat_rank == 2):
    """Fences Hopper WGMMA accumulator registers against code motion.

    The pointer walk requires contiguous scalar register storage so every
    accumulator register receives the code-motion fence exactly once.

    Parameters:
        accum_type: Element data type of the accumulator; must be float32.

    Args:
        accum: A static row-major scalar register tile in local memory.
    """
    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for warpgroup fence"
    comptime assert _layout_row_major[
        type_of(accum).LayoutType
    ](), "Native WGMMA fence requires contiguous static row-major registers"

    @inline(.always)
    def _warpgroup_fence_operand(reg: Scalar[accum_type]):
        inlined_assembly["", NoneType, constraints="+f", has_side_effect=True](
            reg
        )

    comptime num_regs = type_of(accum).LayoutType.static_product
    var accum_flat = accum.reshape(tile_row_major[num_regs]())
    comptime for i in range(num_regs):
        _warpgroup_fence_operand(accum_flat.load[invariant=False]((Idx[i],)))


@inline(.never)
def _check_k_fits_swizzle[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    BK: Int,
]():
    """Rejects a K extent that does not span a whole number of swizzle atoms.

    Parameters:
        dtype: Element data type of the tensor.
        swizzle_mode: Memory access pattern swizzling mode.
        BK: Size of the K dimension in the tile.

    Constraints:
        `BK` must cover a whole number of `swizzle_mode` atoms; the swizzled
        WGMMA tile cannot be split along K.
    """
    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime k_bytes = BK * size_of[dtype]()
        comptime assert (k_bytes % swizzle_mode.bytes()) == 0, String(
            t"K dim {k_bytes} doesn't match {swizzle_mode}"
        )


# Helper: swizzle width in elements for a given dtype and swizzle mode.
comptime _sw_K[
    dtype: DType, swizzle_mode: TensorMapSwizzle
] = swizzle_mode.bytes() // size_of[dtype]()


def _outer_k_stride_of[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool,
]() -> Int:
    """Returns the outer K stride of the K-major typed tile layout.

    When `BK == sw_K` (outer K dim has shape 1), the stride is 0 to match the
    compact layout produced by `tile_to_shape` / `tile_layout_k_major()`. The
    page-dense form never collapses it: its K atoms are page-contiguous, so
    crossing one atom row advances a whole page.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
        page_dense: Whether the tile uses the page-dense (chunk-inner) form.

    Returns:
        The stride, in elements, from one K swizzle atom to the next.

    Constraints:
        The page-dense form is only defined for `SWIZZLE_128B`.
    """
    comptime sw_K = _sw_K[dtype, swizzle_mode]
    comptime if page_dense:
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense k-major layout is only defined for SWIZZLE_128B"
        return _CM_NUM_ROWS * sw_K
    else:
        return 0 if BK == sw_K else BM * sw_K


def _outer_m_stride_of[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool,
]() -> Int:
    """Returns the outer M stride of the K-major typed tile layout.

    Follows the same shape-1 rule as `_outer_k_stride_of` in the chunk-outer
    form; the page-dense form steps over a whole page of K atoms.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
        page_dense: Whether the tile uses the page-dense (chunk-inner) form.

    Returns:
        The stride, in elements, from one MN swizzle-atom row to the next.

    Constraints:
        The page-dense form is only defined for `SWIZZLE_128B`.
    """
    comptime sw_K = _sw_K[dtype, swizzle_mode]
    comptime if page_dense:
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense k-major layout is only defined for SWIZZLE_128B"
        return (BK // sw_K) * _CM_NUM_ROWS * sw_K
    else:
        return 0 if BM == _CM_NUM_ROWS else _CM_NUM_ROWS * sw_K


def _mn_outer_k_stride_of[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool,
]() -> Int:
    """Returns the outer K stride of the MN-major typed tile layout.

    Parameters:
        dtype: Element data type of the tensor.
        mn_dim: Size of the MN dimension in the tile.
        k_dim: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
        page_dense: Whether the tile uses the page-dense (chunk-inner) form.

    Returns:
        The stride, in elements, from one K swizzle atom to the next.

    Constraints:
        The page-dense form is only defined for `SWIZZLE_128B`.
    """
    comptime if page_dense:
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense mn-major layout is only defined for SWIZZLE_128B"
        return _CM_NUM_ROWS * _sw_K[dtype, swizzle_mode]
    else:
        return _outer_k_stride_of[dtype, k_dim, mn_dim, swizzle_mode, False]()


def _mn_outer_m_stride_of[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool,
]() -> Int:
    """Returns the outer MN stride of the MN-major typed tile layout.

    Parameters:
        dtype: Element data type of the tensor.
        mn_dim: Size of the MN dimension in the tile.
        k_dim: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
        page_dense: Whether the tile uses the page-dense (chunk-inner) form.

    Returns:
        The stride, in elements, from one MN swizzle-atom row to the next.

    Constraints:
        The page-dense form is only defined for `SWIZZLE_128B`.
    """
    comptime if page_dense:
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense mn-major layout is only defined for SWIZZLE_128B"
        return _CM_NUM_ROWS * mn_dim
    else:
        return _outer_m_stride_of[dtype, k_dim, mn_dim, swizzle_mode, False]()


# Outer stride for the K dimension in tile_layout_k_major_typed.
comptime _outer_k_stride[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool = False,
] = _outer_k_stride_of[dtype, BM, BK, swizzle_mode, page_dense]()

# Outer stride for the M dimension in tile_layout_k_major_typed, following the
# same shape-1 rule as _outer_k_stride.
comptime _outer_m_stride[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    page_dense: Bool = False,
] = _outer_m_stride_of[dtype, BM, BK, swizzle_mode, page_dense]()


comptime tile_layout_k_major_typed[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    page_dense: Bool = False,
] = Layout(
    Coord(
        Coord(
            Idx[_CM_NUM_ROWS],
            Idx[BM // _CM_NUM_ROWS],
        ),
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[BK // _sw_K[dtype, swizzle_mode]],
        ),
    ),
    Coord(
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[_outer_m_stride[dtype, BM, BK, swizzle_mode, page_dense]],
        ),
        Coord(
            Idx[1],
            Idx[_outer_k_stride[dtype, BM, BK, swizzle_mode, page_dense]],
        ),
    ),
)
"""K-major typed Layout for tensor core operations.

Shape ``((CM, BM/CM), (sw_K, BK/sw_K))``, stride ``((sw_K, CM*sw_K), (1, BM*sw_K))``
where CM=8 and sw_K = swizzle_mode.bytes() / sizeof(dtype).
An outer dimension of extent 1 carries stride 0 (compact), matching
`tile_layout_k_major()`.

Parameters:
    dtype: Element data type of the tensor.
    BM: Size of the M dimension in the tile.
    BK: Size of the K dimension in the tile.
    swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
    page_dense: When `True`, build the page-dense (chunk-inner, row-major
        atoms) form that `tile_layout_k_major[page_dense=True]()` returns
        instead of the chunk-outer form; the outer strides are the only
        difference between the two. `SWIZZLE_128B` only (defaults to
        `False`).
"""


comptime tile_layout_mn_major_typed[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    page_dense: Bool = False,
] = Layout(
    Coord(
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[mn_dim // _sw_K[dtype, swizzle_mode]],
        ),
        Coord(
            Idx[_CM_NUM_ROWS],
            Idx[k_dim // _CM_NUM_ROWS],
        ),
    ),
    Coord(
        Coord(
            Idx[1],
            Idx[
                _mn_outer_k_stride_of[
                    dtype, mn_dim, k_dim, swizzle_mode, page_dense
                ]()
            ],
        ),
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[
                _mn_outer_m_stride_of[
                    dtype, mn_dim, k_dim, swizzle_mode, page_dense
                ]()
            ],
        ),
    ),
)
"""MN-major typed Layout for tensor core operations.

Equivalent to ``tile_layout_k_major_typed[dtype, k_dim, mn_dim, swizzle_mode].transpose()``,
except in the page-dense form, which is the native chunk-inner MN-major tile
that ``tile_layout_mn_major[page_dense=True]()`` returns (not a transpose).

Parameters:
    dtype: Element data type of the tensor.
    mn_dim: Size of the MN dimension.
    k_dim: Size of the K dimension.
    swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
    page_dense: When `True`, build the page-dense (chunk-inner, row-major
        atoms) form that `tile_layout_mn_major[page_dense=True]()` returns.
        `SWIZZLE_128B` only (defaults to `False`).
"""


# Scale-factor atom geometry, mirrored from `tile_sf_layout_k_major`. The
# equivalence test keeps the two definitions in step.
comptime _SF_ATOM_M0 = 32
comptime _SF_ATOM_M1 = 4
comptime _SF_ATOM_K = 4
comptime _SF_MN_GROUP_SIZE = _SF_ATOM_M0 * _SF_ATOM_M1
comptime _SF_ATOM_SIZE = _SF_ATOM_M0 * _SF_ATOM_M1 * _SF_ATOM_K


def _checked_sf_mn_tiles[BM: Int]() -> Int:
    """Returns the number of scale-factor atoms along the MN dimension.

    Parameters:
        BM: Size of the M dimension in the tile.

    Returns:
        The atom count `BM / 128`.

    Constraints:
        BM must be a positive multiple of the 128-row scale-factor group.
    """
    comptime is_grouped = BM > 0 and BM % _SF_MN_GROUP_SIZE == 0
    comptime assert is_grouped, "BM must be a positive multiple of 128"
    return BM // _SF_MN_GROUP_SIZE


def _checked_sf_k_tiles[BK: Int, SF_SCALE_SIZE: Int]() -> Int:
    """Returns the number of scale-factor atoms along the K dimension.

    Parameters:
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.

    Returns:
        The atom count `BK / (4 * SF_SCALE_SIZE)`.

    Constraints:
        BK must be a positive multiple of `4 * SF_SCALE_SIZE`.
    """
    comptime k_group = _SF_ATOM_K * SF_SCALE_SIZE
    comptime is_grouped = BK > 0 and BK % k_group == 0
    comptime assert (
        is_grouped
    ), "BK must be a positive multiple of 4*SF_SCALE_SIZE"
    return BK // k_group


comptime _sf_mn_tiles[BM: Int] = _checked_sf_mn_tiles[BM]()

comptime _sf_k_tiles[BK: Int, SF_SCALE_SIZE: Int] = _checked_sf_k_tiles[
    BK, SF_SCALE_SIZE
]()

comptime _sf_mn_stride[
    BM: Int, BK: Int, SF_SCALE_SIZE: Int
] = 0 if _sf_mn_tiles[BM] == 1 else _sf_k_tiles[
    BK, SF_SCALE_SIZE
] * _SF_ATOM_SIZE

comptime _sf_k_stride[BK: Int, SF_SCALE_SIZE: Int] = 0 if _sf_k_tiles[
    BK, SF_SCALE_SIZE
] == 1 else _SF_ATOM_SIZE


comptime tile_sf_layout_k_major_typed[
    BM: Int,
    BK: Int,
    SF_SCALE_SIZE: Int,
] = Layout(
    Coord(
        Coord(
            Idx[_SF_ATOM_M0],
            Idx[_sf_mn_tiles[BM]],
        ),
        Coord(
            Coord(Idx[_SF_ATOM_M1], Idx[_SF_ATOM_K]),
            Idx[_sf_k_tiles[BK, SF_SCALE_SIZE]],
        ),
    ),
    Coord(
        Coord(
            Idx[_SF_ATOM_M1 * _SF_ATOM_K],
            Idx[_sf_mn_stride[BM, BK, SF_SCALE_SIZE]],
        ),
        Coord(
            Coord(Idx[1], Idx[_SF_ATOM_M1]),
            Idx[_sf_k_stride[BK, SF_SCALE_SIZE]],
        ),
    ),
)
"""K-major typed Layout for tensor core scale factors.

Closed form of ``tile_sf_layout_k_major()``: the ``(32, (4, 4))`` scale-factor
atom repeated over ``BM/128`` MN atoms and ``BK/(4*SF_SCALE_SIZE)`` K atoms,
K-atom-minor. An atom count of 1 carries stride 0, matching the compact layout
produced by `tile_to_shape`.

Unlike `tile_sf_layout_k_major()`, which silently drops a mode when an atom
count rounds down to 0, this rejects tile extents that do not cover a whole
number of atoms.

Parameters:
    BM: Size of the M dimension in the tile.
    BK: Size of the K dimension in the tile.
    SF_SCALE_SIZE: Number of elements in a scale factor vector.
"""


comptime _ModeProduct[mode: CoordLike]: Int = Coord[
    *mode.ParamListType
].static_product
"""The element count of one layout mode, nested or flat."""


def wgmma_canonical_strides[
    dtype: DType,
    layout_t: TensorLayout,
    is_k_major: Bool,
    swizzle_mode: TensorMapSwizzle,
]() -> Tuple[Int, Int, Int]:
    """Returns the canonical descriptor layout's two sub-strides, in elements.

    The canonical layout an operand descriptor describes is the operand tile
    tiled by `(8, 16 B)` along K. For a K-major operand that layout follows from
    the tile geometry and the swizzle mode alone; for an MN-major operand it is
    the operand tile's own layout. `wgmma_desc_sbo_lbo` turns these into the
    WGMMA descriptor's 16 B units and applies WGMMA's majorness swap; the SM100
    descriptor builders take them in elements and do their own swap.

    Parameters:
        dtype: Element data type of the tensor.
        layout_t: Type of the operand tile's layout, with mode 0 the MN extent
            and mode 1 the K extent.
        is_k_major: Whether the operand is K-major (True) or MN-major (False).
        swizzle_mode: Memory access pattern swizzling mode.

    Returns:
        The canonical mode-0 shape, its second sub-stride and the mode-1 second
        sub-stride.

    Constraints:
        An MN-major operand needs two sub-modes per mode, the nesting the
        descriptor reads its sub-strides from.
    """
    comptime es = size_of[dtype]()
    comptime if is_k_major:
        # Tiling the operand by one core-matrix column (`_CM_NUM_ROWS` rows)
        # and one 16 B core-matrix row leaves `(8, ...)` in mode 0 with the
        # swizzle atom as its first sub-stride, and a 16 B row as mode 1's.
        # An unswizzled tile keeps its own outer K stride there, which is 0
        # when K is one core matrix wide.
        comptime BMN = _ModeProduct[layout_t._shape_types[0]]
        comptime BK = _ModeProduct[layout_t._shape_types[1]]
        comptime sw_K = swizzle_mode.bytes() // es
        comptime canonical_K = (
            BK if swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE else sw_K
        )
        comptime sub_stride01 = (
            0 if BMN == _CM_NUM_ROWS else _CM_NUM_ROWS * sw_K
        )
        comptime sub_stride11 = (
            (0 if canonical_K == sw_K else BMN * sw_K) if swizzle_mode
            == TensorMapSwizzle.SWIZZLE_NONE else _CM_ROW_BYTES // es
        )
        return (_CM_NUM_ROWS, sub_stride01, sub_stride11)

    comptime assert (
        layout_t.rank == 2
        and layout_t._shape_types[0].__len__() == 2
        and layout_t._shape_types[1].__len__() == 2
    ), "an MN-major tile layout should have structure (rank-2, rank-2)."
    return (
        layout_t.static_shape[0],
        layout_t.static_stride[1],
        layout_t.static_stride[3],
    )


def wgmma_desc_sbo_lbo[
    dtype: DType,
    layout_t: TensorLayout,
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Tuple[Int, Int]:
    """Returns the (stride, leading) byte offsets of a WGMMA operand descriptor.

    The two offsets are the sub-strides of the canonical descriptor layout, in
    16 B units: mode 0's second sub-stride is the stride byte offset and mode
    1's second sub-stride the leading byte offset. An MN-major operand swaps the
    two when the swizzle is off, per the PTX "asynchronous warpgroup-level
    majorness" table.

    Parameters:
        dtype: Element data type of the tensor.
        layout_t: Type of the operand tile's layout, with mode 0 the MN extent
            and mode 1 the K extent.
        is_k_major: Whether the operand is K-major (True) or MN-major (False).
        swizzle_mode: Memory access pattern swizzling mode.

    Returns:
        The stride byte offset (SBO) and leading byte offset (LBO).
    """
    comptime canonical = wgmma_canonical_strides[
        dtype, layout_t, is_k_major, swizzle_mode
    ]()
    comptime no_swizzle = swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE
    comptime es = size_of[dtype]()
    # Ignore the 4 LSB: descriptor offsets are in 16 B units.
    comptime sbo = (
        (canonical[1] if is_k_major or no_swizzle else canonical[2]) * es
    ) >> 4
    comptime lbo = (
        (canonical[2] if is_k_major or no_swizzle else canonical[1]) * es
    ) >> 4
    return (sbo, lbo)


comptime _st_matrix_frag_elems[c_type: DType] = 16 // size_of[c_type]()
"""Elements of `c_type` in one 128-bit `st_matrix` fragment."""


comptime st_matrix_n_layout[
    c_type: DType, WG_BN: Int, num_m_mmas: Int, num_consumer: Int
] = Layout(
    Coord(
        Coord(Idx[16], Idx[2], Idx[4]),
        Coord(Idx[WG_BN // 16], Idx[num_m_mmas], Idx[num_consumer]),
    ),
    Coord(
        Coord(
            Idx[2 * (WG_BN // 16) * _st_matrix_frag_elems[c_type]],
            Idx[_st_matrix_frag_elems[c_type]],
            Idx[32 * (WG_BN // 16) * _st_matrix_frag_elems[c_type]],
        ),
        Coord(
            Idx[2 * _st_matrix_frag_elems[c_type]],
            Idx[128 * (WG_BN // 16) * _st_matrix_frag_elems[c_type]],
            Idx[
                128 * (WG_BN // 16) * num_m_mmas * _st_matrix_frag_elems[c_type]
            ],
        ),
    ),
)
"""Layout for N-major `st_matrix` in the context of the WGMMA C matrix.

The modes are the warp group local thread index, decomposed as
`(16, 2, 4)`, and `(WG_BN // 16, num_m_mmas, num_consumer)`: the N tile,
the MMA tile in M and the consumer warp group. The layout maps a coordinate
to an element offset of `c_type`.

Each 128-bit fragment atom is a `(64, 2 * WG_BN // 16)` row-major grid of
fragments: thread `(t0, t1, t2)` owns fragment `(t0 + 16 * t2, t1)`. The
outer modes step by one fragment column, by one atom (128 fragments) and by
`WG_BN // 16` atoms respectively.

Parameters:
    c_type: Data type of the C matrix.
    WG_BN: Size of the N dimension of the C matrix in shared memory.
    num_m_mmas: Number of MMA tiles in the M dimension.
    num_consumer: Number of consumer warp groups.
"""


comptime st_matrix_m_layout[
    c_type: DType, WG_BM: Int, num_m_mmas: Int, num_consumer: Int
] = Layout(
    Coord(
        Coord(Idx[8], Idx[2], Idx[2], Idx[4]),
        Coord(Idx[WG_BM // 16], Idx[num_m_mmas], Idx[num_consumer]),
    ),
    Coord(
        Coord(
            Idx[8 * num_consumer * _st_matrix_frag_elems[c_type]],
            Idx[_st_matrix_frag_elems[c_type]],
            Idx[64 * num_consumer * _st_matrix_frag_elems[c_type]],
            Idx[2 * _st_matrix_frag_elems[c_type]],
        ),
        Coord(
            Idx[
                128 * num_consumer * num_m_mmas * _st_matrix_frag_elems[c_type]
            ],
            Idx[128 * num_consumer * _st_matrix_frag_elems[c_type]],
            Idx[8 * _st_matrix_frag_elems[c_type]],
        ),
    ),
)
"""Layout for M-major `st_matrix` in the context of the WGMMA C matrix.

This is for swapAB, where the C matrix is transposed during the write phase,
and must be paired with the transposed `st_matrix` modifier. The modes are
the warp group local thread index, decomposed as `(8, 2, 2, 4)`, and
`(WG_BM // 16, num_m_mmas, num_consumer)`: the M tile, the MMA tile in N and
the consumer warp group. The layout maps a coordinate to an element offset
of `c_type`.

With `num_consumer = 2` and `WG_BM // 16 = 2`, one atom of one warp holds
these thread indices, each owning the next 128-bit fragment:

```
|  0  |  8  |
|  1  |  9  |
| ... | ... |
|  7  | 15  |

| 16  | 24  |
| ... | ... |
| 23  | 31  |
```

The four warps of a warp group are laid out next to each other.

Parameters:
    c_type: Data type of the C matrix.
    WG_BM: Size of the M dimension of the C matrix in shared memory.
    num_m_mmas: Number of MMA tiles in the N dimension.
    num_consumer: Number of consumer warp groups.
"""


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction


@inline(.always)
def _convert_cfrags_to_tuple[
    c_type: DType, c_frag_size: Int
](
    c_frags: TileTensor[
        c_type,
        address_space=.LOCAL,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
) -> StaticTuple[Scalar[c_type], c_frag_size] where (
    c_frags.rank == c_frags.flat_rank == 2
):
    comptime assert _layout_row_major[type_of(c_frags).LayoutType]()
    comptime assert c_frags.static_shape[0] == 1
    comptime assert c_frags.static_shape[1] == c_frag_size
    var c_frags_in_tuple = StaticTuple[Scalar[c_type], c_frag_size]()
    comptime for i in range(c_frag_size):
        c_frags_in_tuple[i] = c_frags[0, i]
    return c_frags_in_tuple


@inline(.always)
def _convert_cfrags_to_simd[
    c_type: DType, c_frag_size: Int
](
    c_frags_in_tuple: StaticTuple[Scalar[c_type], c_frag_size],
    c_frags: TileTensor[
        mut=True,
        c_type,
        address_space=.LOCAL,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
) where (c_frags.rank == c_frags.flat_rank == 2):
    comptime assert _layout_row_major[type_of(c_frags).LayoutType]()
    comptime assert c_frags.static_shape[0] == 1
    comptime assert c_frags.static_shape[1] == c_frag_size
    comptime for i in range(c_frag_size):
        c_frags[0, i] = c_frags_in_tuple[i]


struct TensorCoreAsync[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    mma_shape: Coord,
    /,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = False,
](Defaultable):
    """High-performance asynchronous tensor core operations for matrix multiplication.

    This struct provides methods for utilizing NVIDIA's Tensor Cores for asynchronous
    matrix multiplication operations, with support for various data types and swizzling
    configurations.

    Parameters:
        c_type: Data type of the output matrix C.
        a_type: Data type of the input matrix A.
        b_type: Data type of the input matrix B.
        mma_shape: Dimensions for the matrix multiply-accumulate (MMA) operation as [M, N, K].
        a_swizzle: Swizzling mode for matrix A (default: SWIZZLE_NONE).
        b_swizzle: Swizzling mode for matrix B (default: SWIZZLE_NONE).
        transpose_b: Whether to transpose matrix B (default: False).
    """

    @inline(.always)
    def __init__(out self):
        """Initialize the `TensorCoreAsync` instance.

        Ensures that the provided MMA shape is supported.

        Note:
            Fails to compile if `mma_shape` is not supported.
        """
        comptime assert _supported_mma_shape[Self.mma_shape](), String(
            t"WGMMA operation of shape '{Self.mma_shape}' is not supported"
        )

    @staticmethod
    @inline(.always)
    def wgmma[
        num_warp_groups: Int = 1,
        scale_c: Int = 1,
        scale_a: Int = 1,
        scale_b: Int = 1,
        num_k_iters: Optional[Int] = None,
    ](
        a_smem_tile: TileTensor[
            mut=True, Self.a_type, address_space=.SHARED, ...
        ],
        b_smem_tile: TileTensor[
            mut=True, Self.b_type, address_space=.SHARED, ...
        ],
        c_reg_tile: TileTensor[
            mut=True,
            Self.c_type,
            address_space=.LOCAL,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        wg_idx: Int = 0,
    ) where (c_reg_tile.rank == c_reg_tile.flat_rank == 2):
        """Performs Hopper WGMMA with native shared-memory and register tiles.

        Descriptor metadata keeps the exact nested shape and stride the
        descriptor formulas expect; the tensors stay native. C must be a contiguous row-major tile with one row per MMA and one
        scalar per fragment register, matching the tuple lane order.

        Parameters:
            num_warp_groups: Number of warp groups to distribute work across (default: 1).
            scale_c: Scale factor for matrix C. Valid values are 1 or 0 (default: 1).
            scale_a: Scale factor for matrix A. Valid values are 1 or -1 (default: 1).
            scale_b: Scale factor for matrix B. Valid values are 1 or -1 (default: 1).
            num_k_iters: Number of iterations for the K dimension. This is useful to save computation when we pad shared memory. (default: None, which is the A tile's K extent over `mma_shape[2]`).

        Args:
            a_smem_tile: Matrix A in shared memory.
            b_smem_tile: Matrix B in shared memory.
            c_reg_tile: Output matrix C in register memory.
            wg_idx: Warp group index for multi-warp group scenarios (default: 0).
        """
        comptime assert scale_c == 1 or scale_c == 0
        comptime assert scale_a == 1 or scale_a == -1
        comptime assert scale_b == 1 or scale_b == -1
        comptime assert a_smem_tile.element_size == 1
        comptime assert b_smem_tile.element_size == 1
        comptime assert a_smem_tile.all_dims_known
        comptime assert b_smem_tile.all_dims_known
        comptime assert _layout_row_major[
            type_of(c_reg_tile).LayoutType
        ](), "Native WGMMA requires contiguous static row-major C registers"
        comptime a_smem_layout_t = type_of(a_smem_tile).LayoutType
        comptime b_smem_layout_t = type_of(b_smem_tile).LayoutType

        comptime BM = _ModeProduct[a_smem_layout_t._shape_types[0]]
        comptime BN = _ModeProduct[b_smem_layout_t._shape_types[0]]
        comptime BK = _ModeProduct[a_smem_layout_t._shape_types[1]]

        # Canonical layouts conform to WGMMA's layout requirement e.g.
        # K-major layout requires BK = swizzle.bytes() // size_of[T]().
        comptime a_canonical_K = Self.a_swizzle.bytes() // size_of[
            Self.a_type
        ]() if Self.a_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        comptime b_canonical_K = Self.b_swizzle.bytes() // size_of[
            Self.b_type
        ]() if Self.b_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        comptime a_canonical = wgmma_canonical_strides[
            Self.a_type, a_smem_layout_t, True, Self.a_swizzle
        ]()
        comptime b_canonical = wgmma_canonical_strides[
            Self.b_type, b_smem_layout_t, Self.transpose_b, Self.b_swizzle
        ]()
        comptime a_shape00 = a_canonical[0]
        comptime b_shape00 = b_canonical[0]
        comptime assert Int(Self.mma_shape[0]) % a_shape00 == 0
        comptime assert Int(Self.mma_shape[1]) % b_shape00 == 0

        # fmt: off
        # Strides between WGMMA tiles
        comptime a_m_stride = a_canonical[1] * (Int(Self.mma_shape[0]) // a_shape00) * size_of[Self.a_type]()
        comptime b_n_stride = b_canonical[1] * (Int(Self.mma_shape[1]) // b_shape00) * size_of[Self.b_type]()
        # K dim is stepped by 2 core matrices.
        comptime a_k_stride = a_canonical[2] * 2 * size_of[Self.a_type]()
        comptime b_k_stride = b_canonical[2] * 2 * size_of[Self.b_type]()

        comptime num_m_mmas = BM // Int(Self.mma_shape[0]) // num_warp_groups
        comptime num_n_mmas = BN // Int(Self.mma_shape[1])
        comptime num_k_mmas = num_k_iters.or_else(BK // Int(Self.mma_shape[2]))

        # Number of wgmma per canonical layout. There can be multiple canonical layouts
        # per K dim e.g. BF16 128B swizzle has BK = 64 while input K = 128.
        comptime a_num_k_mmas_per_tile = a_canonical_K // Int(Self.mma_shape[2])
        comptime b_num_k_mmas_per_tile = b_canonical_K // Int(Self.mma_shape[2]) if Self.transpose_b else num_k_mmas
        # fmt: on

        comptime a_sbo_lbo = wgmma_desc_sbo_lbo[
            Self.a_type,
            a_smem_layout_t,
            is_k_major=True,
            swizzle_mode=Self.a_swizzle,
        ]()
        comptime b_sbo_lbo = wgmma_desc_sbo_lbo[
            Self.b_type,
            b_smem_layout_t,
            is_k_major=Self.transpose_b,
            swizzle_mode=Self.b_swizzle,
        ]()
        var a_desc = WGMMADescriptor.create[
            a_sbo_lbo[0], a_sbo_lbo[1], Self.a_swizzle
        ](a_smem_tile.ptr)
        var b_desc = WGMMADescriptor.create[
            b_sbo_lbo[0], b_sbo_lbo[1], Self.b_swizzle
        ](b_smem_tile.ptr)

        comptime if num_warp_groups > 1:
            a_desc += a_m_stride * num_m_mmas * wg_idx

        comptime layout_b = "col" if Self.transpose_b else "row"
        comptime c_frag_size = Int(Self.mma_shape[0]) * Int(
            Self.mma_shape[1]
        ) // 128
        comptime assert c_reg_tile.static_shape[0] == num_m_mmas * num_n_mmas
        comptime assert c_reg_tile.static_shape[1] == c_frag_size

        comptime for k_mma in range(num_k_mmas):
            comptime scale_d = scale_c if k_mma == 0 else 1

            # Offsets when K is multiple of canonical layouts. A canonical
            # layout is one block tile wide and `canonical_K` deep.
            comptime a_offset_bytes = (
                k_mma // a_num_k_mmas_per_tile
            ) * BM * a_canonical_K * size_of[Self.a_type]()
            comptime b_offset_bytes = (
                k_mma // b_num_k_mmas_per_tile
            ) * BN * b_canonical_K * size_of[
                Self.b_type
            ]() if Self.transpose_b else 0

            comptime a_k_mma_offset = (
                k_mma % a_num_k_mmas_per_tile
            ) * a_k_stride
            comptime b_k_mma_offset = (
                k_mma % b_num_k_mmas_per_tile
            ) * b_k_stride

            comptime for m_mma in range(num_m_mmas):
                comptime a_offset = m_mma * a_m_stride + a_k_mma_offset + a_offset_bytes
                var a_desc_m = a_desc + a_offset

                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    comptime b_offset = n_mma * b_n_stride + b_k_mma_offset + b_offset_bytes
                    var b_desc_n = b_desc + b_offset

                    var c_frags = c_reg_tile.tile[1, c_frag_size](mma_id, 0)
                    comptime assert c_frags.rank == c_frags.flat_rank == 2

                    var c_frags_in_tuple = _convert_cfrags_to_tuple[
                        Self.c_type, c_frag_size
                    ](c_frags)

                    var c_frags_out_tuple = wgmma_async[
                        Int(Self.mma_shape[0]),
                        Int(Self.mma_shape[1]),
                        Int(Self.mma_shape[2]),
                        a_type=Self.a_type,
                        b_type=Self.b_type,
                        layout_b=layout_b,
                        scale_d=scale_d,
                        scale_a=scale_a,
                        scale_b=scale_b,
                    ](a_desc_m, b_desc_n, c_frags_in_tuple)

                    _convert_cfrags_to_simd[Self.c_type, c_frag_size](
                        c_frags_out_tuple, c_frags
                    )

    @staticmethod
    @inline(.always)
    def wgmma(
        a_frag_tile: TileTensor[Self.a_type, address_space=.LOCAL, ...],
        b_smem_tile: TileTensor[
            mut=True, Self.b_type, address_space=.SHARED, ...
        ],
        c_reg_tile: TileTensor[
            mut=True,
            Self.c_type,
            address_space=.LOCAL,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
    ) where (
        a_frag_tile.rank == a_frag_tile.flat_rank == 2
        and c_reg_tile.rank == c_reg_tile.flat_rank == 2
    ):
        """Performs Hopper WGMMA with A in registers and B in shared memory.

        Row `m_mma + k_mma * num_m_mmas` of `a_frag_tile` holds the A
        fragment for that MMA, and row `n_mma * num_m_mmas + m_mma` of
        `c_reg_tile` holds the accumulator, both as contiguous row-major
        scalar register storage.

        Args:
            a_frag_tile: Matrix A in register memory, one row per A fragment.
            b_smem_tile: Matrix B in shared memory.
            c_reg_tile: Output matrix C in register memory, one row per MMA.
        """
        comptime assert a_frag_tile.element_size == 1
        comptime assert b_smem_tile.element_size == 1
        comptime assert b_smem_tile.all_dims_known
        comptime assert _layout_row_major[type_of(a_frag_tile).LayoutType]()
        comptime assert _layout_row_major[type_of(c_reg_tile).LayoutType]()
        comptime b_smem_layout_t = type_of(b_smem_tile).LayoutType
        comptime BN = _ModeProduct[b_smem_layout_t._shape_types[0]]
        comptime BK = _ModeProduct[b_smem_layout_t._shape_types[1]]
        comptime b_canonical_K = Self.b_swizzle.bytes() // size_of[
            Self.b_type
        ]() if Self.b_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        comptime b_canonical = wgmma_canonical_strides[
            Self.b_type, b_smem_layout_t, Self.transpose_b, Self.b_swizzle
        ]()
        comptime b_shape00 = b_canonical[0]
        comptime assert Int(Self.mma_shape[1]) % b_shape00 == 0, String(
            t"b_shape00 = {b_shape00}, Int(mma_shape[1]) = "
            t"{Int(Self.mma_shape[1])}"
        )
        # fmt: off
        comptime b_n_stride = b_canonical[1] * (Int(Self.mma_shape[1]) // b_shape00) * size_of[Self.b_type]()
        # K dim is stepped by 2 core matrices.
        comptime b_k_stride = b_canonical[2] * 2 * size_of[Self.b_type]()
        comptime assert b_k_stride > 0

        comptime num_n_mmas = BN // Int(Self.mma_shape[1])
        comptime num_k_mmas = BK // Int(Self.mma_shape[2])
        comptime num_m_mmas = a_frag_tile.static_shape[0] // num_k_mmas

        comptime b_num_k_mmas_per_tile = b_canonical_K // Int(Self.mma_shape[2]) if Self.transpose_b else num_k_mmas
        # fmt: on

        comptime assert b_n_stride > 0 or (
            b_n_stride == 0 and num_n_mmas == 1
        ), String(t"b_n_stride = {b_n_stride}, num_n_mmas = {num_n_mmas}")

        # Vectorize each wgmma's fragment size.
        comptime a_frag_size = Int(Self.mma_shape[0]) * Int(
            Self.mma_shape[2]
        ) // 128
        comptime c_frag_size = Int(Self.mma_shape[0]) * Int(
            Self.mma_shape[1]
        ) // 128
        comptime assert a_frag_tile.static_shape[1] == a_frag_size
        comptime assert c_reg_tile.static_shape[0] == num_m_mmas * num_n_mmas
        comptime assert c_reg_tile.static_shape[1] == c_frag_size
        var a_frags = a_frag_tile.vectorize[1, a_frag_size]()
        var c_frags = c_reg_tile.vectorize[1, c_frag_size]()

        comptime b_sbo_lbo = wgmma_desc_sbo_lbo[
            Self.b_type,
            b_smem_layout_t,
            is_k_major=Self.transpose_b,
            swizzle_mode=Self.b_swizzle,
        ]()
        var b_desc = WGMMADescriptor.create[
            b_sbo_lbo[0], b_sbo_lbo[1], Self.b_swizzle
        ](b_smem_tile.ptr)
        comptime layout_b = "col" if Self.transpose_b else "row"

        comptime for k_mma in range(num_k_mmas):
            comptime b_offset_bytes = (
                k_mma // b_num_k_mmas_per_tile
            ) * BN * b_canonical_K * size_of[
                Self.b_type
            ]() if Self.transpose_b else 0
            comptime b_k_mma_offset = (
                k_mma % b_num_k_mmas_per_tile
            ) * b_k_stride

            comptime for m_mma in range(num_m_mmas):
                var a_frag = a_frags[m_mma + k_mma * num_m_mmas, 0]

                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma
                    comptime offset = n_mma * b_n_stride + b_k_mma_offset + b_offset_bytes
                    var b_desc_n = b_desc + offset

                    c_frags[mma_id, 0] = wgmma_async[
                        Int(Self.mma_shape[0]),
                        Int(Self.mma_shape[1]),
                        Int(Self.mma_shape[2]),
                        a_type=Self.a_type,
                        b_type=Self.b_type,
                        layout_b=layout_b,
                    ](
                        a_frag,
                        b_desc_n,
                        c_frags[mma_id, 0],
                    )

    @staticmethod
    @inline(.always)
    def arrive():
        """Ensures memory consistency by creating a fence for WGMMA operations.

        This method should be called before committing a group to ensure all
        shared memory accesses are properly aligned and visible.
        """
        wgmma_fence_aligned()

    @staticmethod
    @inline(.always)
    def commit_group():
        """Commits the current warp group for execution.

        This synchronizes the warp group and commits all pending WGMMA operations
        that have been previously issued.
        """
        wgmma_commit_group_sync()

    @staticmethod
    @inline(.always)
    def wait_group[group: Int = 0]():
        """Waits for the completion of a specific warp group's operations.

        This method blocks until all WGMMA operations from the specified group are complete.

        Parameters:
            group: The group ID to wait for (default: 0).
        """
        wgmma_wait_group_sync[group]()
