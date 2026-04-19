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
"""MMA and data-movement helpers for AMD matmul kernels.

Structs:
    TiledMma: Stateless MMA computation on TileTensors (mirrors
        TiledTensorCore.mma). Pure computation, no register ownership.
    MmaOp: Register ownership + SMEM loading + schedule API. Wraps
        TiledMma for per-k-tile load_frag/mma dispatch.
    QuadrantMmaOp: Owns A/B/C register tiles in LOCAL, provides quadrant
        load/compute methods for ping-pong double-buffering schedule.
    TileLoaderLDS: Cooperative global→LDS loader via buffer_load_to_lds.
"""

from std.sys import simd_width_of, size_of
from std.gpu.compute.mma import mma as gpu_mma
from std.gpu import lane_id, WARP_SIZE
from std.gpu.intrinsics import AMDBufferResource
from std.gpu._utils import to_i64
from std.sys.intrinsics import readfirstlane
from std.memory.unsafe import bitcast
from std.math import min
from std.utils import IndexList
from layout import TensorLayout, TileTensor
from layout._utils import make_amd_buffer_resource
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from layout.tile_layout import Layout, row_major, col_major
from layout.tile_tensor import stack_allocation
from structured_kernels.amd_tile_io import GMemTile, SMemTile, RegTile


# ===----------------------------------------------------------------------=== #
# TileLoaderLDS: Cooperative Global→LDS loader
# ===----------------------------------------------------------------------=== #


struct TileLoaderLDS[
    dtype: DType,
    tile_rows: Int,
    tile_cols: Int,
    stride: Int,
    num_loading_warps: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    load_width: Int = simd_width_of[dtype](),
    use_full_tile_width: Bool = False,
](TrivialRegisterPassable):
    """Cooperative global→LDS loader via load_to_lds.

    Cooperative global→LDS loader using AMDBufferResource.load_to_lds
    for direct DRAM→LDS DMA with OOB clamping.

    Parameters:
        dtype: Element data type.
        tile_rows: Height of each half-tile to load.
        tile_cols: Width (K dimension) of each half-tile.
        stride: Row stride of the source GMEM tensor.
        num_loading_warps: Warps cooperating on each load (typically 8).
        swizzle: Optional byte-space swizzle for LDS bank conflicts.
        load_width: Elements per load (SIMD width).
        use_full_tile_width: FP8 row-major mode.
    """

    comptime subtile_cols = Self.tile_cols if Self.use_full_tile_width else 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width
    comptime thread_rows = WARP_SIZE // Self.threads_per_row

    comptime num_warp_cols = Self.tile_cols // Self.subtile_cols
    comptime num_warp_rows = Self.num_loading_warps // Self.num_warp_cols

    comptime elements_per_warp = WARP_SIZE * Self.load_width
    comptime rows_per_warp = Self.elements_per_warp // Self.tile_cols

    comptime loading_threads = Self.num_loading_warps * WARP_SIZE
    comptime rows_per_iteration = Self.loading_threads // (
        Self.tile_cols // Self.load_width
    )
    comptime num_iterations = Self.tile_rows // Self.rows_per_iteration

    comptime warp_subtile_bytes = Self.rows_per_warp * Self.tile_cols * size_of[
        Self.dtype
    ]()
    comptime lane_load_bytes = Self.load_width * size_of[Self.dtype]()
    comptime row_bytes = Self.tile_cols * size_of[Self.dtype]()

    comptime _needs_per_iter_swizzle = Bool(
        Self.swizzle
    ) and Self.use_full_tile_width

    var buffer: AMDBufferResource
    var thread_row: Int
    var thread_col: Int
    var warp_id: Int
    var lane_id: Int

    @always_inline
    def __init__(
        out self,
        src: GMemTile[Self.dtype, _, _],
        warp_id: Int,
        lane_id: Int,
    ):
        """Build from a GMEM tile (block-level A or B tile)."""
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id
        self.lane_id = lane_id

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[Self.dtype, _, _],
        src_row: Int,
        src_col: Int,
    ):
        """Load from GMEM at (src_row, src_col) into SMEM dst via load_to_lds.

        Args:
            dst: Destination TileTensor in SHARED (half-tile sized).
            src_row: Row offset in the block's GMEM tile.
            src_col: Column (K) offset.
        """
        comptime SmemPtr = UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]

        comptime if Self._needs_per_iter_swizzle:
            var lane_byte = self.lane_id * Self.lane_load_bytes

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var full_byte = tile_idx * Self.warp_subtile_bytes + lane_byte
                var swizzled_byte = Self.swizzle.value()(full_byte)

                var swizzled_row = swizzled_byte // Self.row_bytes
                var swizzled_col = (swizzled_byte % Self.row_bytes) // size_of[
                    Self.dtype
                ]()

                var lane_offset = swizzled_col + swizzled_row * Self.stride
                var uniform_offset = src_col + src_row * Self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )
        else:
            var lane_offset = self.thread_col + self.thread_row * Self.stride

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var tile_row = src_row + i * Self.rows_per_iteration
                var uniform_offset = src_col + tile_row * Self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )


# ===----------------------------------------------------------------------=== #
# _load_from_lds: Alias-scoped LDS load
# ===----------------------------------------------------------------------=== #


@always_inline
def _load_from_lds[
    dtype: DType,
    //,
    width: Int = 1,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype], _, address_space=AddressSpace.SHARED
    ],
) -> SIMD[dtype, width]:
    """Alias-scoped LDS load via LLVM intrinsic with noalias annotations."""
    comptime alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
    comptime no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)

    comptime load_bytes = width * size_of[dtype]()
    comptime alignment = min(load_bytes, 16)

    comptime if dtype == DType.bfloat16 and width == 4:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<4 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == DType.bfloat16 and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype.is_float8() and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 8]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 16:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 32:
        var llvm_res0 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var shared_ptr_offset = shared_ptr + 16
        var shared_ptr3_hi = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](shared_ptr_offset)
        var llvm_res1 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3_hi)
        var uint8_vec0 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res0)
        var uint8_vec1 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res1)
        var uint8_vec = rebind[SIMD[DType.uint8, 16]](uint8_vec0).join(
            rebind[SIMD[DType.uint8, 16]](uint8_vec1)
        )
        return bitcast[dtype, width](uint8_vec)
    else:
        comptime assert False, "Unsupported dtype/width for _load_from_lds"


# ===----------------------------------------------------------------------=== #
# load_lds_fragment: MMA LDS→register load
# ===----------------------------------------------------------------------=== #


@always_inline
def load_lds_fragment[
    smem_layout: TensorLayout,
    reg_layout: TensorLayout,
    //,
    MMA_K: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    smem_tile: SMemTile[mut=False, _, smem_layout, _],
    reg_tile: RegTile[mut=True, smem_tile.dtype, reg_layout, _],
):
    """Load MMA fragments from SMEM to registers using hardware access pattern.

    Dimensions are derived from the tile layouts:
        - num_mmas = reg rows, MMA_M = smem rows / num_mmas
        - lds_frag_width = MMA_M * MMA_K / WARP_SIZE
        - lds_row_stride: MMA_K (BF16 dense), smem stride (FP8 or strided)
        - num_iterations = reg flat elements / lds_frag_width

    Parameters:
        smem_layout: Inferred layout of the SMEM source tile.
        reg_layout: Inferred layout of the register destination tile.
        MMA_K: MMA K dimension (hardware instruction width).
        swizzle: Optional element-space swizzle.

    Args:
        smem_tile: Source [num_mmas * MMA_M, K] in SHARED.
        reg_tile: Destination [num_mmas, K_frags * frag_width] in LOCAL.
    """
    comptime dtype = smem_tile.dtype
    comptime smem_rows = smem_layout.static_shape[0]
    comptime smem_cols = smem_layout.static_shape[1]
    comptime smem_stride = smem_layout.static_stride[0]
    comptime num_mmas = reg_layout.static_shape[0]
    comptime reg_cols = reg_layout.static_shape[1]
    comptime reg_stride = reg_layout.static_stride[0]
    comptime MMA_M = smem_rows // num_mmas
    comptime mma_frag_width = MMA_M * MMA_K // WARP_SIZE
    comptime use_fp8_split = (
        dtype.is_float8() and MMA_M == 16 and MMA_K == 128
    )
    comptime lds_frag_width = 16 if use_fp8_split else mma_frag_width
    comptime num_iterations = (num_mmas * reg_cols) // lds_frag_width

    # SMEM row stride: when the smem tile is a narrow sub-tile of a wider
    # allocation (stride > cols), use the physical stride. Otherwise use the
    # MMA-native stride: smem_cols for FP8 (contiguous BK), MMA_K for BF16
    # (mma_access_layout packs MMA_K elements per logical row).
    comptime smem_is_subtile = smem_stride > smem_cols
    comptime lds_row_stride = (
        smem_stride if smem_is_subtile else (
            smem_cols if dtype.is_float8() else MMA_K
        )
    )

    # Register stride: strided sub-tile (stride > cols) spaces fragments at
    # row stride; dense tile packs fragments contiguously.
    comptime _reg_stride = (
        reg_stride if reg_stride > reg_cols else lds_frag_width
    )

    var smem_ptr = smem_tile.ptr
    var reg_ptr = reg_tile.ptr
    comptime FragElement = SIMD[dtype, lds_frag_width]

    comptime col_groups = WARP_SIZE // MMA_M
    var lane = lane_id()
    var lane_offset = (
        Int(lane % MMA_M) * lds_row_stride + Int(lane // MMA_M) * lds_frag_width
    )

    comptime elements_per_iter = col_groups * lds_frag_width
    comptime use_split_k = lds_row_stride > elements_per_iter

    comptime if use_split_k:
        comptime k_splits = lds_row_stride // elements_per_iter
        comptime m_positions = num_iterations // k_splits
        comptime k_stride = elements_per_iter
        comptime m_stride = lds_row_stride * MMA_M

        comptime for m_idx in range(m_positions):
            comptime for k_idx in range(k_splits):
                var iter_base = m_idx * m_stride + k_idx * k_stride
                var full_offset = iter_base + lane_offset

                comptime if swizzle:
                    full_offset = swizzle.value()(full_offset)

                comptime frag_idx = m_idx * k_splits + k_idx
                reg_ptr.store[width=lds_frag_width](
                    frag_idx * _reg_stride,
                    rebind[FragElement](
                        _load_from_lds[width=lds_frag_width](
                            smem_ptr + full_offset
                        )
                    ),
                )
    else:
        comptime for i in range(num_iterations):
            var iter_base = i * WARP_SIZE * lds_frag_width
            var full_offset = iter_base + lane_offset

            comptime if swizzle:
                full_offset = swizzle.value()(full_offset)

            reg_ptr.store[width=lds_frag_width](
                i * _reg_stride,
                rebind[FragElement](
                    _load_from_lds[width=lds_frag_width](smem_ptr + full_offset)
                ),
            )


# ===----------------------------------------------------------------------=== #
# QuadrantMmaOp
# ===----------------------------------------------------------------------=== #


struct QuadrantMmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    k_group_size: Int,
    num_k_groups: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    swizzle: Optional[Swizzle] = None,
]:
    """MMA operator for AMD matmul ping-pong schedule.

    Owns A/B/C register tiles in LOCAL address space. Provides quadrant
    load/compute methods for the ping-pong double-buffering schedule:
    load_a_quadrant/load_b_quadrant fill half the register tile from
    SMEM via load_lds_fragment, then mma_quadrant computes on it.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        k_group_size: Number of MMA K-groups per fragment load.
        num_k_groups: Number of k-groups across the full BK dimension.
        num_m_mmas: MMA tiles along M within the warp tile.
        num_n_mmas: MMA tiles along N within the warp tile.
        swizzle: Optional SMEM swizzle for load helpers.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime c_frag_size = num_matrix_reg[Self.MMA_M, Self.MMA_N]()

    # Fragment width for load_lds_fragment.
    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE

    # Derived SMEM warp tile shapes.
    comptime num_k_mmas = Self.num_k_groups * Self.k_group_size
    comptime WM = Self.num_m_mmas * Self.MMA_M
    comptime WN = Self.num_n_mmas * Self.MMA_N
    comptime BK = Self.num_k_mmas * Self.MMA_K

    # Quadrant shapes (ping-pong schedule).
    comptime quad_m = Self.num_m_mmas // 2
    comptime quad_n = Self.num_n_mmas // 2
    comptime quad_WM = Self.quad_m * Self.MMA_M
    comptime quad_WN = Self.quad_n * Self.MMA_N

    # Register layouts: [num_m_mmas, num_k_mmas * mma_frag_width]
    # so quadrant tiling works.
    comptime _a_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_k_mmas * Self._mma_frag_width,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas,
        Self.num_k_mmas * Self._mma_frag_width,
    ]()
    # C layout: [num_m_mmas, num_n_mmas * c_frag_size]
    # so quadrant tiling [quad_m, quad_n * c_frag](which_a, which_b) works.
    comptime accum_width = Self.c_frag_size
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas, Self.num_n_mmas * Self.accum_width
    ]()

    var _a_reg: TileTensor[
        Self.in_type,
        type_of(Self._a_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_reg: TileTensor[
        Self.in_type,
        type_of(Self._b_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _c_reg: TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        self._a_reg = stack_allocation[Self.in_type, AddressSpace.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[Self.in_type, AddressSpace.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[Self.out_type, AddressSpace.LOCAL](
            Self._c_reg_layout
        )
        comptime num_c_elems = (
            Self.num_m_mmas * Self.num_n_mmas * Self.c_frag_size
        )
        comptime for i in range(num_c_elems):
            self._c_reg.ptr[i] = Scalar[Self.out_type](0)

    # === Quadrant methods for ping-pong schedule ===
    # The schedule calls load_a_quadrant/load_b_quadrant to fill one half
    # of the register tile, then mma_quadrant to compute on it.

    @always_inline
    def load_a_quadrant[
        which: Int
    ](self, smem_tile: SMemTile[Self.in_type, _, _],):
        """Load A quadrant `which` from SMEM sub-tile to registers.

        Tiles a_reg as [quad_m, reg_cols](which, 0) to get the register
        sub-tile for this quadrant, then loads via load_lds_fragment.
        """
        comptime assert type_of(smem_tile).static_shape[0] == Self.quad_WM
        comptime assert type_of(smem_tile).static_shape[1] == Self.BK

        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        var reg_quad = self._a_reg.tile[Self.quad_m, reg_cols](which, 0)
        load_lds_fragment[Self.MMA_K, Self.swizzle](smem_tile, reg_quad)

    @always_inline
    def load_b_quadrant[
        which: Int
    ](self, smem_tile: SMemTile[Self.in_type, _, _],):
        """Load B quadrant `which` from SMEM sub-tile to registers."""
        comptime assert type_of(smem_tile).static_shape[0] == Self.quad_WN
        comptime assert type_of(smem_tile).static_shape[1] == Self.BK

        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        var reg_quad = self._b_reg.tile[Self.quad_n, reg_cols](which, 0)
        load_lds_fragment[Self.MMA_K, Self.swizzle](smem_tile, reg_quad)

    @always_inline
    def mma_quadrant[which_a: Int, which_b: Int](self):
        """Execute MMA for quadrant (which_a, which_b) via TiledMma.

        Slices A/B/C register tiles to the quadrant and delegates to
        TiledMma for stateless computation.
        """
        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        comptime c_quad_cols = Self.quad_n * Self.c_frag_size

        var a_quad = self._a_reg.tile[Self.quad_m, reg_cols](which_a, 0)
        var b_quad = self._b_reg.tile[Self.quad_n, reg_cols](which_b, 0)
        var c_quad = self._c_reg.tile[Self.quad_m, c_quad_cols](
            which_a, which_b
        )
        TiledMma[Self.out_type, Self.in_type, Self.shape, Self.num_k_mmas].mma(
            a_quad, b_quad, c_quad
        )

    @always_inline
    def accum_tile(
        self,
    ) -> TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Return the accumulator register tile."""
        return self._c_reg


# ===----------------------------------------------------------------------=== #
# TiledMma — stateless MMA computation on TileTensors
# ===----------------------------------------------------------------------=== #


struct TiledMma[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    group_size: Int,
]:
    """Stateless MMA computation on TileTensors.

    Direct TileTensor port of TiledTensorCore.mma. Iterates group_size
    k-steps, indexes A/B register tiles per step, and calls gpu_mma.
    No register ownership, no SMEM loading — pure computation.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        group_size: Number of k-steps per mma() call.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime a_frag_size = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    @staticmethod
    @always_inline
    def mma[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
    ](
        a_reg: TileTensor[
            Self.in_type, a_layout, _, address_space=AddressSpace.LOCAL
        ],
        b_reg: TileTensor[
            Self.in_type, b_layout, _, address_space=AddressSpace.LOCAL
        ],
        c_reg: TileTensor[
            Self.out_type,
            c_layout,
            MutExternalOrigin,
            address_space=AddressSpace.LOCAL,
        ],
    ):
        """Execute group_size MMA operations across the K dimension.

        Mirrors TiledTensorCore.mma: iterates group_size k-steps, tiles
        A/B registers per step via vectorize, and accumulates into C.

        Parameters:
            a_layout: Inferred layout of A register tile.
            b_layout: Inferred layout of B register tile.
            c_layout: Inferred layout of C register tile.

        Args:
            a_reg: A fragments [num_m_mmas, group_size * a_frag_size].
            b_reg: B fragments [num_n_mmas, group_size * a_frag_size].
            c_reg: Accumulator [num_m_mmas, num_n_mmas * c_frag_size].
        """
        comptime a_frag_size = Self.a_frag_size
        comptime c_frag_size = Self.c_frag_size
        comptime num_m_mmas = a_layout.static_shape[0]
        comptime num_n_mmas = b_layout.static_shape[0]

        var a_vec = a_reg.vectorize[1, a_frag_size]()
        var b_vec = b_reg.vectorize[1, a_frag_size]()
        var c_vec = c_reg.vectorize[1, c_frag_size]()

        # Provide evidence for TileTensor.__getitem__'s `where flat_rank`
        # constraint. With inferred layouts the compiler can't prove rank
        # from the generic type alone — these asserts supply the proof.
        comptime assert type_of(a_vec).flat_rank == 2
        comptime assert type_of(b_vec).flat_rank == 2
        comptime assert type_of(c_vec).flat_rank == 2

        comptime for k in range(Self.group_size):
            comptime for m in range(num_m_mmas):
                comptime for n in range(num_n_mmas):
                    gpu_mma(
                        c_vec[m, n],
                        b_vec[n, k],
                        a_vec[m, k],
                        c_vec[m, n],
                    )


# ===----------------------------------------------------------------------=== #
# MmaOp — register ownership + SMEM loading + schedule API
# ===----------------------------------------------------------------------=== #


struct MmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    swizzle: Optional[Swizzle] = None,
]:
    """Register ownership + SMEM loading + schedule API for AMD matmul.

    Owns A/B/C register tiles in LOCAL address space. Provides the
    schedule-facing API: load_frag[k] loads from SMEM to registers,
    mma[k] delegates to TiledMma for computation.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        k_group_size: Number of MMA k-steps per fragment load.
        num_k_tiles: Number of k-tiles across the warp K dimension.
        num_m_mmas: MMA tiles along M within the warp tile.
        num_n_mmas: MMA tiles along N within the warp tile.
        swizzle: Optional SMEM swizzle for fragment loading.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    # simd_width = k_group_size * _mma_frag_width: elements per k-tile load.
    comptime simd_width = Self.k_group_size * Self._mma_frag_width

    # Derived SMEM warp tile shapes.
    comptime WM = Self.num_m_mmas * Self.MMA_M
    comptime WN = Self.num_n_mmas * Self.MMA_N
    comptime k_tile_size = Self.MMA_K * Self.k_group_size

    # Register layouts: height = num_mmas * num_k_tiles,
    # width = simd_width (one k-tile's worth of elements per row).
    comptime _a_reg_layout = row_major[
        Self.num_m_mmas * Self.num_k_tiles,
        Self.simd_width,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas * Self.num_k_tiles,
        Self.simd_width,
    ]()
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
    ]()

    var _a_reg: TileTensor[
        Self.in_type,
        type_of(Self._a_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_reg: TileTensor[
        Self.in_type,
        type_of(Self._b_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _c_reg: TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        self._a_reg = stack_allocation[Self.in_type, AddressSpace.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[Self.in_type, AddressSpace.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[Self.out_type, AddressSpace.LOCAL](
            Self._c_reg_layout
        )
        comptime num_c_elems = (
            Self.num_m_mmas * Self.num_n_mmas * Self.c_frag_size
        )
        comptime for i in range(num_c_elems):
            self._c_reg.ptr[i] = Scalar[Self.out_type](0)

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_frag[
        k_tile_idx: Int
    ](
        self,
        a_smem_warp: SMemTile[Self.in_type, _, _],
        b_smem_warp: SMemTile[Self.in_type, _, _],
    ):
        """Load A and B MMA fragments for k-tile k_tile_idx from SMEM.

        Expects block-local warp tiles of shape WM x k_tile_size (or
        WN x k_tile_size), where each k-tile block is contiguous in
        SMEM (blocked_product layout). Uses direct distribute with
        swizzle — correct because each block starts at a
        swizzle-aligned offset.
        """
        comptime assert type_of(a_smem_warp).static_shape[0] == Self.WM
        comptime assert type_of(b_smem_warp).static_shape[0] == Self.WN
        comptime assert type_of(a_smem_warp).static_shape[1] == Self.k_tile_size
        comptime assert type_of(b_smem_warp).static_shape[1] == Self.k_tile_size

        # B fragments first for ds_read scheduling.
        comptime for i in range(Self.num_n_mmas):
            var b_row = k_tile_idx * Self.num_n_mmas + i
            var dist = (
                b_smem_warp.tile[Self.MMA_N, Self.k_tile_size](i, 0)
                .vectorize[1, Self.simd_width]()
                .distribute[
                    col_major[Self.MMA_N, WARP_SIZE // Self.MMA_N](),
                    swizzle=Self.swizzle,
                ](lane_id())
            )
            self._b_reg.vectorize[1, Self.simd_width]()[b_row, 0] = dist[0, 0]

        # A fragments.
        comptime for i in range(Self.num_m_mmas):
            var a_row = k_tile_idx * Self.num_m_mmas + i
            var dist = (
                a_smem_warp.tile[Self.MMA_M, Self.k_tile_size](i, 0)
                .vectorize[1, Self.simd_width]()
                .distribute[
                    col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M](),
                    swizzle=Self.swizzle,
                ](lane_id())
            )
            self._a_reg.vectorize[1, Self.simd_width]()[a_row, 0] = dist[0, 0]

    @always_inline
    def mma[k_tile_idx: Int](self):
        """Execute MMA for k-tile k_tile_idx via TiledMma.

        Slices A/B registers for this k-tile and delegates to
        TiledMma.mma for stateless computation.
        """
        var a_slice = self._a_reg.tile[Self.num_m_mmas, Self.simd_width](
            k_tile_idx, 0
        )
        var b_slice = self._b_reg.tile[Self.num_n_mmas, Self.simd_width](
            k_tile_idx, 0
        )
        TiledMma[
            Self.out_type, Self.in_type, Self.shape, Self.k_group_size
        ].mma(a_slice, b_slice, self._c_reg)
