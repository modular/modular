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

from std.gpu.compute.mma import mma as gpu_mma
from std.gpu import lane_id, WARP_SIZE
from std.gpu.intrinsics import ds_read_tr16_b64
from std.math.uutils import ufloordiv, umod
from std.memory import AddressSpace
from std.sys import simd_width_of
from std.utils import IndexList
from std.utils.numerics import get_accum_type
from layout import TileTensor
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation
from layout import Coord, ComptimeInt, MixedLayout
from structured_kernels.amd_tile_io import (
    TiledMmaLoader,
    smem_mma_subtile_offset,
)


struct TiledMmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
]:
    """TileTensor-native MMA operation for AMD attention kernels.

    Wraps the raw GPU MMA intrinsic and operates directly on TileTensor
    register tiles.

    Parameters:
        out_type: Accumulator data type.
        in_type: Input matrix element data type.
        shape: MMA instruction shape [M, N, K].
        transpose_b: Whether to transpose the B matrix.
    """

    comptime a_frag_size = num_matrix_reg[Self.shape[0], Self.shape[2]]()
    comptime b_frag_size = num_matrix_reg[Self.shape[2], Self.shape[1]]()
    comptime c_frag_size = num_matrix_reg[Self.shape[0], Self.shape[1]]()

    @staticmethod
    @always_inline
    def mma[
        swap_a_b: Bool = False
    ](
        a: TileTensor[_, _, address_space=AddressSpace.LOCAL, ...],
        b: TileTensor[_, _, address_space=AddressSpace.LOCAL, ...],
        c: TileTensor[mut=True, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Perform MMA on TileTensor operands.

        Tiles down to individual MMA fragments so the compiler can
        prove static shapes, then calls the raw gpu_mma intrinsic
        directly.

        Parameters:
            swap_a_b: Whether to swap A and B operands. Only controls the
                argument order of `gpu_mma`; accumulator indexing is always
                col-major over (M, N).

        Args:
            a: A operand tile [num_m_mmas, a_frag_size].
            b: B operand tile [num_n_mmas, b_frag_size].
            c: Accumulator tile [num_m_mmas * num_n_mmas, c_frag_size],
                modified in-place.
        """
        comptime num_m_mmas = type_of(a).static_shape[0]
        comptime num_n_mmas = type_of(b).static_shape[0]

        comptime a_frag = Self.a_frag_size
        comptime b_frag = Self.b_frag_size
        comptime c_frag = Self.c_frag_size

        var a_k = a.tile[num_m_mmas, a_frag](0, 0).vectorize[1, a_frag]()
        var b_k = b.tile[num_n_mmas, b_frag](0, 0).vectorize[1, b_frag]()

        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                # col_major(M, N): c_idx = m + n*M.
                comptime c_idx = m_mma + n_mma * num_m_mmas
                # Tile to a single [1, c_frag] fragment, then
                # vectorize to [1, 1] — provably rank-2.
                var c_frag_vec = c.tile[1, c_frag](c_idx, 0).vectorize[
                    1, c_frag
                ]()
                comptime if swap_a_b:
                    gpu_mma(
                        c_frag_vec[0, 0],
                        b_k[n_mma, 0],
                        a_k[m_mma, 0],
                        c_frag_vec[0, 0],
                    )
                else:
                    gpu_mma(
                        c_frag_vec[0, 0],
                        a_k[m_mma, 0],
                        b_k[n_mma, 0],
                        c_frag_vec[0, 0],
                    )

    @staticmethod
    @always_inline
    def load_b[
        swizzle: Optional[Swizzle] = None,
    ](
        warp_tile: TileTensor[
            Self.in_type, _, _, address_space=AddressSpace.SHARED, ...
        ],
        reg_tile: TileTensor[
            mut=True, Self.in_type, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        k_group_idx: Int = 0,
    ):
        """Load B-matrix fragments from SMEM to registers.

        Distributes the warp tile across threads with optional swizzle,
        loading one MMA fragment per iteration. Handles both transposed
        and non-transposed B layouts via comptime dispatch.

        Parameters:
            swizzle: Optional swizzle for SMEM bank-conflict avoidance.

        Args:
            warp_tile: Source warp tile in shared memory.
            reg_tile: Destination register tile for MMA fragments.
            k_group_idx: K-dimension group index within the warp tile.
        """
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        comptime simd_width = num_matrix_reg[mma_k, mma_n]()

        comptime num_frags = type_of(reg_tile).static_shape[0]
        var reg_vec = reg_tile.vectorize[1, simd_width]()
        comptime assert type_of(reg_vec).flat_rank == 2

        comptime if Self.transpose_b:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[mma_n, mma_k](i, k_group_idx)
                var dist = mma_tile.vectorize[1, simd_width]().distribute[
                    col_major[mma_n, WARP_SIZE // mma_n](),
                    swizzle=swizzle,
                ](lane_id())
                comptime assert type_of(dist).flat_rank == 2
                reg_vec[i, 0] = dist[0, 0]
        else:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[mma_k, mma_n](k_group_idx, i)
                var dist = mma_tile.vectorize[simd_width, 1]().distribute[
                    row_major[WARP_SIZE // mma_n, mma_n](),
                    swizzle=swizzle,
                ](lane_id())
                comptime assert type_of(dist).flat_rank == 2
                reg_vec[i, 0] = dist[0, 0]

    @staticmethod
    @always_inline
    def load_a[
        swizzle: Optional[Swizzle] = None,
    ](
        warp_tile: TileTensor[
            Self.in_type, _, _, address_space=AddressSpace.SHARED, ...
        ],
        reg_tile: TileTensor[
            mut=True, Self.in_type, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        k_group_idx: Int = 0,
    ):
        """Load A-matrix fragments from SMEM to registers.

        Distributes the warp tile across threads with optional swizzle,
        loading one MMA fragment per iteration. Always uses col_major
        thread distribution.

        Parameters:
            swizzle: Optional swizzle for SMEM access.

        Args:
            warp_tile: Source warp tile in shared memory.
            reg_tile: Destination register tile for MMA fragments.
            k_group_idx: K-dimension group index within the warp tile.
        """
        comptime mma_m = Self.shape[0]
        comptime mma_k = Self.shape[2]
        comptime simd_width = num_matrix_reg[mma_m, mma_k]()
        comptime num_frags = type_of(reg_tile).static_shape[0]
        var reg_vec = reg_tile.vectorize[1, simd_width]()
        comptime assert type_of(reg_vec).flat_rank == 2

        comptime for i in range(num_frags):
            var mma_tile = warp_tile.tile[mma_m, mma_k](i, k_group_idx)
            var dist = mma_tile.vectorize[1, simd_width]().distribute[
                col_major[mma_m, WARP_SIZE // mma_m](),
                swizzle=swizzle,
            ](lane_id())
            comptime assert type_of(dist).flat_rank == 2
            reg_vec[i, 0] = dist[0, 0]


struct KVMmaOp[
    in_type: DType,
    mma_shape: IndexList[3],
    num_mmas: Int,
    num_k_mmas: Int,
    num_k_tiles: Int,
    BN: Int,
    BK: Int,
    transpose_b: Bool = True,
    swizzle: Optional[Swizzle] = None,
    out_type: DType = get_accum_type[in_type](),
]:
    """Owns the K or V operand register tile and its SMEM→reg load logic.

    Attention has two sequential GEMMs (P = Q @ K^T, O += P @ V). Instantiate
    one `KVMmaOp` per operand role. This keeps KVBuffer focused on SMEM
    storage + DMA and moves MMA-side concerns (reg layout, frag size,
    fragment loads) here.

    The register layout is organized as
    `[num_k_tiles][num_k_mmas][num_mmas] x input_frag_size`: each BK strip
    holds `num_k_mmas * num_mmas` fragments back-to-back.

    Parameters:
        in_type: Operand element type (bfloat16 or float8_e4m3fn).
        mma_shape: MMA instruction shape [M, N, K].
        num_mmas: MMA tiles along the warp's M or N axis (WN/MMA_M for K).
        num_k_mmas: MMA tiles along K within a single BK strip.
        num_k_tiles: Number of BK strips across the full depth.
        BN: KV block height (needed by V load methods for SMEM offset math).
        BK: KV block width (needed by V load methods for SMEM offset math).
        transpose_b: True for K (transposed load), False for V.
        swizzle: Optional SMEM swizzle — vector-space for prefill,
            element-space for decode.
        out_type: Accumulator data type (defaults to accum(in_type)).
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime input_frag_size = (Self.MMA_K * Self.MMA_N) // WARP_SIZE

    comptime _rows_per_k_tile = Self.num_mmas * Self.num_k_mmas
    comptime _total_rows = Self.num_k_tiles * Self._rows_per_k_tile
    comptime _reg_layout = row_major[Self._total_rows, Self.input_frag_size]()

    var reg_tile: TileTensor[
        Self.in_type,
        type_of(Self._reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        self.reg_tile = stack_allocation[Self.in_type, AddressSpace.LOCAL](
            Self._reg_layout
        )

    @always_inline
    def load_prefill[
        bk_tile: Int
    ](
        self,
        warp_smem: TileTensor[
            Self.in_type, _, _, address_space=AddressSpace.SHARED, ...
        ],
    ):
        """Load the `bk_tile`-th BK strip of K fragments from SMEM.

        Delegates to `TiledMmaLoader.load_b` (M-outer iteration,
        vector-space swizzle). Handles both BF16 (single load per MMA
        tile) and FP8 (two-half-K load + join) via the `num_packs`
        branch inside `load_b`.
        """
        comptime assert Self.transpose_b, "load_prefill is K-operand only"
        comptime total_frags = Self.num_mmas * Self.num_k_mmas
        var frags = TiledMmaLoader[
            Self.in_type, Self.mma_shape, Self.swizzle
        ].load_b[total_frags, Self.input_frag_size](warp_smem)

        var dst = self.reg_tile.tile[
            Self._rows_per_k_tile, Self.input_frag_size
        ](bk_tile, 0).vectorize[1, Self.input_frag_size]()

        comptime for i in range(total_frags):
            dst[Int(i), 0] = rebind[type_of(dst[Int(i), 0])](frags[Int(i)])

    @always_inline
    def mma_tile_at[
        bk_tile: Int, kg: Int
    ](self) -> TileTensor[
        Self.in_type,
        type_of(row_major[Self.num_mmas, Self.input_frag_size]()),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]:
        """Sub-view of the reg tile for a given (bk_tile, k-group) pair."""
        return self.reg_tile.tile[Self._rows_per_k_tile, Self.input_frag_size](
            bk_tile, 0
        ).tile[Self.num_mmas, Self.input_frag_size](kg, 0)

    @always_inline
    def load_v_bf16[
        bk_tile: Int
    ](
        self,
        smem_base: UnsafePointer[
            Scalar[Self.in_type],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
    ):
        """Load the `bk_tile`-th BK strip of BF16 V fragments from SMEM.

        V SMEM is blocked (num_repeats × BN × BK, row-major within each
        block). For each (k, i) ∈ [num_k_mmas] × [num_mmas], build an
        MMA sub-tile view with the correct `(BK, 1)` row stride (not
        `(MMA_M, 1)` — see `smem_mma_subtile` header), call
        `TiledMmaLoader.load_b_tr`, and write the fragment into the
        reg slot `mma_tile_at[bk_tile, k][i]`.

        Only valid when `transpose_b == False` and `in_type ==
        bfloat16`.
        """
        comptime assert not Self.transpose_b, "load_v_bf16 is V-operand only"
        comptime assert (
            Self.in_type == DType.bfloat16
        ), "load_v_bf16 is bf16 only"

        comptime _MmaTileLayout = MixedLayout[
            Coord[
                ComptimeInt[Self.MMA_K], ComptimeInt[Self.MMA_M]
            ].element_types,
            Coord[ComptimeInt[Self.BK], ComptimeInt[1]].element_types,
        ]

        # mla_kv_alias mode: V reads from K's swizzled SMEM, so V's
        # `ds_read_tr16_b64` per-lane addresses must apply the same
        # swizzle K's writer applies.  `load_b_tr` /
        # `ds_read_tr16_b64_warp` / `ds_read_tr16_b64_row` only know
        # about `tile.ptr` and have no `block_base` to swizzle relative
        # to, so we inline the addressing here and call the bare
        # `ds_read_tr16_b64` intrinsic with a swizzled per-lane address.
        # Mirrors `load_v_fp8_strip`'s swizzle path.  No-swizzle case
        # keeps the original helper-based load.
        comptime if Self.swizzle:
            # Per-lane element offset within a (4, 16) shared_b_tile
            # (inside a (16, 16) half of the (MMA_K, MMA_N) sub-tile).
            # Stride along the K-dim of the tile is `BK` elements.
            # `ds_read_tr16_b64_row` distributes 16 lanes over a
            # `row_major[4, 4]` grid of (1, 4)-vectorized cells.
            comptime simd_w_elem = simd_width_of[Self.in_type]()
            var lid = lane_id()
            var row_idx = ufloordiv(lid, 16)  # 0..3 (within warp's 4-row split)
            var lane_in_row = umod(lid, 16)
            var lir_row = ufloordiv(lane_in_row, 4)  # 0..3
            var lir_col = umod(lane_in_row, 4)  # 0..3
            var lane_off_in_half = (
                Int(row_idx) * 4 * Self.BK
                + Int(lir_row) * Self.BK
                + Int(lir_col) * 4
            )

            comptime for k in range(Self.num_k_mmas):
                comptime for i in range(Self.num_mmas):
                    var mma_tile_off = smem_mma_subtile_offset[
                        Self.MMA_K, Self.MMA_M, Self.BN, Self.BK
                    ](bk_tile, Int(k), Int(i))

                    @parameter
                    def _read_half[half_idx: Int]() -> SIMD[Self.in_type, 4]:
                        comptime half_off = (
                            half_idx * (Self.MMA_K // 2) * Self.BK
                        )
                        var total_elem = (
                            mma_tile_off + half_off + lane_off_in_half
                        )
                        var vec_idx = total_elem // simd_w_elem
                        var sub_vec = total_elem & (simd_w_elem - 1)
                        var swizzled_elem = (
                            Self.swizzle.value()(vec_idx) * simd_w_elem
                            + sub_vec
                        )
                        return ds_read_tr16_b64(smem_base + swizzled_elem)

                    var part_1 = _read_half[0]()
                    var part_2 = _read_half[1]()
                    var frag = part_1.join(part_2)

                    var dst = self.mma_tile_at[bk_tile, Int(k)]().vectorize[
                        1, Self.input_frag_size
                    ]()
                    dst[Int(i), 0] = rebind[type_of(dst[Int(i), 0])](frag)
        else:
            comptime for k in range(Self.num_k_mmas):
                comptime for i in range(Self.num_mmas):
                    var offset = smem_mma_subtile_offset[
                        Self.MMA_K, Self.MMA_M, Self.BN, Self.BK
                    ](bk_tile, Int(k), Int(i))
                    var mma_smem = TileTensor[
                        Self.in_type,
                        _MmaTileLayout,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                    ](smem_base + offset, _MmaTileLayout())
                    var frag = TiledMmaLoader[
                        Self.in_type, Self.mma_shape
                    ].load_b_tr(mma_smem)

                    var dst = self.mma_tile_at[bk_tile, Int(k)]().vectorize[
                        1, Self.input_frag_size
                    ]()
                    dst[Int(i), 0] = rebind[type_of(dst[Int(i), 0])](frag)

    @always_inline
    def load_v_fp8_strip[
        bk_tile: Int
    ](
        self,
        smem_base: UnsafePointer[
            Scalar[Self.in_type],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        rel_key: Int,
        hw_key_shift: Int,
        depth_base: Int,
    ):
        """Load the `bk_tile`-th BK strip of FP8 V fragments from SMEM.

        Iterates `dt` over the depth direction and calls
        `TiledMmaLoader.load_v_fp8_strip` per (bk_tile, dt) pair,
        writing the joined 32-element SIMD into
        `mma_tile_at[bk_tile, 0][dt]`.

        Only valid when `transpose_b == False` and `in_type` is FP8.
        Caller precomputes the lane-only coords (`rel_key`,
        `hw_key_shift`, `depth_base`) once before a multi-bk loop —
        they don't depend on bk_tile or dt.
        """
        comptime assert (
            not Self.transpose_b
        ), "load_v_fp8_strip is V-operand only"
        comptime assert Self.in_type.is_float8(), "load_v_fp8_strip is FP8 only"
        comptime assert (
            Self.num_k_mmas == 1
        ), "FP8 V expects num_k_mmas == 1 (MMA_K >= BK)"

        var dst = self.mma_tile_at[bk_tile, 0]().vectorize[
            1, Self.input_frag_size
        ]()
        comptime for dt in range(Self.num_mmas):
            var joined = TiledMmaLoader[
                Self.in_type, Self.mma_shape, swizzle=Self.swizzle
            ].load_v_fp8_strip[Self.BN, Self.BK, bk_tile, Int(dt)](
                smem_base, rel_key, hw_key_shift, depth_base
            )
            dst[Int(dt), 0] = rebind[type_of(dst[Int(dt), 0])](joined)

    @always_inline
    def mma[
        swap_a_b: Bool = False,
    ](
        self,
        a: TileTensor[_, _, address_space=AddressSpace.LOCAL, ...],
        c: TileTensor[mut=True, _, _, address_space=AddressSpace.LOCAL, ...],
        bk_tile: Int,
        kg: Int,
    ):
        """Compute C += A * B using this op's reg tile as B operand."""
        TiledMmaOp[
            Self.out_type,
            Self.in_type,
            Self.mma_shape,
            transpose_b=Self.transpose_b,
        ].mma[swap_a_b=swap_a_b](
            a,
            self.reg_tile.tile[Self._rows_per_k_tile, Self.input_frag_size](
                bk_tile, 0
            ).tile[Self.num_mmas, Self.input_frag_size](kg, 0),
            c,
        )
