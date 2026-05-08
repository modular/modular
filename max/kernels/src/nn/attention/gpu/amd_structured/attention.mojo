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
"""Attention struct for gfx950 MHA/MLA kernels (prefill + decode).

TileTensor-only API: no LayoutTensor in struct fields, constructor
parameters, method signatures, or internal bridges. Mask application is
delegated to `MaskTileOp` (see `mask_op.mojo`), which uses
TileTensor.vectorize with local SIMD read/modify/write for per-element edits.

Inlines `AMDStructuredConfig` directly (no `AttentionConfig` trait indirection).
Prefill kernels handle MMA inline — there is no `_dma_loop`/`mma_qk`/`mma_pv`
helper on the struct.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.math.constants import log2e
from std.memory import bitcast, stack_allocation
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from std.sys import align_of, simd_width_of, size_of
from std.sys.intrinsics import _type_is_eq, readfirstlane
from std.gpu import barrier, block_idx, lane_id, thread_idx
from layout import TileTensor
from layout import row_major
from layout.swizzle import Swizzle
from layout.tile_layout import (
    ComptimeInt,
    Idx,
    RuntimeInt,
    Layout as TileLayout,
)
from layout.coord import Coord
from structured_kernels.amd_tile_io import RegTileWriter
from layout.tensor_core import num_matrix_reg
from nn.attention.mha_mask import CausalMask, MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig
from std.utils import Index, IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf

from .softmax import Softmax
from .buffers import (
    OutputRegisterBuffer,
    PRegisterBuffer,
    QRegisterBuffer,
)
from .config import AMDStructuredConfig
from .mask_op import MaskTileOp
from .mma import TiledMmaOp
from .utils import get_warp_coords


# ===----------------------------------------------------------------------=== #
# Attention struct — gfx950, no traits
# ===----------------------------------------------------------------------=== #


struct Attention[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MHAConfig,
    group: Int,
    sink: Bool,
    token_gen: Bool = False,
    q_depth: Int = config.depth,
    cache_depth: Int = config.depth,
    output_depth: Int = config.depth,
    mla_mode: Bool = False,
    # MLA decode: alias V onto K's SMEM (skip V DMA); K loaded unswizzled.
    # Requires `shared_kv=True` in `AMDStructuredConfig` — that's where the
    # `v_smem_ptr` is bitcast'd from `k_smem_ptr` (no separate V allocation).
    # Enforced via `comptime assert` in `__init__`.
    mla_kv_alias: Bool = False,
]:
    # Block/warp dimensions from MHAConfig.
    comptime BM = Self.config.block_m()
    comptime BN = Self.config.block_n()
    comptime BK = Self.config.block_k()
    comptime WM = Self.config.warp_m()
    comptime WN = Self.config.warp_n()
    comptime num_threads = Self.config.num_threads()
    comptime num_heads = Self.config.num_heads
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps_m = Self.BM // Self.WM
    comptime depth = Self.config.depth

    comptime accum_type = get_accum_type[Self.q_type]()

    # gfx950 config — concrete struct, no trait.
    comptime amd_structured_config = AMDStructuredConfig[
        Self.config, Self.group, Self.token_gen, Self.mla_mode
    ]

    comptime mma_shape = Self.amd_structured_config.get_mma_shape()

    # Per-lane MMA fragment geometry. On gfx950, the fragment is always a
    # row-vector: 1 row of `output_frag_size` registers per lane.
    comptime frag_num_rows = 1
    comptime output_frag_size = num_matrix_reg[
        Self.mma_shape[0], Self.mma_shape[1]
    ]()

    comptime num_m_mmas = ceildiv(Self.WM, Self.mma_shape[0])
    comptime num_n_mmas = ceildiv(Self.WN, Self.mma_shape[1])

    comptime num_n_mmas_output = ceildiv(
        Self.output_depth // Self.num_warps_n, Self.mma_shape[1]
    )

    comptime swap_a_b = True
    comptime use_exp2 = True

    # k_group_size is an artifact of gfx942, where some MFMAs required
    # grouping two MMA tiles along K to fill a 16-byte per-lane fragment.
    # On gfx950, every `mma_shape` we dispatch already yields a 16-byte
    # fragment, so it is always 1. RDNA paths still thread
    # `MHAConfig.k_group_size`; gfx950 drops the plumbing entirely.

    comptime num_k_mmas2 = ceildiv(Self.BK, Self.mma_shape[2])

    # Per-warp lane grid: 32×32 MMAs use 32×2 col-major, 16×16 use 16×4.
    # Stored as (rows, cols) Int pair; RegTileWriter / QRegisterBuffer each
    # rebuild the col-major layout internally from these dims.
    comptime warp_rows = 32 if Self.mma_shape[0] == 32 else 16
    comptime warp_cols = 2 if Self.mma_shape[0] == 32 else 4

    # Prefill pipelines QK/PV across two stages (see mha_prefill's
    # mma_pv_incremental[1]), so it needs num_stages=2 on the P register
    # buffer. Decode only ever references stage 0, so a 1-stage buffer is
    # enough and saves `num_n_mmas * num_m_mmas * output_frag_size` VGPRs.
    comptime _p_buffer_stages = 1 if Self.token_gen else 2

    # KV DMA staging for the decode kernels in `mha_decode.mojo`: the
    # outer loop prefetches tile N+1 into stage (i+1) % N while the MMAs
    # consume stage i, so 2 stages.
    comptime num_stages = 2

    # --- Buffer type aliases ---

    comptime OutputRegisterBufferType = OutputRegisterBuffer[
        Self.accum_type,
        Self.num_m_mmas,
        Self.num_n_mmas_output,
        Self.output_frag_size,
    ]

    # P swizzle for shared_memory_backed path (BN != WN, 16x16 MMA).
    # Mirrors amd/attention.mojo: Swizzle(3,0,3) XORs bits [5:3] into [2:0]
    # of the 8-element group index, spreading rows across LDS bank groups.
    comptime _p_shared_memory_backed = Self.BN != Self.WN
    comptime _p_swizzle = (
        Swizzle(3, 0, 3) if (
            Self._p_shared_memory_backed
            and Self.mma_shape[0] == 16
            and Self.mma_shape[2] <= 2 * Self.mma_shape[1]
        ) else Optional[Swizzle](None)
    )

    # Raw FP8 cast (no clamp / NaN scrub) is safe whenever the values fed
    # to the cast are provably bounded and finite. That holds for softmax
    # output (in (0, 1]) across all attention paths we currently ship, so
    # enable it automatically for FP8. If a future caller feeds unbounded
    # values into the P→PV cast, set this False at the callsite.
    comptime _raw_fp8_cast = Self.q_type.is_float8()

    comptime PRegisterBufferType = PRegisterBuffer[
        Self.accum_type,
        Self.q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.WM,
        Self.WN,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.output_frag_size,
        Self._p_shared_memory_backed,
        Self.mma_shape,
        tr_load_enabled=True,
        num_stages=Self._p_buffer_stages,
        p_swizzle=Self._p_swizzle,
        raw_fp8_cast=Self._raw_fp8_cast,
    ]

    # --- TileTensor layouts for Q, output, and KV tiles ---
    # RuntimeInt row dim carries `valid_rows` for SRD OOB clamping.

    comptime kv_num_heads = Self.num_heads // Self.group

    comptime _q_stride0 = (
        Self.num_heads * Self.q_depth if not Self.token_gen else Self.q_depth
    )
    comptime QTileLayout = TileLayout[
        Coord[RuntimeInt[DType.int64], ComptimeInt[Self.q_depth]].element_types,
        Coord[ComptimeInt[Self._q_stride0], ComptimeInt[1]].element_types,
    ]

    comptime _output_stride0 = (
        Self.num_heads
        * Self.output_depth if not Self.token_gen else Self.output_depth
    )
    comptime OutputTileLayout = TileLayout[
        Coord[
            RuntimeInt[DType.int64], ComptimeInt[Self.output_depth]
        ].element_types,
        Coord[ComptimeInt[Self._output_stride0], ComptimeInt[1]].element_types,
    ]

    comptime _kv_stride0 = Self.kv_num_heads * Self.depth
    comptime KvTileLayout = TileLayout[
        Coord[RuntimeInt[DType.int64], ComptimeInt[Self.depth]].element_types,
        Coord[ComptimeInt[Self._kv_stride0], ComptimeInt[1]].element_types,
    ]

    # --- SMEM sizing (inlined from the former SharedMemoryManager struct) ---
    comptime _smem_alignment = align_of[
        SIMD[Self.q_type, simd_width_of[Self.q_type]()]
    ]()
    comptime _k_smem_size = Self.BN * (
        Self.depth if Self.amd_structured_config.full_kv else Self.BK
    ) * (
        2 if (
            Self.amd_structured_config.double_buffer
            or Self.amd_structured_config.double_buffer_k_only
        ) else 1
    )
    comptime _v_smem_size = (
        Self.BN if Self.amd_structured_config.full_kv else Self.BK
    ) * Self.depth * (2 if Self.amd_structured_config.double_buffer else 1)
    comptime _max_kv_smem_size = max(Self._k_smem_size, Self._v_smem_size)

    comptime QRegisterBufferType = QRegisterBuffer[
        dtype=Self.q_type,
        mma_shape=Self.mma_shape,
        WM=Self.WM,
        WN=Self.WN,
        BN=Self.BN,
        BK=Self.BK,
        depth=Self.q_depth,
        thread_rows=Self.warp_rows,
        thread_cols=Self.warp_cols,
    ]

    # --- MMA op alias ---
    comptime MmaOp = TiledMmaOp[
        get_accum_type[Self.q_type](),
        Self.q_type,
        Self.mma_shape,
    ]

    # --- Instance fields ---

    var out_reg_buffer: Self.OutputRegisterBufferType
    var p_reg_buffer: Self.PRegisterBufferType

    var k_smem_ptr: UnsafePointer[
        Scalar[Self.k_t.dtype],
        MutExternalOrigin,
        address_space=AddressSpace.SHARED,
    ]
    var v_smem_ptr: UnsafePointer[
        Scalar[Self.v_t.dtype],
        MutExternalOrigin,
        address_space=AddressSpace.SHARED,
    ]
    # Dedicated warp-reduction scratch SMEM. Decoupling from K SMEM
    # avoids races between softmax's scratch ds_writes and other warps'
    # in-flight K ds_reads (the prior overlay only happened to be safe
    # for non-kv-alias decode because V's later DMA wiped the corruption).
    var warp_scratch_ptr: UnsafePointer[
        Scalar[Self.accum_type],
        MutExternalOrigin,
        address_space=AddressSpace.SHARED,
    ]

    var q_buffer: Self.QRegisterBufferType
    var output_tile: TileTensor[
        Self.output_type,
        Self.OutputTileLayout,
        MutAnyOrigin,
    ]

    var batch_idx: Int

    var k: Self.k_t
    var v: Self.v_t
    var mask: Self.mask_t

    # Raw per-warp lane coords (invariant for the kernel lifetime).
    var warp_row: Int
    var warp_col: Int

    var mask_block_row: UInt32
    # Mask offsets: `mask_warp_row = warp_row * WM` is a cache (never mutated).
    # `mask_warp_col` starts at `warp_col * WN` but is advanced by `BN` each
    # K-tile (see `mask_advance`) so it tracks the current K-tile column.
    var mask_warp_row: UInt32
    var mask_warp_col: UInt32
    var kv_start_row: UInt32

    var scale: Scalar[Self.accum_type]

    var seq_len: Int
    var num_keys: Int
    var start_pos: Int
    var cache_start_pos: Int

    var softmax: Softmax[
        Self.accum_type,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_warps_m,
        Self.num_warps_n,
        Self.mma_shape[0],
        Self.use_exp2,
    ]

    comptime _warp_scratch_layout = row_major[
        2 * Int(Self.num_warps_n), Int(Self.BM)
    ]()
    comptime _WarpScratchTileType = TileTensor[
        Self.accum_type,
        type_of(Self._warp_scratch_layout),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    comptime _warp_scratch_size = (2 * Int(Self.num_warps_n) * Int(Self.BM))

    # --- Config delegation (static methods) ---

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        return Self.amd_structured_config.q_head_idx()

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return Self.amd_structured_config.q_tile_idx()

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        return Self.amd_structured_config.kv_head_idx()

    # --- KV tile factory ---

    @staticmethod
    @always_inline
    def make_kv_tile[
        operand_t: MHAOperand,
        //,
    ](
        operand: operand_t,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        kv_tile_num_rows: UInt32,
    ) -> TileTensor[operand_t.dtype, Self.KvTileLayout, ImmutAnyOrigin]:
        return operand.block_paged_tile[Int(Self.BN)](
            batch_idx,
            start_tok_idx,
            head_idx,
            Self.KvTileLayout(
                Coord(
                    RuntimeInt[DType.int64](Int64(kv_tile_num_rows)),
                    Idx[Self.depth](),
                ),
                Coord(Idx[Self._kv_stride0](), Idx[1]()),
            ),
        )

    # --- Constructor ---

    @always_inline
    def __init__(
        out self,
        output_ptr: UnsafePointer[Scalar[Self.output_type], MutAnyOrigin],
        q: UnsafePointer[Scalar[Self.q_type], ImmutAnyOrigin],
        k: Self.k_t,
        v: Self.v_t,
        mask: Self.mask_t,
        sink_weights: OptionalReg[
            UnsafePointer[Scalar[Self.q_type], ImmutAnyOrigin]
        ],
        batch_idx: Int,
        scale: Float32,
        seq_len: Int,
        num_keys: Int,
        start_pos: Int,
        cache_start_pos: Int = 0,
    ):
        # `mla_kv_alias` relies on the `shared_kv` aliasing below to make
        # `v_smem_ptr` a bitcast view of `k_smem_ptr`; without it we'd
        # double-allocate SMEM and PV would read from an empty V buffer.
        comptime assert (
            not Self.mla_kv_alias or Self.amd_structured_config.shared_kv
        ), "mla_kv_alias=True requires shared_kv=True"

        self.softmax = type_of(self.softmax)()
        self.out_reg_buffer = Self.OutputRegisterBufferType()
        self.out_reg_buffer.zero()

        self.k_smem_ptr = stack_allocation[
            Self._max_kv_smem_size if Self.amd_structured_config.shared_kv else Self._k_smem_size,
            Self.k_t.dtype,
            address_space=AddressSpace.SHARED,
            alignment=Self._smem_alignment,
        ]()
        self.v_smem_ptr = self.k_smem_ptr.bitcast[
            Scalar[Self.v_t.dtype]
        ]() if Self.amd_structured_config.shared_kv else stack_allocation[
            Self._v_smem_size,
            Self.v_t.dtype,
            address_space=AddressSpace.SHARED,
            alignment=Self._smem_alignment,
        ]()

        self.warp_scratch_ptr = stack_allocation[
            Self._warp_scratch_size,
            Self.accum_type,
            address_space=AddressSpace.SHARED,
        ]()

        self.p_reg_buffer = Self.PRegisterBufferType(
            tt_stack_allocation[Self.q_type, address_space=AddressSpace.SHARED](
                Self.PRegisterBufferType._smem_layout
            )
        )

        # Pre-offset Q/output TileTensors. `valid_rows` clamps the final
        # prefill row tile; decode always sees exactly `group` rows. The
        # RuntimeInt row dim is what `make_amd_buffer_resource` reads to
        # compute the SRD OOB bound.
        var valid_rows: UInt32 = UInt32(Self.group) if Self.token_gen else min(
            UInt32(Self.BM),
            UInt32(seq_len) - UInt32(Self.q_tile_idx()) * UInt32(Self.BM),
        )
        var q_offset = Self.amd_structured_config.get_q_offset[Self.q_depth]()
        var output_offset = Self.amd_structured_config.get_output_offset[
            Self.output_depth
        ]()

        var q_tile = TileTensor[Self.q_type, Self.QTileLayout, ImmutAnyOrigin](
            ptr=q + Int(q_offset),
            layout=Self.QTileLayout(
                Coord(
                    RuntimeInt[DType.int64](Int64(valid_rows)),
                    Idx[Self.q_depth](),
                ),
                Coord(Idx[Self._q_stride0](), Idx[1]()),
            ),
        )
        self.q_buffer = Self.QRegisterBufferType(q_tile)

        self.output_tile = TileTensor[
            Self.output_type, Self.OutputTileLayout, MutAnyOrigin
        ](
            ptr=output_ptr + Int(output_offset),
            layout=Self.OutputTileLayout(
                Coord(
                    RuntimeInt[DType.int64](Int64(valid_rows)),
                    Idx[Self.output_depth](),
                ),
                Coord(Idx[Self._output_stride0](), Idx[1]()),
            ),
        )

        self.k = k
        self.v = v
        self.mask = mask

        self.mask_block_row = UInt32(self.q_tile_idx() * Self.BM)
        self.warp_row, self.warp_col = get_warp_coords[Self.BN, Self.WN]()
        self.mask_warp_row = UInt32(self.warp_row * Self.WM)
        self.mask_warp_col = UInt32(self.warp_col * Self.WN)

        self.batch_idx = batch_idx

        comptime scaling_factor = (
            log2e if (
                Self.use_exp2 and (not Self.mask_t.apply_log2e_after_mask)
            ) else Scalar[Self.accum_type](1)
        )
        var scale_log2e: Scalar[Self.accum_type] = (
            scale.cast[Self.accum_type]() * scaling_factor
        )

        comptime is_causal_mask = _type_is_eq[Self.mask_t, CausalMask]()

        comptime if is_causal_mask:
            self.scale = readfirstlane(scale_log2e)
        else:
            self.scale = scale_log2e
        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos
        self.kv_start_row = 0

        comptime if Self.sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            # Raw pointer: sink_weights.value()[q_head_idx] → scalar.
            var sink_weight = (
                sink_weights.value()[self.q_head_idx()].cast[Self.accum_type]()
                * log2e
            )
            _ = self.softmax.rowmax_tensor.fill(sink_weight)
            _ = self.softmax.rowsum_tensor.fill(1)
        else:
            _ = self.softmax.rowmax_tensor.fill(
                min_or_neg_inf[Self.accum_type]()
            )
            _ = self.softmax.rowsum_tensor.fill(0)

    # --- Buffer operations ---

    @always_inline
    def zero_p_buffer[stage: Int = 0](self):
        self.p_reg_buffer.zero[stage]()

    @always_inline
    def scale_q_buffer(self):
        """Pre-scale Q registers by scale factor (scale * log2e)."""
        self.q_buffer.scale[Self.accum_type](self.scale)

    @always_inline
    def scale_p_reg[stage: Int = 0](self):
        var p_vec = self.p_reg_buffer.stage_tile[stage]().vectorize[
            1, Self.output_frag_size
        ]()

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                p_vec[mma_id, 0] = p_vec[mma_id, 0] * self.scale

    # --- Mask operations ---

    @always_inline
    def mask_status(
        self,
        kv_tile_start_row: UInt32,
    ) -> TileMaskStatus:
        comptime if Self.token_gen:
            # Decode: check single token at num_keys-1.
            return self.mask.status(
                Index[dtype=DType.uint32](
                    self.num_keys - 1,
                    Int(kv_tile_start_row),
                ),
                Index[dtype=DType.uint32](1, Self.BN),
            )
        else:
            return self.mask.status(
                Index[dtype=DType.uint32](
                    Int(self.mask_block_row + UInt32(self.start_pos)),
                    Int(kv_tile_start_row + UInt32(self.cache_start_pos)),
                ),
                Index[dtype=DType.uint32](Self.BM, Self.BN),
            )

    @always_inline
    def mask_advance(mut self):
        # Decode: no-op (kv_tile_start_row added directly in MaskTileOp.apply).
        comptime if not Self.token_gen:
            self.mask_warp_col += UInt32(Self.BN)

    @always_inline
    def mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    def mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var status = self.mask_status(kv_tile_start_row)
            if self.mask_skip_tile(status):
                self.mask_advance()
                return True
        return False

    @always_inline
    def mask_apply[
        stage: Int = 0
    ](
        mut self,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        not_last_iter: Bool,
    ):
        @always_inline
        @parameter
        def _mask_apply_impl(masked: Bool):
            MaskTileOp[
                accum_type=Self.accum_type,
                token_gen=Self.token_gen,
                mma_shape=Self.mma_shape,
                num_m_mmas=Self.num_m_mmas,
                num_n_mmas=Self.num_n_mmas,
                mask_t=Self.mask_t,
                group=Self.group,
                mma_m=Self.mma_shape[0],
                use_exp2=Self.use_exp2,
            ].apply(
                masked,
                kv_tile_start_row,
                kv_tile_num_rows,
                UInt32(self.start_pos),
                UInt32(self.seq_len),
                UInt32(self.num_keys),
                UInt32(Int(self.mask_block_row)),
                UInt32(Int(self.mask_warp_row)),
                self.mask_warp_col,
                self.scale,
                self.mask,
                self.p_reg_buffer.stage_tile[stage](),
                not_last_iter,
                UInt32(self.cache_start_pos),
            )

        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var mask_status = self.mask_status(kv_tile_start_row)
            _mask_apply_impl(mask_status == TileMaskStatus.PARTIAL_MASK)
        else:
            _mask_apply_impl(masked=True)
        self.mask_advance()

    @always_inline
    def get_num_rows(self) -> UInt32:
        var end = min(
            self.kv_start_row + UInt32(Self.BN), UInt32(self.num_keys)
        )
        var num_rows = max(
            min(Int32(end - self.kv_start_row), Int32(UInt32(Self.BN))), 0
        )
        return UInt32(num_rows)

    @always_inline
    def apply_mask[
        stage: Int, scale: Bool = True
    ](mut self, not_last_iter: Bool = False):
        comptime if scale:
            self.scale_p_reg[stage]()
        var num_rows = self.get_num_rows()
        self.mask_apply[stage](self.kv_start_row, num_rows, not_last_iter)
        self.kv_start_row += UInt32(Self.BN)

    # --- Softmax ---

    @always_inline
    def warp_scratch_tile(self) -> Self._WarpScratchTileType:
        """Warp-reduction scratch tile (decode only)."""
        return Self._WarpScratchTileType(
            self.warp_scratch_ptr, Self._warp_scratch_layout
        )

    @always_inline
    def online_softmax[stage: Int = 0](mut self):
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, self.warp_row)

        self.softmax.full(
            self.out_reg_buffer.reg_tile,
            self.p_reg_buffer.stage_tile[stage](),
            warp_scratch,
        )

    # --- Split online softmax (for MLA double-buffered kernel) ---

    @always_inline
    def online_softmax_step_0[stage: Int, mask: Bool = True](mut self):
        """Step 0: mask + max + exp(even tiles)."""
        comptime if mask:
            self.apply_mask[stage]()
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_tile)

    @always_inline
    def online_softmax_step_1[stage: Int](mut self):
        """Step 1: exp(odd tiles) + sum + correction + update max/sum."""
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp[start=1, stride=2](score_tile)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_step_0_fma[stage: Int, mask: Bool = True](mut self):
        """Step 0 with deferred scaling: mask (no scale), max, exp_scaled.

        Avoids pre-scaling all scores by deferring the scale multiply into the
        exp computation. Uses exp_scaled which subtracts the unscaled max
        first (exact for the maximum element), then scales inside exp2.
        score_frag_rowmax remains unscaled after this call — scale_rowmax is
        deferred to step_1_fma (before calculate_correction needs it).
        """
        comptime if mask:
            self.apply_mask[stage, scale=False]()
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp_scaled[start=0, stride=2](score_tile, self.scale)

    @always_inline
    def online_softmax_step_1_fma[stage: Int](mut self):
        """Step 1 with deferred scaling for odd-indexed tiles.

        Processes remaining score tiles with exp_scaled, then scales the
        rowmax before calculate_correction (which compares against the
        previous iteration's scaled rowmax_tensor).
        """
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp_scaled[start=1, stride=2](score_tile, self.scale)
        self.softmax.scale_rowmax(self.scale)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_step_0_prescaled[
        stage: Int, mask: Bool = True
    ](mut self):
        """Softmax step 0 for pre-scaled Q: no scale needed on scores.

        When Q is pre-multiplied by (scale * log2e), the QK matmul already
        produces scaled scores. We just need mask + max + exp2(score - max).
        """
        comptime if mask:
            self.apply_mask[stage, scale=False]()
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.calculate_qk_max(score_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_tile)

    @always_inline
    def online_softmax_step_1_prescaled[stage: Int](mut self):
        """Softmax step 1 for pre-scaled Q: no scale needed on scores."""
        var warp_scratch = self.warp_scratch_tile().tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_tile = self.p_reg_buffer.stage_tile[stage]()
        self.softmax.exp[start=1, stride=2](score_tile)
        self.softmax.calculate_qk_sum(score_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    def online_softmax_update_output(mut self):
        """Apply correction to output accumulator."""
        self.softmax.update_output(self.out_reg_buffer.reg_tile)

    # --- Output store ---

    @always_inline
    def store_output(self):
        var output_warp_tile = self.output_tile.tile[
            Self.WM, Self.output_depth // Self.num_warps_n
        ](self.warp_row, self.warp_col)
        var writer = RegTileWriter[
            Self.output_type,
            Self.warp_rows,
            Self.warp_cols,
        ](self.output_tile)
        # vectorize[1, 4] works for both MMA sizes:
        #  - 16×16: output_frag_size == 4, so [1, 4] = [frag_num_rows,
        #    output_frag_size] exactly.
        #  - 32×32: the MFMA permutation iterates over 4-register groups;
        #    elem_size must be 4 so `store_mfma32` reads one group at a
        #    time, not the full 16-register fragment.
        #
        # `out_reg_buffer.reg_tile` is laid out m_mma-INNER (row =
        # `n_mma * num_m_mmas + m_mma`) to match the MMA accumulator
        # convention from `mma.mojo` (`c_idx = m_mma + n_mma * num_m_mmas`),
        # whereas `RegTileWriter.store[mfma32]` reads its source tile
        # row-by-row assuming m_mma-OUTER. Materialize the m_mma-th strided
        # slice into a contiguous (num_n_mmas_output, output_frag_size)
        # temp tile and store one 32-row sub-tile per m_mma. For
        # num_m_mmas == 1 the inner copy is an identity that the compiler
        # folds, so a single path covers BM=32 and BM=64.
        comptime sub_layout = row_major[
            Self.num_n_mmas_output, Self.output_frag_size
        ]()
        comptime for m_mma in range(Self.num_m_mmas):
            var sub_warp_tile = output_warp_tile.tile[
                Self.mma_shape[0],
                Self.output_depth // Self.num_warps_n,
            ](m_mma, 0)
            var sub_reg = tt_stack_allocation[
                Self.accum_type, address_space=AddressSpace.LOCAL
            ](sub_layout)
            comptime for n_mma in range(Self.num_n_mmas_output):
                comptime for k in range(Self.output_frag_size):
                    sub_reg[n_mma, k] = self.out_reg_buffer.reg_tile[
                        n_mma * Self.num_m_mmas + m_mma, k
                    ]
            writer.store[mfma32=Self.mma_shape[0] == 32](
                sub_warp_tile.vectorize[1, 4](),
                sub_reg,
            )

    # --- Decode-specific methods ---

    @always_inline
    def copy_fragment_to_smem(self):
        """Copy P scores from registers to shared memory (decode only)."""
        comptime if not Self.token_gen:
            return
        self.p_reg_buffer.copy_to_shared()

    @always_inline
    def store_partition_info(
        self,
        num_partitions: Int,
        exp_sum_ptr: UnsafePointer[Scalar[Self.accum_type], MutAnyOrigin],
        qk_max_ptr: UnsafePointer[Scalar[Self.accum_type], MutAnyOrigin],
    ):
        """Write softmax stats for split-K reduction (decode only).

        With BM > MMA_M (e.g. MLA decode at BM=64, MMA_M=32), each warp
        covers `num_m_mmas = BM/MMA_M` row tiles of the score matrix; row
        stats live in `softmax.rowsum_tensor[m_mma, 0]` per tile. Filter
        to lane_col=0 of warp 0 (the only lanes that hold reduced row
        stats post-softmax) and write one entry per m_mma.
        """
        comptime if not Self.token_gen:
            return

        if num_partitions <= 1:
            return

        # `q_head_idx()` is per-thread for both MHA and MLA: it folds
        # `lane_id % MMA_M` into the tile base, so we just gate on the
        # filter and write. m_mma=0 covers the lane's "natural" head row
        # (rows 0..MMA_M-1 of the BM tile); m_mma=k>0 covers rows k*MMA_M
        # onward, which the same lane also owns when num_m_mmas>1
        # (only happens for MLA decode at BM>=64).
        var head_idx = self.q_head_idx()
        if (
            thread_idx.x < Self.amd_structured_config.heads_per_tile()
            and head_idx < Self.num_heads
        ):
            exp_sum_ptr[head_idx] = self.softmax.rowsum_tensor[0, 0][0]
            qk_max_ptr[head_idx] = self.softmax.rowmax_tensor[0, 0][0]

            comptime for m_mma in range(1, Self.num_m_mmas):
                var head_idx_m = head_idx + m_mma * Self.mma_shape[0]
                if head_idx_m < Self.num_heads:
                    exp_sum_ptr[head_idx_m] = self.softmax.rowsum_tensor[
                        m_mma, 0
                    ][0]
                    qk_max_ptr[head_idx_m] = self.softmax.rowmax_tensor[
                        m_mma, 0
                    ][0]
