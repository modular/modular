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

from std.sys import size_of
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    warp_id,
    thread_idx,
    WARP_SIZE,
    barrier,
)
from std.math import ceildiv, exp2
from std.math.constants import log2e
from std.gpu.primitives import elect_one_sync
from std.gpu.primitives.cluster import cluster_sync
import std.gpu.primitives.warp as warp
from std.gpu.memory import (
    AddressSpace,
    external_memory,
    fence_mbarrier_init,
    fence_async_view_proxy,
)
from std.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.host import DeviceContext, FuncAttribute
from std.ffi import UnsafeUnion
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_cp,
    tcgen05_store_wait,
    tcgen05_fence_after,
    tcgen05_fence_before,
)

from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from nn.attention.mha_mask import MHAMask
from kv_cache.types import KVCacheT
from nn.attention.mha_utils import OptionallyStaticInt, MHAPartitionScheme
from nn.attention.gpu.nvidia.sm90.attention import (
    OptionalPointer,
    NullPointer,
    NonNullPointer,
)

from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.mha import q_num_matrix_view_rows

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from linalg.arch.sm100.mma import smem_descriptor


from layout import TileTensor, row_major, Idx, TensorLayout, Coord
from layout.swizzle import make_swizzle
from layout.tma_async import (
    create_tensor_tile,
    TMATensorTile,
    SharedMemBarrier,
    RaggedTMA3DTile,
    _gather4_box_width,
    _default_desc_shape,
)
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.utils.numerics import min_or_neg_inf


struct MLASparseConfig[qkv_dtype: DType]:
    var num_q_heads: Int
    var num_kv_heads: Int
    var qk_depth: Int
    var v_depth: Int
    var indices_stride: Int
    var group: Int

    # the leftmost q_depth is store in smem,
    # the rightmost q_depth is store in tmem
    # for the leftmost qk_depth mma, we do ss_mma,
    # for the rightmost qk_depth mma, we do ts_mma,
    comptime cta_group = 2
    comptime q_smem_depth = 192
    comptime q_tmem_depth = 384
    comptime B_TOPK = 128
    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime num_threads: Int = 512
    comptime sm100_tmem_cols = 512

    comptime q_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime k_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

    @always_inline
    def __init__(
        out self,
        *,
        num_q_heads: Int,
        num_kv_heads: Int,
        qk_depth: Int,
        v_depth: Int,
        indices_stride: Int,
        group: Int,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_depth = qk_depth
        self.v_depth = v_depth
        self.indices_stride = indices_stride
        self.group = group


struct MLASparseSharedMemory[config: MLASparseConfig]:
    comptime num_q_heads = Self.config.num_q_heads
    comptime qkv_dtype = Self.config.qkv_dtype
    comptime qk_depth = Self.config.qk_depth
    comptime num_mbars = 2

    # per cta dimension
    comptime BH = Self.num_q_heads // Self.config.cta_group
    comptime TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    # Per-CTA TMEM cluster N (= MMA_N for one SV atom). Each CTA holds the
    # full cluster N output for its M slice (M-split, N-shared on output).
    comptime V_DEPTH_PER_CTA = Self.config.v_depth // Self.config.cta_group
    # Per-atom per-CTA V cols.  SV_MMA's b_bmn = MMA_N / cta_group, so each
    # CTA contributes MMA_N/cta_group cols of V per atom to the cluster
    # MMA (B is split across CTAs).
    comptime V_BMN_PER_ATOM = Self.V_DEPTH_PER_CTA // Self.config.cta_group
    # V smem holds BOTH SV atoms' worth of V per CTA — 128 cols for atom1
    # (cluster depths 0..255) followed by 128 cols for atom2 (cluster
    # depths 256..511), so the full v_depth=512 is covered.
    comptime NUM_SV_ATOMS = 2
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS

    comptime FULL_Q_SIZE = Self.BH * Self.qk_depth
    # split num_q_heads per cta
    comptime SHARED_Q_SIZE = Self.BH * Self.config.q_smem_depth
    # split b_topk per cta
    comptime K_SIZE = Self.TOPK_PER_CTA * Self.config.qk_depth
    # V smem: B_TOPK rows × V_SMEM_COLS_PER_CTA cols per CTA.
    comptime V_SIZE = Self.config.B_TOPK * Self.V_SMEM_COLS_PER_CTA
    comptime SHARED_QKV_SIZE = Self.SHARED_Q_SIZE + Self.K_SIZE + Self.V_SIZE
    comptime S_SIZE = Self.BH * Self.config.B_TOPK
    comptime O_SIZE = Self.BH * Self.config.v_depth

    comptime FULL_Q_TYPE = InlineArray[Scalar[Self.qkv_dtype], Self.FULL_Q_SIZE]
    comptime SHARED_QKV_TYPE = InlineArray[
        Scalar[Self.qkv_dtype], Self.SHARED_QKV_SIZE
    ]
    comptime O_TYPE = InlineArray[Scalar[Self.qkv_dtype], Self.O_SIZE]

    var qkvo_union: UnsafeUnion[
        Self.FULL_Q_TYPE, Self.SHARED_QKV_TYPE, Self.O_TYPE
    ]
    var scores: InlineArray[Scalar[Self.qkv_dtype], Self.S_SIZE]
    var p: InlineArray[Float32, Self.S_SIZE]

    var prologue_q: InlineArray[SharedMemBarrier, 1]
    var prologue_q_cp: InlineArray[SharedMemBarrier, 1]

    # use for qk ss_mma completion (used by qk_ss_mma)
    var qk_ss_done: InlineArray[SharedMemBarrier, Self.num_mbars]
    # use for qk ts_mma completion (used by qk_ts_mma)
    var qk_ts_done: InlineArray[SharedMemBarrier, Self.num_mbars]

    # use for sv_p0 completion
    var sv_p0_done: InlineArray[SharedMemBarrier, Self.num_mbars]
    # use for sv_p1 completion
    var sv_p1_done: InlineArray[SharedMemBarrier, Self.num_mbars]

    # use for k_p0 ready TMA load completion (used by k_p0)
    var k_p0_ready: InlineArray[SharedMemBarrier, Self.num_mbars]
    # use for k_p1 ready TMA load completion (used by k_p1)
    var k_p1_ready: InlineArray[SharedMemBarrier, Self.num_mbars]

    # use for v_p0 ready TMA load completion (used by v_p0)
    var v_p0_ready: InlineArray[SharedMemBarrier, Self.num_mbars]
    # use for v_p1 ready TMA load completion (used by v_p1)
    var v_p1_ready: InlineArray[SharedMemBarrier, Self.num_mbars]

    var p_free: InlineArray[SharedMemBarrier, Self.num_mbars]
    var so_ready: InlineArray[SharedMemBarrier, Self.num_mbars]

    # k_valid_ready / k_valid_free coordinate the WG3 warp-13 producer
    # that writes the per-position validity bitmask (`is_k_valid` below)
    # consumed by WG0's mask step.  Mirrors phase1.cuh's `bar_k_valid_*`
    # + `is_k_valid` slot.
    var k_valid_ready: InlineArray[SharedMemBarrier, Self.num_mbars]
    var k_valid_free: InlineArray[SharedMemBarrier, Self.num_mbars]
    # Validity bitmask: 1 bit per topk position.  Packed as
    # MASK_BYTES_PER_BUF = B_TOPK / 8 bytes per buffer; byte `j` of buffer
    # `b` carries bits for positions [j*8 .. j*8+8).  Bit set = "this index
    # is in range AND its absolute position is < topk_lengths[seq]".
    # Stored flat (no nested InlineArray) so the byte-offset math at
    # producer/consumer relies on simple array layout, not on the
    # absence of inter-element padding in nested InlineArray.
    comptime MASK_BYTES_PER_BUF = Self.config.B_TOPK // 8
    # Producer parallelism: one lane per output mask byte; each lane
    # packs INDICES_PER_LANE = 8 bits.  Coupled by definition.
    comptime NUM_KV_VALID_LANES = Self.MASK_BYTES_PER_BUF
    comptime INDICES_PER_LANE = 8
    var is_k_valid: InlineArray[UInt8, Self.num_mbars * Self.MASK_BYTES_PER_BUF]
    var tmem_addr: InlineArray[UInt32, 1]

    # store rowwise max and sum for each threads in a warp group
    var rowwise_max: InlineArray[Float32, WARPGROUP_SIZE]
    var rowwise_sum: InlineArray[Float32, WARPGROUP_SIZE]


struct QKMMAOp[dtype: DType, accum_dtype: DType, config: MLASparseConfig]:
    comptime SSMMAType = SM100TensorAccumulatorSS[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.num_q_heads,
        MMA_N=Self.config.B_TOPK,
        BK=Self.config.q_smem_depth,
        mma_kind=UMMAKind.KIND_F16,
        cta_group=Self.config.cta_group,
    ]
    # 3 stages: BK=384 with MMA_K=16 needs 24 k-mmas, which exceeds the
    # bulk_mma per-instruction limit of 16. Splitting into 3 stages of 8
    # k-mmas each keeps every issued instruction inside the limit. Each
    # `.mma[stage_idx=i]` call issues its share; stages 1+ always
    # accumulate (c_scale forced to 1 internally).
    comptime TSMMAType = SM100TensorAccumulatorTS[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.num_q_heads,
        MMA_N=Self.config.B_TOPK,
        BK=Self.config.q_tmem_depth,
        mma_kind=UMMAKind.KIND_F16,
        cta_group=Self.config.cta_group,
        num_stages=3,
    ]
    comptime NUM_TS_STAGES = 3

    @staticmethod
    @always_inline
    def smem_descriptor_q(
        q_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.num_q_heads // Self.config.cta_group,
            BK=Self.config.q_smem_depth,
            swizzle_mode=Self.config.q_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    def tmem_descriptor_q(
        q_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.num_q_heads // Self.config.cta_group,
            BK=Self.config.q_tmem_depth,
            swizzle_mode=Self.config.q_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    def descriptor_k_p0(
        k_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.B_TOPK // Self.config.cta_group,
            BK=Self.config.q_smem_depth,
            swizzle_mode=Self.config.k_swizzle_mode,
            is_k_major=True,
        ](k_smem)

    @staticmethod
    @always_inline
    def descriptor_k_p1(
        k_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.B_TOPK // Self.config.cta_group,
            BK=Self.config.q_tmem_depth,
            swizzle_mode=Self.config.k_swizzle_mode,
            is_k_major=True,
        ](k_smem)


struct SVMMAType[dtype: DType, accum_dtype: DType, config: MLASparseConfig]:
    # MMA_N is per-atom cluster N.  The UMMA descriptor encodes N>>3 in 6
    # bits (max value 504), so a single atom can't cover MMA_N=512.  We
    # split SV into NUM_SV_ATOMS=2 atoms of MMA_N=v_depth/cta_group=256
    # each (matching phase1.cuh's `SM100_MMA_F16BF16_2x1SM_SS_NOELECT<...,
    # 256, ...>`).  The caller issues both atoms per SV iter into disjoint
    # O TMEM regions (O_TMEM_ADDR and O_TMEM_ADDR_ATOM2).
    # A (S) is written flat by WG0 → swizzle_a=NONE. B (V) is loaded via
    # gather4 with SWIZZLE_128B, producing col-group-blocked smem the
    # SW128 MMA descriptor reads correctly (same pattern as
    # `DecodeSM100PVSS.descriptor_v_block`).
    comptime SS_P0MMAType = SM100TensorAccumulatorSS[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.num_q_heads,
        MMA_N=Self.config.v_depth // Self.config.cta_group,
        BK=Self.config.B_TOPK // 2,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=TensorMapSwizzle.SWIZZLE_NONE,
        swizzle_b=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=False,
        cta_group=Self.config.cta_group,
    ]
    comptime SS_P1MMAType = SM100TensorAccumulatorSS[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.num_q_heads,
        MMA_N=Self.config.v_depth // Self.config.cta_group,
        BK=Self.config.B_TOPK // 2,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=TensorMapSwizzle.SWIZZLE_NONE,
        swizzle_b=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=False,
        cta_group=Self.config.cta_group,
    ]

    # S smem is written by WG0 with a flat (non-swizzled) layout — see the
    # store loop in WG0's per-block softmax that writes `s_bf16` into
    # `scores_ptr` at uint128 strides matching phase1.cuh:166-167. To make
    # the MMA's read interpretation match, the descriptor must declare
    # SWIZZLE_NONE — phase1.cuh uses `Layout_K_INTER_Atom<bf16>` (the
    # "INTER" = interleave-only = non-swizzled atom) for the same reason.
    # Previously this descriptor used SWIZZLE_128B; with our flat writes,
    # the MMA was reading cols permuted by the swizzle XOR and producing
    # wrong O values.
    @staticmethod
    @always_inline
    def descriptor_s(
        s_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.num_q_heads // Self.config.cta_group,
            BK=Self.config.B_TOPK // 2,  # 64 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            is_k_major=True,
        ](s_smem)

    # V smem is loaded via gather4 with `swizzle_mode=SWIZZLE_128B` and
    # `tile_width=V_SMEM_COLS_PER_CTA` (= 256 = 2 atoms × b_bmn).  gather4
    # splits into 4 col-groups (each 64 cols × B_TOPK rows) at smem stride
    # B_TOPK*64.  BMN here = V_BMN_PER_ATOM = MMA_N/cta_group = 128 matches
    # one SV atom's per-CTA b_bmn (atom1 reads smem cols 0..127, atom2
    # reads cols 128..255 with a shifted base pointer).
    @staticmethod
    @always_inline
    def descriptor_v(
        v_smem: UnsafePointer[
            Scalar[Self.dtype], address_space=AddressSpace.SHARED, ...
        ],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=(Self.config.v_depth // Self.config.cta_group)
            // Self.config.cta_group,
            BK=Self.config.B_TOPK // 2,  # 64 K rows per part
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](v_smem)


struct MLAPrefillSparse[
    KVLUTType: MHAOperand,
    output_dtype: DType,
    config: MLASparseConfig,
](TrivialRegisterPassable):
    comptime qkv_dtype = Self.config.qkv_dtype
    comptime accum_dtype = DType.float32

    comptime q_smem_depth = Self.config.q_smem_depth
    comptime q_tmem_depth = Self.config.q_tmem_depth
    comptime qkv_dtype_size = size_of[Self.qkv_dtype]()

    comptime NUM_Q_HEADS_PER_CTA = Self.config.num_q_heads // Self.config.cta_group
    comptime B_TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    comptime V_DEPTH_PER_CTA = Self.config.v_depth // Self.config.cta_group
    # SV is split into NUM_SV_ATOMS=2 atoms (each MMA_N=v_depth/cta_group=256,
    # b_bmn=V_BMN_PER_ATOM=128 per CTA), because MMA_N>504 doesn't fit the
    # 6-bit N field of the UMMA inst descriptor.  V smem holds both atoms'
    # cols per CTA: atom1 = cols 0..127, atom2 = cols 128..255.
    comptime NUM_SV_ATOMS = 2
    comptime V_BMN_PER_ATOM = Self.V_DEPTH_PER_CTA // Self.config.cta_group
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS

    comptime q_tile_shape = Index(
        1, Self.NUM_Q_HEADS_PER_CTA, Self.config.qk_depth
    )
    comptime q_desc_shape = _default_desc_shape[
        3, Self.qkv_dtype, Self.q_tile_shape, Self.config.q_swizzle_mode
    ]()

    comptime k_tile_width = Self.config.qk_depth
    comptime k_swizzle_mode = Self.config.k_swizzle_mode
    comptime k_tile_height = Self.B_TOPK_PER_CTA
    comptime k_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.k_tile_width, Self.k_swizzle_mode
    ]()
    comptime k_tile_shape = Index(Self.k_tile_height, Self.k_gather_box)
    comptime k_desc_shape = Index(1, Self.k_gather_box)

    comptime v_tile_width = Self.V_SMEM_COLS_PER_CTA
    # V uses SWIZZLE_128B; gather4 splits tile_width=128 into 2 col-groups
    # of 64 bf16 each (box_width = 128B / sizeof(bf16) = 64).
    comptime v_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    # tile_height is HALF of B_TOPK: each call to async_copy_gather4_tile
    # loads one K-half (64 rows × V_SMEM_COLS_PER_CTA cols), and load_v
    # invokes it twice with different smem bases. With tile_height=64 the
    # gather4 col-group stride (tile_height*box_w = 4096 elems) matches
    # the SW128 BMN=128 MN-major descriptor's mn_outer stride.
    comptime v_tile_height = Self.config.B_TOPK // 2
    comptime v_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.v_tile_width, Self.v_swizzle_mode
    ]()
    comptime v_tile_shape = Index(Self.v_tile_height, Self.v_gather_box)
    comptime v_desc_shape = Index(1, Self.v_gather_box)

    comptime SMemType = MLASparseSharedMemory[Self.config]
    comptime FULL_Q_TYPE = Self.SMemType.FULL_Q_TYPE
    comptime SHARED_QKV_TYPE = Self.SMemType.SHARED_QKV_TYPE
    comptime O_TYPE = Self.SMemType.O_TYPE

    comptime QKMMAOpType = QKMMAOp[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]
    comptime SVMMAType = SVMMAType[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]

    comptime O_TMEM_ADDR = 0
    # SV atom2's O accumulator sits immediately after atom1's.  Each atom
    # occupies V_BMN_PER_ATOM=128 TMEM cells per lane (MMA_M=128 ×
    # MMA_N=256 cluster / 128 lanes / 2 cta_group = 128 cells per lane).
    comptime O_TMEM_ADDR_ATOM2 = Self.O_TMEM_ADDR + Self.V_BMN_PER_ATOM
    comptime P_TMEM_ADDR = 256
    comptime Q_TMEM_ADDR = 512 - Self.q_tmem_depth // 2

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
    @__name(
        t"mla_prefill_sparse_{Self.qkv_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel[
        TopKLengthLayout: TensorLayout,
        IndicesLayout: TensorLayout,
    ](
        q_tma_op: TMATensorTile[
            Self.qkv_dtype,
            3,
            Self.q_tile_shape,
            Self.q_desc_shape,
        ],
        k_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.k_tile_shape,
            Self.k_desc_shape,
        ],
        v_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.v_tile_shape,
            Self.v_desc_shape,
        ],
        topk_lengths: TileTensor[DType.uint32, TopKLengthLayout, MutAnyOrigin],
        indices: TileTensor[DType.uint32, IndicesLayout, MutAnyOrigin],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        # Per-head attention sink (fp32). Pass a null pointer (default) to
        # skip the sink term entirely; pass a real buffer of size
        # `num_q_heads` to add `exp2(sink_h - mi)` to the softmax normalizer
        # per head.
        attn_sink_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
        indices_stride: Int32,
        # Raw gmem output pointer. Used for the per-thread output store in
        # the WG0 epilogue. RaggedTMA3DTile's box-along-dim-2 stride is
        # `num_heads*v_depth`, which writes BM SEQ positions per call —
        # but our cluster produces 1 seq * BM heads, the wrong shape for
        # that descriptor. Direct global store sidesteps that.
        output_gmem_ptr: UnsafePointer[Scalar[Self.output_dtype], MutAnyOrigin],
    ) where (topk_lengths.flat_rank == 1 and indices.flat_rank == 1):
        var cta_id = UInt32(block_idx.x % Self.config.cta_group)
        var seq_idx = UInt32(block_idx.x // Self.config.cta_group)
        var warp_idx = warp_id()
        var lane_idx = thread_idx.x % WARP_SIZE
        var warpgroup_idx = warp.broadcast(thread_idx.x // WARPGROUP_SIZE)
        var top_k_length = topk_lengths[Idx(seq_idx)]
        var num_k_blocks = max(ceildiv(top_k_length, Self.config.B_TOPK), 1)
        var num_kv_rows = kv_lut.num_kv_rows()
        # Per-query base offset into the indices buffer; each query row owns
        # `indices_stride` indices.
        var indices_base = seq_idx * UInt32(indices_stride)

        if thread_idx.x == 0:
            q_tma_op.prefetch_descriptor()
            k_tma_op.prefetch_descriptor()
            v_tma_op.prefetch_descriptor()

        ref smem = external_memory[
            UInt8, address_space=AddressSpace.SHARED, alignment=128
        ]().bitcast[Self.SMemType]()[]
        ref qkvo_union = smem.qkvo_union

        var full_q_ptr = qkvo_union.unsafe_get[Self.FULL_Q_TYPE]().unsafe_ptr()
        var shared_qkv_ptr = qkvo_union.unsafe_get[
            Self.SHARED_QKV_TYPE
        ]().unsafe_ptr()
        var q_smem_ptr = shared_qkv_ptr
        var v_smem_ptr = shared_qkv_ptr + Self.SMemType.SHARED_Q_SIZE
        var k_smem_ptr = v_smem_ptr + Self.SMemType.V_SIZE
        var o_ptr = qkvo_union.unsafe_get[Self.O_TYPE]().unsafe_ptr()
        var scores_ptr = smem.scores.unsafe_ptr()
        var p_ptr = smem.p.unsafe_ptr()
        var prologue_q_ptr = smem.prologue_q.unsafe_ptr()
        var prologue_q_cp_ptr = smem.prologue_q_cp.unsafe_ptr()
        var qk_ss_done_ptr = smem.qk_ss_done.unsafe_ptr()
        var qk_ts_done_ptr = smem.qk_ts_done.unsafe_ptr()
        var sv_p0_done_ptr = smem.sv_p0_done.unsafe_ptr()
        var sv_p1_done_ptr = smem.sv_p1_done.unsafe_ptr()
        var k_p0_ready_ptr = smem.k_p0_ready.unsafe_ptr()
        var k_p1_ready_ptr = smem.k_p1_ready.unsafe_ptr()
        var v_p0_ready_ptr = smem.v_p0_ready.unsafe_ptr()
        var v_p1_ready_ptr = smem.v_p1_ready.unsafe_ptr()
        var p_free_ptr = smem.p_free.unsafe_ptr()
        var so_ready_ptr = smem.so_ready.unsafe_ptr()
        var k_valid_ready_ptr = smem.k_valid_ready.unsafe_ptr()
        var k_valid_free_ptr = smem.k_valid_free.unsafe_ptr()
        # Byte at offset `buf * MASK_BYTES_PER_BUF + j` holds bits for
        # keys `[j*8, j*8+8)` of buffer `buf`.
        var is_k_valid_ptr = smem.is_k_valid.unsafe_ptr()
        var tmem_addr_ptr = smem.tmem_addr.unsafe_ptr()
        var rowwise_max_ptr = smem.rowwise_max.unsafe_ptr()
        var rowwise_sum_ptr = smem.rowwise_sum.unsafe_ptr()

        var q_full_smem_tensor = TileTensor(
            full_q_ptr,
            row_major[1, Self.NUM_Q_HEADS_PER_CTA, Self.config.qk_depth](),
        )

        if warp_idx == 0:
            if elect_one_sync():
                prologue_q_ptr[].init(1)
                prologue_q_cp_ptr[].init(1)
                comptime for i in range(Self.SMemType.num_mbars):
                    qk_ss_done_ptr[i].init(1)
                    qk_ts_done_ptr[i].init(1)
                    sv_p0_done_ptr[i].init(1)
                    sv_p1_done_ptr[i].init(1)
                    k_p0_ready_ptr[i].init(1)
                    k_p1_ready_ptr[i].init(1)
                    v_p0_ready_ptr[i].init(1)
                    v_p1_ready_ptr[i].init(1)
                    p_free_ptr[i].init(Int32(WARPGROUP_SIZE * 2))
                    so_ready_ptr[i].init(Int32(WARPGROUP_SIZE * 2))
                    k_valid_ready_ptr[i].init(
                        Int32(Self.SMemType.NUM_KV_VALID_LANES)
                    )
                    k_valid_free_ptr[i].init(Int32(WARPGROUP_SIZE))

                fence_mbarrier_init()

        cluster_sync()

        if warp_idx == 0:
            if elect_one_sync():
                # The TMA coord on the head dim is in ELEMENTS, not tiles
                # — passing `cta_id` (0 or 1) made CTA 1 load heads
                # `1..64` instead of `64..127`, shifting CTA 1's Q in
                # smem by -63 heads.  The Q_TMEM dump confirmed this:
                # h=33 (CTA 0 local 33) was duplicated into CTA 1's
                # local row 32 (= position of h=33 in heads 1..64).
                q_tma_op.async_copy[cta_group=Self.config.cta_group](
                    q_full_smem_tensor,
                    prologue_q_ptr[],
                    StaticTuple[UInt32, 3](
                        0,
                        cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA),
                        seq_idx,
                    ),
                )

            tcgen05_alloc[Self.config.cta_group](
                tmem_addr_ptr, Self.config.sm100_tmem_cols
            )
            tcgen05_release_allocation_lock[Self.config.cta_group]()

        barrier()

        if warpgroup_idx == 0:
            warpgroup_reg_alloc[144]()

            var idx_in_wg = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)

            # FlashAttention online-softmax state. Mirrors phase1.cuh:154-164:
            # `mi` is the running max used to scale Pi; `li` is the running
            # sumexp; `real_mi` is the true running max (used only by the
            # all-invalid epilogue case). Both rows of one head are owned by
            # threads (t, t^64) and these three values stay identical between
            # paired threads after every update.
            comptime MAX_INIT_VAL = Float32(-1e30)
            var mi: Float32 = MAX_INIT_VAL
            var li: Float32 = 0.0
            var real_mi: Float32 = Float32(min_or_neg_inf[DType.float32]())

            var scale_log2e = scale * Float32(log2e)
            comptime P_PER_THREAD = Self.config.B_TOPK // 2  # 64
            comptime O_RESCALE_CHUNK = 32
            # Same "per-CTA = D_V/2" convention as in phase1.cuh's
            # rescale loop (`(D_V/2)/CHUNK_SIZE`): we cover the full
            # per-CTA O accumulator (V_DEPTH_PER_CTA cols) in chunks of
            # CHUNK_SIZE. Previously this was double-halved.
            comptime NUM_O_RESCALE_CHUNKS = (
                Self.V_DEPTH_PER_CTA // O_RESCALE_CHUNK
            )

            # Per-thread base offset (in units of bf16x8 = uint128) into
            # the scores smem, matching the K-major SW128B layout the
            # subsequent SV MMA expects. See phase1.cuh:166-167 — base =
            # (idx%64) + 64*((idx/64)*8). Upper-half threads write the
            # left half of S; lower-half threads write the right half.
            var s_smem_uint128_base = (
                scores_ptr.bitcast[SIMD[Self.qkv_dtype, 8]]()
                + (idx_in_wg % UInt32(64))
                + UInt32(64) * ((idx_in_wg / UInt32(64)) * UInt32(8))
            )

            for k in range(num_k_blocks):
                var cur_buf = k % Self.SMemType.num_mbars
                var cur_phase = (k / Self.SMemType.num_mbars) & 1

                # Wait for P = QK^T (TS MMA done).
                qk_ts_done_ptr[cur_buf].wait(cur_phase)
                tcgen05_fence_after()

                # Load P from TMEM (64 fp32 per thread).
                var p = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=P_PER_THREAD,
                    dtype=DType.float32,
                    pack=False,
                    width=P_PER_THREAD,
                ](UInt32(Self.P_TMEM_ADDR))
                tcgen05_load_wait()
                tcgen05_fence_before()
                # P is now in registers; release the P TMEM tile so MMA can
                # overwrite it on the next iteration. p_free is initialized
                # with count = WARPGROUP_SIZE*2 = 256 and the MMA leader
                # waits on CTA 0's instance, so every WG0 thread in *both*
                # cluster CTAs must arrive on CTA 0's barrier. Plain
                # `.arrive()` would only credit the local CTA; use
                # `arrive_cluster(0, 1)` to mirror phase1.cuh:180's
                # `bar_p_free[k%NUM_BUFS].arrive(0u)` (CUTLASS's
                # ClusterTransactionBarrier targeting cta 0).
                p_free_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))

                # Mask step (phase1.cuh:182-210): wait on warp-13's
                # validity bitmask for this k-block, then poison invalid
                # P entries with -inf so they drop out of the softmax.
                # Each thread owns P_PER_THREAD = B_TOPK/2 keys; thread
                # `t<64` reads the low half of the mask, thread `t>=64`
                # the high half.
                comptime MASK_BYTES_PER_BUF = Self.SMemType.MASK_BYTES_PER_BUF
                comptime MASK_BYTES_PER_THREAD = MASK_BYTES_PER_BUF // 2
                k_valid_ready_ptr[cur_buf].wait(cur_phase)
                var mask_byte_base = (
                    Int(cur_buf) * MASK_BYTES_PER_BUF
                    + Int(idx_in_wg // UInt32(64)) * MASK_BYTES_PER_THREAD
                )
                comptime for i in range(P_PER_THREAD):
                    comptime byte_offset = i // 8
                    comptime bit_idx = i % 8
                    var mask_byte = is_k_valid_ptr[mask_byte_base + byte_offset]
                    if ((mask_byte >> UInt8(bit_idx)) & UInt8(1)) == UInt8(0):
                        p[i] = Float32(min_or_neg_inf[DType.float32]())
                _ = k_valid_free_ptr[cur_buf].arrive()

                # Per-thread row max over local P (scaled to log2 domain).
                var cur_pi_max: Float32 = Float32(
                    min_or_neg_inf[DType.float32]()
                )
                comptime for i in range(P_PER_THREAD):
                    cur_pi_max = max(cur_pi_max, rebind[Float32](p[i]))
                cur_pi_max = mul_ftz(cur_pi_max, scale_log2e)

                # Cross-thread max reduction: threads t and t^64 own the
                # same head row of P. Each writes its partial to a small
                # smem buffer, syncs, then reads its peer's value. Two
                # sync points avoid a WAR race.
                named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
                rowwise_max_ptr[idx_in_wg] = cur_pi_max
                named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
                cur_pi_max = max(
                    cur_pi_max,
                    rowwise_max_ptr[idx_in_wg ^ UInt32(64)],
                )
                real_mi = max(real_mi, cur_pi_max)

                # Warp-uniform "should we rescale O?" decision (>6 log2-units
                # means rescaling lifts mass by < 1/64 — phase1.cuh skips
                # the rescale below that threshold to reduce TMEM traffic).
                var should_scale_o = warp.vote[DType.uint32](
                    cur_pi_max - mi > Float32(6.0)
                ) != UInt32(0)

                var new_max: Float32
                var scale_for_old: Float32
                if not should_scale_o:
                    scale_for_old = 1.0
                    new_max = mi
                else:
                    new_max = max(cur_pi_max, mi)
                    scale_for_old = exp2(mi - new_max)
                mi = new_max
                li = mul_ftz(li, scale_for_old)

                # S = exp2(P * scale_log2e - new_max), accumulate li, and
                # convert to bf16 ready for the SV MMA.
                var s_bf16 = InlineArray[Scalar[Self.qkv_dtype], P_PER_THREAD](
                    uninitialized=True
                )
                comptime for i in range(P_PER_THREAD):
                    var d: Float32 = (
                        rebind[Float32](p[i]) * scale_log2e - new_max
                    )
                    var ed: Float32 = exp2(d)
                    li = li + ed
                    s_bf16[i] = ed.cast[Self.qkv_dtype]()

                # Wait until the previous SV MMA has drained the scores
                # smem before overwriting it. (sv_p1_done implies the
                # second half of the prev S@V completed, which is the last
                # use of the prev scores tile.)
                if k > 0:
                    var prev_buf = (k - 1) % Self.SMemType.num_mbars
                    var prev_phase = ((k - 1) / Self.SMemType.num_mbars) & 1
                    sv_p1_done_ptr[prev_buf].wait(prev_phase)

                # Write S to scores smem as 8 bf16 per uint128, stride 64
                # uint128 between writes — exactly the SS-MMA layout.
                comptime for i in range(P_PER_THREAD // 8):
                    s_smem_uint128_base[i * 64] = SIMD[Self.qkv_dtype, 8](
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 0]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 1]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 2]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 3]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 4]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 5]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 6]),
                        rebind[Scalar[Self.qkv_dtype]](s_bf16[i * 8 + 7]),
                    )

                # Rescale O (in TMEM) if mi changed materially. The first
                # iteration (k==0) has no O to scale yet.
                if k > 0 and should_scale_o:
                    tcgen05_fence_after()
                    comptime for chunk_idx in range(NUM_O_RESCALE_CHUNKS):
                        var o_chunk = tcgen05_ld[
                            datapaths=32,
                            bits=32,
                            repeat=O_RESCALE_CHUNK,
                            dtype=DType.float32,
                            pack=False,
                            width=O_RESCALE_CHUNK,
                        ](
                            UInt32(Self.O_TMEM_ADDR)
                            + UInt32(chunk_idx * O_RESCALE_CHUNK)
                        )
                        tcgen05_load_wait()
                        var o_scaled = InlineArray[
                            Scalar[DType.float32], O_RESCALE_CHUNK
                        ](uninitialized=True)
                        comptime for j in range(O_RESCALE_CHUNK):
                            o_scaled[j] = mul_ftz(
                                rebind[Float32](o_chunk[j]),
                                scale_for_old,
                            )
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat=O_RESCALE_CHUNK,
                            pack=False,
                        ](
                            UInt32(Self.O_TMEM_ADDR)
                            + UInt32(chunk_idx * O_RESCALE_CHUNK),
                            o_scaled,
                        )
                    tcgen05_store_wait()
                    tcgen05_fence_before()

                # Make scores smem writes (and any TMEM stores) visible to
                # the SV MMA, then release the so_ready slot for this k.
                # Same cluster-arrive reasoning as p_free above.
                fence_async_view_proxy()
                so_ready_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))

            # ---------------- Epilogue (phase1.cuh:288-386) ----------------

            # All-invalid query case: real_mi stayed -inf, meaning no row
            # ever contributed. Reset li/mi to match the definition that
            # output_scale = 1/(li + exp2(sink - mi)) gives 0 when output
            # is unused.
            if real_mi == Float32(min_or_neg_inf[DType.float32]()):
                li = 0.0
                mi = Float32(min_or_neg_inf[DType.float32]())

            # Cross-thread li sum (paired threads share the row).
            rowwise_sum_ptr[idx_in_wg] = li
            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            li = add_ftz(li, rowwise_sum_ptr[idx_in_wg ^ UInt32(64)])

            # Wait for the final SV MMA to retire before reading O.
            var last_buf = (num_k_blocks - 1) % Self.SMemType.num_mbars
            var last_phase = ((num_k_blocks - 1) / Self.SMemType.num_mbars) & 1
            sv_p1_done_ptr[last_buf].wait(last_phase)
            tcgen05_fence_after()

            # Per-head attention sink. A null pointer is treated as -inf
            # (same as phase1.cuh:315); the sink contributes one extra term
            # exp2(sink_h - mi) in log2 domain to the softmax normalizer.
            var output_scale: Float32
            if Int(attn_sink_ptr) == 0:
                output_scale = 1.0 / li
            else:
                var sink_head_idx = cta_id * UInt32(
                    Self.NUM_Q_HEADS_PER_CTA
                ) + (idx_in_wg % UInt32(64))
                var attn_sink_val = attn_sink_ptr[Int(sink_head_idx)] * Float32(
                    log2e
                )
                output_scale = 1.0 / (li + exp2(attn_sink_val - mi))

            # Guard against deadlocks if some lanes' li==0 (entirely
            # invalid rows): tcgen05_ld below must run uniformly across
            # the warpgroup, so we vote and pick a uniform path.
            var have_valid_indices = warp.vote[DType.uint32](
                li != Float32(0.0)
            ) != UInt32(0)
            if not have_valid_indices:
                output_scale = 1.0

            # Single-load + warp-distributed write, matching the proven
            # pattern in `test_bulk_mma_pair_cta_sm100.mojo`.  Each SV atom
            # produced a per-CTA accumulator of shape (BM=64 heads ×
            # MMA_N=256 cols) stored in TMEM at addresses
            # O_TMEM_ADDR (atom1) and O_TMEM_ADDR_ATOM2 (atom2).  We load
            # 128 fp32 per thread per atom in two 64-cell chunks
            # (splitting reduces register pressure under WG0's 144-reg
            # budget).  Each warp's depth range:
            #   - Warp 0: heads 0..31 of CTA's head range, cluster cols 0..127
            #   - Warp 1: heads 32..63,                    cluster cols 0..127
            #   - Warp 2: heads 0..31,                     cluster cols 128..255
            #   - Warp 3: heads 32..63,                    cluster cols 128..255
            # Atom1 writes those cols to v_depth offsets 0..255; atom2
            # adds V_DEPTH_PER_CTA=256 to land in 256..511.
            var local_warp_idx_wg0 = UInt32(warp_idx)  # 0..3 for WG0
            var head_row_block = local_warp_idx_wg0 % UInt32(2)
            var depth_col_block = local_warp_idx_wg0 // UInt32(2)

            var local_lane = UInt32(lane_idx)  # 0..31 within a warp
            var head_local = head_row_block * UInt32(32) + local_lane
            var head_global = (
                cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA) + head_local
            )
            var depth_col_start = depth_col_block * UInt32(Self.V_BMN_PER_ATOM)

            # gmem byte offset for the start of this thread's row.
            var gmem_row_stride: Int = (
                Self.config.num_q_heads * Self.config.v_depth
            )
            var gmem_row_offset_base: Int = (
                Int(seq_idx) * gmem_row_stride
                + Int(head_global) * Self.config.v_depth
                + Int(depth_col_start)
            )

            comptime for atom_idx in range(Self.NUM_SV_ATOMS):
                # Atom 0: TMEM addr=0,   gmem depth offset = 0
                # Atom 1: TMEM addr=128, gmem depth offset = +V_DEPTH_PER_CTA
                comptime atom_o_tmem_addr = (
                    Self.O_TMEM_ADDR + atom_idx * Self.V_BMN_PER_ATOM
                )
                comptime atom_gmem_depth_shift = (
                    atom_idx * Self.V_DEPTH_PER_CTA
                )

                comptime for chunk in range(2):
                    comptime CHUNK = 64
                    var c_chunk: InlineArray[Scalar[DType.float32], CHUNK]
                    c_chunk = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=CHUNK,
                        dtype=DType.float32,
                        pack=False,
                        width=CHUNK,
                    ](UInt32(atom_o_tmem_addr + chunk * CHUNK))
                    tcgen05_load_wait()

                    # This chunk holds elements 2*(chunk*32) ..
                    # 2*(chunk*32)+63 of the per-thread fragment.  Scale
                    # in fp32 and cast to bf16 ONCE — phase1.cuh:332-334
                    # uses `float2_mul(o, scale)` then a single
                    # `__float22bfloat162_rn`.  Casting first and then
                    # multiplying in bf16 (the old code path) doubled the
                    # rounding count and showed up in `max_err`.
                    comptime for i in range(CHUNK // 2):
                        var v0_f32 = (
                            rebind[Scalar[DType.float32]](c_chunk[2 * i])
                            * output_scale
                        )
                        var v1_f32 = (
                            rebind[Scalar[DType.float32]](c_chunk[2 * i + 1])
                            * output_scale
                        )
                        var v = SIMD[Self.output_dtype, 2](
                            v0_f32.cast[Self.output_dtype](),
                            v1_f32.cast[Self.output_dtype](),
                        )
                        var gmem_offset = (
                            gmem_row_offset_base
                            + atom_gmem_depth_shift
                            + (chunk * (CHUNK // 2) + i) * 2
                        )
                        (output_gmem_ptr + gmem_offset).store[width=2](v)

            if warp_idx == 0:
                tcgen05_dealloc[Self.config.cta_group](
                    tmem_addr_ptr[], Self.config.sm100_tmem_cols
                )

        elif warpgroup_idx == 1:
            # K producer
            warpgroup_reg_dealloc[96]()
            var local_warp_idx = UInt32(warp_id() - 4)

            if elect_one_sync():
                for k in range(num_k_blocks):
                    Self.load_k(
                        k_tma_op,
                        indices,
                        k_smem_ptr,
                        qk_ss_done_ptr,
                        qk_ts_done_ptr,
                        k_p0_ready_ptr,
                        k_p1_ready_ptr,
                        k,
                        cta_id,
                        local_warp_idx,
                        Int32(num_kv_rows),
                        indices_base,
                    )

        elif warpgroup_idx == 2:
            # V producer
            warpgroup_reg_dealloc[96]()
            var local_warp_idx = UInt32(warp_id() - 8)

            if elect_one_sync():
                prologue_q_cp_ptr[].wait()

                for k in range(num_k_blocks):
                    Self.load_v(
                        v_tma_op,
                        v_smem_ptr,
                        sv_p0_done_ptr,
                        sv_p1_done_ptr,
                        v_p0_ready_ptr,
                        v_p1_ready_ptr,
                        indices,
                        k,
                        local_warp_idx,
                        cta_id,
                        indices_base,
                    )

        else:
            warpgroup_reg_alloc[168]()

            # leader CTA and MMA warp
            if cta_id == 0 and warp_idx == 12 and elect_one_sync():
                # use for copying q_tmem from smem to tmem
                var q_tmem_desc = Self.QKMMAOpType.tmem_descriptor_q(
                    q_smem_ptr + Self.SMemType.SHARED_Q_SIZE
                )
                # pair cta only signal leader cta for byte arrival
                prologue_q_ptr[].expect_bytes(
                    Int32(
                        Self.SMemType.FULL_Q_SIZE
                        * Self.config.cta_group
                        * Self.qkv_dtype_size
                    )
                )
                prologue_q_ptr[].wait()
                tcgen05_fence_after()

                Self.cp_q_from_smem_to_tmem(q_tmem_desc, Self.Q_TMEM_ADDR)
                mma_arrive_multicast[cta_group=Self.config.cta_group](
                    prologue_q_cp_ptr,
                    0b11,  # arrive at both ctas in the pair
                )

                for k in range(num_k_blocks + 1):
                    Self.mma(
                        q_smem_ptr,
                        k_smem_ptr,
                        scores_ptr,
                        v_smem_ptr,
                        k_p0_ready_ptr,
                        k_p1_ready_ptr,
                        v_p0_ready_ptr,
                        v_p1_ready_ptr,
                        sv_p0_done_ptr,
                        sv_p1_done_ptr,
                        so_ready_ptr,
                        p_free_ptr,
                        qk_ss_done_ptr,
                        qk_ts_done_ptr,
                        k,
                        num_k_blocks,
                    )

            # KV-valid mask producer (mirrors phase1.cuh's warp 13).
            # `NUM_KV_VALID_LANES` active lanes, each owning
            # `INDICES_PER_LANE = B_TOPK / NUM_KV_VALID_LANES` indices
            # per k-block; the lane packs an 8-bit validity mask and
            # writes it to `is_k_valid[cur_buf][lane]`.  Valid means:
            #   (1) the index is in [0, num_kv_rows), and
            #   (2) its absolute position k*B_TOPK + lane*8 + i is
            #       below the per-query top_k_length (handles padding
            #       to indices_stride for short sequences).
            elif warp_idx == 13 and lane_idx < Self.SMemType.NUM_KV_VALID_LANES:
                Self.kv_valid_producer(
                    indices,
                    is_k_valid_ptr,
                    k_valid_ready_ptr,
                    k_valid_free_ptr,
                    UInt32(lane_idx),
                    indices_base,
                    Int32(num_kv_rows),
                    Int32(top_k_length),
                    Int(num_k_blocks),
                )

    @always_inline
    @staticmethod
    def mma(
        q_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        k_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        s_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        v_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        k_p0_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k_p1_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        v_p0_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        v_p1_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        sv_p0_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        sv_p1_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        so_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        p_free: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        qk_ss_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        qk_ts_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k: UInt32,
        num_k_blocks: UInt32,
    ):
        if k < num_k_blocks:
            # QK^T MMA
            # wait for k load p0
            cur_buf = k % Self.SMemType.num_mbars
            cur_phase = k / Self.SMemType.num_mbars & 1
            prev_buf = (k - 1) % Self.SMemType.num_mbars
            prev_phase = (k - 1) / Self.SMemType.num_mbars & 1

            k_p0_ready[cur_buf].expect_bytes(
                Int32(
                    Self.config.B_TOPK
                    * Self.config.q_smem_depth
                    * Self.qkv_dtype_size
                )
            )
            k_p0_ready[cur_buf].wait(cur_phase)
            if k > 0:
                # Wait for WG0 (the consumer of P) to release the prev P
                # buffer before letting the MMA overwrite it via the next
                # QK GEMM. Previously this incorrectly waited on
                # k_p0_ready[prev_buf], which is the producer barrier for K
                # and a separate state — that race was latent because WG0
                # never arrived on p_free in the original stub.
                p_free[prev_buf].wait(prev_phase)

            tcgen05_fence_after()
            var q_smem_desc = Self.QKMMAOpType.smem_descriptor_q(q_smem_ptr)
            var k_p0_smem_desc = Self.QKMMAOpType.descriptor_k_p0(k_smem_ptr)
            # c_scale=0 mirrors phase1.cuh:539 `utcmma_ss(..., true)` — the
            # first k-mma of the QK SS gemm clears P_TMEM (D = A@B) rather
            # than accumulating onto stale tmem.
            Self.QKMMAOpType.SSMMAType.mma(
                q_smem_desc,
                k_p0_smem_desc,
                Self.P_TMEM_ADDR,
                c_scale=0,
                elect=1,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                qk_ss_done[cur_buf].unsafe_ptr(),
                0b11,  # arrive at both ctas in the pair
            )

            # wait for k load p1
            k_p1_ready[cur_buf].expect_bytes(
                Int32(
                    Self.config.B_TOPK
                    * Self.config.q_tmem_depth
                    * Self.qkv_dtype_size
                )
            )
            k_p1_ready[cur_buf].wait(cur_phase)
            tcgen05_fence_after()

            var k_p1_smem_desc = Self.QKMMAOpType.descriptor_k_p1(
                k_smem_ptr + Self.B_TOPK_PER_CTA * Self.config.q_smem_depth
            )
            # TS MMA is split across NUM_TS_STAGES stages (see
            # TSMMAType definition).  Each stage adds its k-batch to P;
            # stage 0 uses the passed c_scale (here 1 = accumulate onto
            # SS's P), stages 1+ force c_scale=1 internally.
            comptime for stage_idx in range(Self.QKMMAOpType.NUM_TS_STAGES):
                Self.QKMMAOpType.TSMMAType.mma[stage_idx=stage_idx](
                    Self.Q_TMEM_ADDR,
                    k_p1_smem_desc,
                    Self.P_TMEM_ADDR,
                    c_scale=1,
                    elect=1,
                )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                qk_ts_done[cur_buf].unsafe_ptr(),
                0b11,  # arrive at both ctas in the pair
            )
        if k > 0:
            # O += S(i-1)V(i-1)
            curr_buf = (k - 1) % Self.SMemType.num_mbars
            cur_phase = (k - 1) / Self.SMemType.num_mbars & 1

            # SV descriptors for the 2 atoms × 2 key-halves = 4 sub-MMAs.
            # Atom1 reads V smem cols 0..127 (cluster depths 0..255), atom2
            # reads cols 128..255 (cluster depths 256..511).  Within a
            # key-half, atom2's base is shifted by B_TOPK/2 *
            # V_BMN_PER_ATOM elements (= 8192 for our dims) — that's the
            # 2-cg sub-tile boundary the v loader produces (cgs 0,1 =
            # atom1; cgs 2,3 = atom2 within each key-half).
            comptime ATOM2_COL_OFFSET = (
                Self.config.B_TOPK // 2 * Self.V_BMN_PER_ATOM
            )
            comptime KEY_HALF_OFFSET = (
                Self.config.B_TOPK // 2 * Self.V_SMEM_COLS_PER_CTA
            )
            var s_p0_smem_desc = Self.SVMMAType.descriptor_s(s_smem_ptr)
            var s_p1_smem_desc = Self.SVMMAType.descriptor_s(
                s_smem_ptr + Self.NUM_Q_HEADS_PER_CTA * Self.config.B_TOPK // 2
            )
            var v_atom1_p0_desc = Self.SVMMAType.descriptor_v(v_smem_ptr)
            var v_atom1_p1_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + KEY_HALF_OFFSET
            )
            var v_atom2_p0_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + ATOM2_COL_OFFSET
            )
            var v_atom2_p1_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + KEY_HALF_OFFSET + ATOM2_COL_OFFSET
            )

            so_ready[curr_buf].wait(cur_phase)

            # Cluster total bytes = cta_group * per-CTA bytes for ONE
            # key-half = cta_group * (B_TOPK/2 * V_SMEM_COLS_PER_CTA *
            # sizeof) = (B_TOPK/2) * (V_DEPTH_PER_CTA * cta_group) *
            # sizeof = (B_TOPK/2) * v_depth * sizeof.  V_SMEM_COLS_PER_CTA
            # now holds both atoms' cols, so the cluster bytes correspond
            # to one key-half of *both* atoms together.
            v_p0_ready[curr_buf].expect_bytes(
                Int32(
                    Self.config.B_TOPK
                    // 2
                    * Self.config.v_depth
                    * Self.qkv_dtype_size
                )
            )
            v_p0_ready[curr_buf].wait(cur_phase)
            # Mirrors phase1.cuh:565 — TMEM fence after the v_part0 wait
            # ensures the SV P0 MMA's read of O_TMEM (under c_scale=1 for
            # k>1) sees the rescaled O written by the softmax warpgroup,
            # not stale TMEM.  Bar_so_ready only orders smem traffic
            # (fence_async_view_proxy), not TMEM.
            tcgen05_fence_after()

            # 2 SV atoms × 2 key-halves = 4 SS_MMA calls per SV iter.
            # accum_init (c_scale=0) is per-atom on the first SV iter
            # (k==1); subsequent iters accumulate.  Both atoms init
            # independently because they write to disjoint O TMEM regions.
            var sv_p0_c_scale: UInt32 = 0 if k == 1 else 1
            Self.SVMMAType.SS_P0MMAType.mma(
                s_p0_smem_desc,
                v_atom1_p0_desc,
                Self.O_TMEM_ADDR,
                c_scale=sv_p0_c_scale,
                elect=1,
            )
            Self.SVMMAType.SS_P0MMAType.mma(
                s_p0_smem_desc,
                v_atom2_p0_desc,
                UInt32(Self.O_TMEM_ADDR_ATOM2),
                c_scale=sv_p0_c_scale,
                elect=1,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                sv_p0_done[curr_buf].unsafe_ptr(),
                0b11,  # arrive at both ctas in the pair
            )

            v_p1_ready[curr_buf].expect_bytes(
                Int32(
                    Self.config.B_TOPK
                    // 2
                    * Self.config.v_depth
                    * Self.qkv_dtype_size
                )
            )
            v_p1_ready[curr_buf].wait(cur_phase)
            tcgen05_fence_after()

            # SV P1 always accumulates onto O (phase1.cuh:574,
            # `accum_init=false`).
            Self.SVMMAType.SS_P1MMAType.mma(
                s_p1_smem_desc,
                v_atom1_p1_desc,
                Self.O_TMEM_ADDR,
                c_scale=1,
                elect=1,
            )
            Self.SVMMAType.SS_P1MMAType.mma(
                s_p1_smem_desc,
                v_atom2_p1_desc,
                UInt32(Self.O_TMEM_ADDR_ATOM2),
                c_scale=1,
                elect=1,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                sv_p1_done[curr_buf].unsafe_ptr(),
                0b11,  # arrive at both ctas in the pair
            )

    @always_inline
    @staticmethod
    def cp_q_from_smem_to_tmem(
        smem_desc: MMASmemDescriptorPair,
        tmem_addr: UInt32,
    ):
        # each cta holds 64 x (q_smem_depth + 384)
        # we do 64x128bit tcgen05_cp
        # break down 384 to 6 64 col tiles, each tcgen05_cp copies 64x8
        # so we are essentially doing 64x(6x8x8)
        comptime NUM_Q_CP_TILES = Self.q_tmem_depth // 64
        comptime NUM_SUB_TILES = 64 // 8
        comptime for tile_id in range(NUM_Q_CP_TILES):
            comptime for sub_tile_id in range(NUM_SUB_TILES):
                # tile_id stride is *Q-row* stride (num head rows per
                # CTA × 64 col atom), NOT K-row stride.  B_TOPK_PER_CTA
                # happens to equal NUM_Q_HEADS_PER_CTA in the current
                # config (both = 64) but the semantically correct name is
                # the Q-side one.
                comptime sub_tile_offset = (
                    tile_id * Self.NUM_Q_HEADS_PER_CTA * 64 + sub_tile_id * 8
                ) * Self.qkv_dtype_size
                comptime TMEM_OFFSET = tile_id * 32 + sub_tile_id * 4
                var sub_tile_desc = smem_desc + UInt32(sub_tile_offset)
                # Multicast pattern matches phase1.cuh
                # `SM100_UTCCP_2x64dp128bitlw0213_2cta::copy` (the "0213"
                # suffix = warpx2::02_13).  The cp_q tile/sub-tile loop
                # math also mirrors phase1.cuh's: tile_idx stride =
                # 8192 bytes (= 64 rows × 64 BF16 × 2), sub_tile stride
                # = 16 bytes (= 8 BF16 × 2).
                tcgen05_cp[
                    cta_group=Self.config.cta_group,
                    datapaths=64,
                    bits=128,
                    multicast="warpx2::02_13",
                ](
                    tmem_addr + UInt32(TMEM_OFFSET),
                    sub_tile_desc.descriptor(),
                )

    @always_inline
    @staticmethod
    def k_tma_gather4_load[
        col_range: Tuple[UInt32, UInt32],
        num_rows: Int,
    ](
        tma_op: TMATensorTile[Self.qkv_dtype, 2, _, _],
        smem_barrier: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        smem_tensor: TileTensor[
            Self.qkv_dtype,
            address_space=AddressSpace.SHARED,
            ...,
        ],
        local_indices: InlineArray[SIMD[DType.int32, 4], num_rows],
        warp_idx: UInt32,
    ):
        # layout for complying to 128B swizzle atom
        comptime col_dim = smem_tensor.static_shape[1]
        comptime num_col_tiles = col_dim // 64
        comptime row_dim = smem_tensor.static_shape[0]
        tma_smem_tensor = TileTensor(
            smem_tensor.ptr,
            row_major[row_dim * num_col_tiles, 64](),
        )

        comptime outer_row_start = col_range[0] // 64
        comptime outer_row_end = col_range[1] // 64
        comptime for outer_row in range(outer_row_start, outer_row_end):
            var tma_tile = tma_smem_tensor.tile[row_dim, 64](
                Coord(Idx[Int(outer_row)](), Idx[0]())
            )
            comptime for inner_row in range(64 // 16):
                var inner_tma_tile = tma_tile.tile[16, 64](
                    Coord(Idx[Int(inner_row)](), Idx[0]())
                )
                var inner_warp_dist = inner_tma_tile.tile[4, 64](
                    Coord(Idx(warp_idx), Idx[0]())
                )
                var indices = local_indices[inner_row]
                tma_op.async_copy_gather4[cta_group=Self.config.cta_group](
                    inner_warp_dist,
                    smem_barrier[],
                    Int32(outer_row * 64),
                    indices[0],
                    indices[1],
                    indices[2],
                    indices[3],
                )

    @always_inline
    @staticmethod
    def v_tma_gather4_load[
        local_row_range: Tuple[Int, Int],
    ](
        tma_op: TMATensorTile[Self.qkv_dtype, 2, _, _],
        smem_barrier: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        smem_tensor: TileTensor[
            Self.qkv_dtype,
            address_space=AddressSpace.SHARED,
            ...,
        ],
        indices: TileTensor[
            DType.uint32, address_space=AddressSpace.GENERIC, ...
        ],
        warp_idx: UInt32,
        k: UInt32,
        cta_id: UInt32,
        indices_base: UInt32,
    ):
        # `local_row_range` is in (local_row) units (each local_row =
        # 4 token-rows × 4 warps = 16 rows of progress). With SWIZZLE_128B
        # and tile_width=V_SMEM_COLS_PER_CTA=128, gather4 box_width=64
        # forces 2 col-group calls per 4-row chunk. tile_height=B_TOPK/2
        # so each col-group occupies tile_height*64 = 4096 elems, matching
        # the BMN=128 SW128 descriptor's mn_outer stride. The smem base
        # passed by the caller selects which K-half this batch writes to.
        comptime row_start = local_row_range[0]
        comptime row_end = local_row_range[1]
        comptime num_warps = 4
        comptime gather_box = Self.v_gather_box  # 64 for SW128 bf16
        comptime num_col_groups = (
            Self.V_SMEM_COLS_PER_CTA // Self.v_gather_box
        )
        comptime num_cgs_per_atom = (Self.V_BMN_PER_ATOM // Self.v_gather_box)
        # Each warp owns 4 contiguous rows within every 16-row stripe.
        # Within a col-group: 4-row stride = 4 * gather_box = 256.
        var v_smem_base = smem_tensor.ptr + warp_idx * UInt32(
            4 * Self.v_gather_box
        )

        comptime for local_row in range(row_start, row_end):
            var token_idx_v4 = indices.load[width=4](
                Coord(
                    Idx(
                        indices_base
                        + k * Self.config.B_TOPK
                        + (UInt32(local_row) * UInt32(num_warps) + warp_idx)
                        * UInt32(4)
                    )
                )
            ).cast[DType.int32]()
            comptime for cg in range(num_col_groups):
                # local_row stride within col-group: 16 rows × 64 cols
                # = 1024 elems. Col-group stride: tile_height*gather_box.
                # smem_offset is relative to the batch base (`smem_tensor.ptr`)
                # so the second call (row_start=HALF_LOCAL_ROWS) maps
                # local_row=HALF_LOCAL_ROWS → smem_offset=0 within the
                # second-batch smem region.
                comptime smem_offset = (
                    (local_row - row_start)
                    * (4 * num_warps)
                    * Self.v_gather_box
                    + cg * Self.v_tile_height * Self.v_gather_box
                )
                # Interleaved gmem → smem mapping for 2 SV atoms:
                #   atom_idx = cg // num_cgs_per_atom, local_cg = cg % num_cgs_per_atom
                #   gmem_col = atom_idx * V_DEPTH_PER_CTA
                #             + cta_id  * V_BMN_PER_ATOM
                #             + local_cg * gather_box
                # The descriptor for atom1 (base = v_smem) sees smem cgs 0..1
                # which span cluster depths 0..255 (CTA0 cols 0..127 + CTA1
                # cols 128..255).  Atom2 (base shifted by 8192 elements) sees
                # smem cgs 2..3 = cluster depths 256..511.
                comptime atom_idx = cg // num_cgs_per_atom
                comptime local_cg = cg % num_cgs_per_atom
                tma_op.async_copy_gather4[cta_group=Self.config.cta_group](
                    TileTensor(v_smem_base + smem_offset, smem_tensor.layout),
                    smem_barrier[],
                    Int32(atom_idx * Self.V_DEPTH_PER_CTA)
                    + Int32(cta_id * UInt32(Self.V_BMN_PER_ATOM))
                    + Int32(local_cg * Self.v_gather_box),
                    token_idx_v4[0],
                    token_idx_v4[1],
                    token_idx_v4[2],
                    token_idx_v4[3],
                )

    @always_inline
    @staticmethod
    def kv_valid_producer(
        indices: TileTensor[
            DType.uint32, address_space=AddressSpace.GENERIC, ...
        ],
        is_k_valid_ptr: UnsafePointer[
            mut=True, UInt8, address_space=AddressSpace.SHARED, ...
        ],
        k_valid_ready_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k_valid_free_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        lane_idx: UInt32,
        indices_base: UInt32,
        num_kv_rows: Int32,
        top_k_length: Int32,
        num_k_blocks: Int,
    ):
        # NUM_KV_VALID_LANES active lanes × INDICES_PER_LANE indices
        # = B_TOPK indices/k-block.  The bit-pack matches
        # phase1.cuh:583-598's `is_ks_valid_mask`.
        comptime INDICES_PER_LANE = Self.SMemType.INDICES_PER_LANE
        comptime MASK_BYTES_PER_BUF = Self.SMemType.MASK_BYTES_PER_BUF
        for k_block in range(num_k_blocks):
            var cur_buf = UInt32(k_block) % UInt32(Self.SMemType.num_mbars)
            # WG0 starts in phase 0 waiting for k_valid_ready; warp 13
            # is the producer, so it waits on k_valid_free with the
            # XOR-flipped phase (initial wait returns immediately since
            # the bar is fresh).  Matches phase1.cuh's
            # `wait((k/NUM_BUFS)&1^1)`.
            var free_phase = (
                UInt32(k_block) // UInt32(Self.SMemType.num_mbars)
            ) & UInt32(1) ^ UInt32(1)

            # Issue the gmem indices load + mask compute BEFORE waiting on
            # k_valid_free, so the producer overlaps gmem latency with the
            # consumer's prior iteration.  The result sits in registers
            # across the wait.
            var gidx_offset = (
                indices_base
                + UInt32(k_block) * UInt32(Self.config.B_TOPK)
                + lane_idx * UInt32(INDICES_PER_LANE)
            )
            # Sentinel-by-design: `indices` is uint32 in gmem; the cast to
            # int32 here is what makes the padding sentinel `0xFFFFFFFF`
            # alias to `-1` and fail the `idx_i >= 0` check below.  Assumes
            # `num_kv_rows` fits in signed int32 (~2B rows); far above any
            # realistic deployment.
            var idx_v8 = indices.load[width=INDICES_PER_LANE](
                Coord(Idx(gidx_offset))
            ).cast[DType.int32]()

            var abs_pos_base = Int32(k_block) * Int32(
                Self.config.B_TOPK
            ) + Int32(lane_idx) * Int32(INDICES_PER_LANE)
            var mask: UInt8 = 0
            comptime for i in range(INDICES_PER_LANE):
                var idx_i = idx_v8[i]
                var abs_pos = abs_pos_base + Int32(i)
                if (
                    idx_i >= Int32(0)
                    and idx_i < num_kv_rows
                    and abs_pos < top_k_length
                ):
                    mask = mask | (UInt8(1) << UInt8(i))

            k_valid_free_ptr[Int(cur_buf)].wait(free_phase)
            is_k_valid_ptr[
                Int(cur_buf) * MASK_BYTES_PER_BUF + Int(lane_idx)
            ] = mask
            _ = k_valid_ready_ptr[Int(cur_buf)].arrive()

    @always_inline
    @staticmethod
    def load_k(
        k_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.k_tile_shape,
            Self.k_desc_shape,
        ],
        indices: TileTensor[
            DType.uint32, address_space=AddressSpace.GENERIC, ...
        ],
        k_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        qk_ss_done: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        qk_ts_done: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k_p0_ready: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k_p1_ready: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        k: UInt32,
        cta_id: UInt32,
        warp_idx: UInt32,
        num_kv_rows: Int32,
        indices_base: UInt32,
    ):
        # we break down B_TOPK // cta_group into groups of 16,
        # each group is loaded by 4 warps, each warp loads 4 rows
        comptime num_warps = 4
        comptime num_rows_per_warp = (
            Self.config.B_TOPK // Self.config.cta_group
        ) // 4 // num_warps

        var local_indices = InlineArray[
            SIMD[DType.int32, 4], num_rows_per_warp
        ](uninitialized=True)
        var max_idx: Int32 = -1
        var min_idx: Int32 = num_kv_rows

        var k_smem_tensor = TileTensor(
            k_smem_ptr,
            row_major[Self.B_TOPK_PER_CTA, Self.config.qk_depth](),
        )

        comptime for local_row in range(num_rows_per_warp):
            # Each (local_row, warp_idx) reads a disjoint 4-int chunk. The
            # ×4 multiplier matches phase1.cuh:401 — that line walks gIndices
            # as an `int4*` (stride 4 ints per pointer step), so successive
            # warps' chunks are 4 ints apart, not 1.
            var indices_offset = (
                indices_base
                + k * Self.config.B_TOPK
                + cta_id * (Self.config.B_TOPK // Self.config.cta_group)
                + (UInt32(local_row) * num_warps + warp_idx) * UInt32(4)
            )
            local_indices[local_row] = indices.load[width=4](
                Coord(Idx(indices_offset))
            ).cast[DType.int32]()
            max_idx = max(max_idx, local_indices[local_row].reduce_max())
            min_idx = min(min_idx, local_indices[local_row].reduce_min())

        var all_inval = min_idx == num_kv_rows or max_idx == -1
        # `>=` not `>`: skipping is only safe once both `num_mbars`
        # pipeline buffers have been filled at least once; otherwise an
        # initial all-invalid block could leave the K buffer in an
        # uninitialized (NaN-poisoned) state for a later valid block.
        var skip_tma = all_inval and k >= Self.SMemType.num_mbars

        var curr_buf = k % Self.SMemType.num_mbars
        var prev_buf = (k - 1) % Self.SMemType.num_mbars
        var prev_phase = (k - 1) / Self.SMemType.num_mbars & 1

        if k > 0:
            qk_ss_done[prev_buf].wait(prev_phase)

        if not skip_tma:
            Self.k_tma_gather4_load[
                (UInt32(0), UInt32(Self.config.q_smem_depth)),
            ](
                k_tma_op,
                k_p0_ready + curr_buf,
                k_smem_tensor,
                local_indices,
                warp_idx,
            )
        else:
            # Skip the TMA and credit the K-load-ready barrier directly so
            # the MMA's outstanding `expect_bytes` is satisfied. (The original
            # branch credited `qk_ss_done` here, but that barrier is the MMA's
            # producer barrier and is signalled by `mma_arrive_multicast` —
            # crediting it from the loader corrupts the MMA handshake.)
            k_p0_ready[curr_buf].complete_transaction(
                0,
                Int32(
                    num_rows_per_warp
                    * 4
                    * Self.config.q_smem_depth
                    * Self.qkv_dtype_size
                ),
                1,
            )

        if k > 0:
            qk_ts_done[prev_buf].wait(prev_phase)

        if not skip_tma:
            Self.k_tma_gather4_load[
                (
                    UInt32(Self.config.q_smem_depth),
                    UInt32(Self.config.qk_depth),
                ),
            ](
                k_tma_op,
                k_p1_ready + curr_buf,
                k_smem_tensor,
                local_indices,
                warp_idx,
            )
        else:
            k_p1_ready[curr_buf].complete_transaction(
                0,
                Int32(
                    num_rows_per_warp
                    * 4
                    * Self.config.q_tmem_depth
                    * Self.qkv_dtype_size
                ),
                1,
            )

    @always_inline
    @staticmethod
    def load_v(
        v_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.v_tile_shape,
            Self.v_desc_shape,
        ],
        v_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=AddressSpace.SHARED, ...
        ],
        sv_p0_done: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        sv_p1_done: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        v_p0_ready: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        v_p1_ready: UnsafePointer[
            SharedMemBarrier, address_space=AddressSpace.SHARED, ...
        ],
        indices: TileTensor[
            DType.uint32, address_space=AddressSpace.GENERIC, ...
        ],
        k: UInt32,
        warp_idx: UInt32,
        cta_id: UInt32,
        indices_base: UInt32,
    ):
        var curr_buf = k % Self.SMemType.num_mbars
        var prev_buf = (k - 1) % Self.SMemType.num_mbars
        var prev_phase = (k - 1) / Self.SMemType.num_mbars & 1
        comptime num_warps = 4

        # Per-batch tensor descriptors: each K-half writes into its own
        # tile_height=B_TOPK/2 region of V smem, sized to match the
        # gather4 layout the SW128 MMA descriptor expects.
        var v_smem_tensor_p0 = TileTensor(
            v_smem_ptr,
            row_major[Self.v_tile_height, Self.V_SMEM_COLS_PER_CTA](),
        )
        # Batch 1 base = v_smem + v_tile_height * V_SMEM_COLS_PER_CTA
        # (= 16384 elems: 64 rows × 256 cols × 1 element, where 256 cols
        # covers both SV atoms' contributions per CTA).
        var v_smem_tensor_p1 = TileTensor(
            v_smem_ptr + Self.v_tile_height * Self.V_SMEM_COLS_PER_CTA,
            row_major[Self.v_tile_height, Self.V_SMEM_COLS_PER_CTA](),
        )

        # Per-warp work within one batch: (B_TOPK/2)/4 rows-per-gather4
        # / NUM_WARPS = HALF_LOCAL_ROWS local rows per warp per batch.
        comptime HALF_LOCAL_ROWS = (Self.config.B_TOPK // 2 // 4 // 4)

        # Split B_TOPK so the second load pipelines with the first S@V.
        if k > 0:
            sv_p0_done[prev_buf].wait(prev_phase)

        # K-half 0 (rows 0..63 in V) → writes to batch-0 smem region.
        Self.v_tma_gather4_load[(0, HALF_LOCAL_ROWS)](
            v_tma_op,
            v_p0_ready + curr_buf,
            v_smem_tensor_p0,
            indices,
            warp_idx,
            k,
            cta_id,
            indices_base,
        )

        if k > 0:
            sv_p1_done[prev_buf].wait(prev_phase)

        # K-half 1 (rows 64..127 in V) → writes to batch-1 smem region.
        # local_row offset relative to the batch so smem_offset computes
        # correctly: pass (HALF_LOCAL_ROWS, 2*HALF_LOCAL_ROWS) and let
        # the inner function compute indices_base offsets via the
        # k * B_TOPK + local_row * 16 formula (which already covers all
        # B_TOPK indices when local_row spans 0..2*HALF_LOCAL_ROWS-1).
        Self.v_tma_gather4_load[(HALF_LOCAL_ROWS, 2 * HALF_LOCAL_ROWS)](
            v_tma_op,
            v_p1_ready + curr_buf,
            v_smem_tensor_p1,
            indices,
            warp_idx,
            k,
            cta_id,
            indices_base,
        )


@always_inline
def mla_prefill_sparse[
    output_dtype: DType,
    q_type: DType,
    cache_t: KVCacheT,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
](
    output: TileTensor[output_dtype, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[DType.uint32, address_space=AddressSpace.GENERIC, ...],
    topk_lengths: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    attn_sink_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    scale: Float32,
    indices_stride: Int32,
    ctx: DeviceContext,
) raises:
    comptime assert q_depth == config.qk_depth
    # DSv3.2 absorbed prefill is the only supported shape: qk_depth =
    # kv_lora_rank (512) + qk_rope_head_dim (64) = 576. The kernel hardcodes
    # q_smem_depth=192 / q_tmem_depth=384 summing to 576; other depths would
    # silently mis-stride the TMA copies.
    comptime assert config.qk_depth == 576
    comptime assert config.num_q_heads == 128
    comptime assert config.num_kv_heads == 1
    # The output smem buffer is allocated as `qkv_dtype` (it shares the
    # smem union with Q/K/V). We bitcast it to `output_dtype` at the TMA
    # store site, which is only sound if the two dtypes have the same
    # bit width.
    comptime assert size_of[output_dtype]() == size_of[q_type]()

    # num_q_rows == batch_size * q_seq_len
    var num_q_rows = q_num_matrix_view_rows(q)

    var kv_operand = KVCacheMHAOperand(kv_cache)

    # we do 2CTA MMA for all the heads in each q token
    # CTA0 loads the upper half of the heads
    # CTA1 loads the lower half of the heads
    q_tma_op = create_tensor_tile[
        Index(1, config.num_q_heads // 2, q_depth),
        swizzle_mode=config.q_swizzle_mode,
    ](ctx, q)

    # for 2CTA MMA B_TOPK == 128
    # we load 64 tokens gathered from different kv blocks for each CTA
    # Both CTA in the pair load 64 into their peer
    # so each CTA will have 128 topk tokens
    k_tma_op = kv_operand.create_gather4_tma_tile[
        tile_width=config.qk_depth,
        tile_stride=config.qk_depth,
        swizzle_mode=config.k_swizzle_mode,
        tile_height=config.B_TOPK // config.cta_group,
    ](ctx)

    # V is loaded with SWIZZLE_128B. With tile_width=V_SMEM_COLS_PER_CTA=
    # 128 and box_width=64, gather4 produces 2 col-groups per CTA, placed
    # at smem offsets 0 and B_TOPK*64 respectively. The per-CTA SV MMA
    # descriptor (descriptor_v) reads this layout via BMN =
    # V_SMEM_COLS_PER_CTA, BK = B_TOPK/2, MN-major, SWIZZLE_128B (matching
    # the decode kernel's `DecodeSM100PVSS.descriptor_v_block` pattern).
    #
    # tile_stride is the gmem row stride (qk_depth=576). tile_width is
    # V_SMEM_COLS_PER_CTA = MMA_N / cta_group; each CTA contributes its
    # slice of V to the cluster MMA, with `col_idx = cta_id *
    # V_SMEM_COLS_PER_CTA` selecting its V col range.
    v_tma_op = kv_operand.create_gather4_tma_tile[
        tile_width=config.v_depth // config.cta_group // config.cta_group,
        tile_stride=config.qk_depth,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        tile_height=config.B_TOPK // 2,
    ](ctx)

    # Output is laid out as [num_q_rows, num_q_heads, v_depth]. Each CTA
    # Output is written via direct gmem store from WG0's epilogue, not
    # TMA — `RaggedTMA3DTile`'s box-along-dim-2 stride is `num_heads *
    # v_depth`, which writes BM SEQ positions per call, but our cluster
    # produces 1 seq * BM heads, the wrong shape for that descriptor.

    comptime assert type_of(topk_lengths).flat_rank == 1
    comptime assert type_of(indices).flat_rank == 1
    comptime kernel = MLAPrefillSparse[
        KVLUTType=type_of(kv_operand),
        output_dtype=output_dtype,
        config=config,
    ].kernel[
        type_of(topk_lengths).LayoutType,
        type_of(indices).LayoutType,
    ]

    comptime smem_size = size_of[MLASparseSharedMemory[config]]()

    ctx.enqueue_function[kernel](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        topk_lengths,
        indices,
        kv_operand,
        scale,
        attn_sink_ptr,
        indices_stride,
        output.ptr,
        grid_dim=(config.cta_group * num_q_rows, 1, 1),
        block_dim=(config.num_threads, 1, 1),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
