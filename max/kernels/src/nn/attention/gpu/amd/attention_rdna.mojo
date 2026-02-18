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
"""RDNA-specific Attention struct for Wave32 WMMA operations.

This module provides an Attention implementation optimized for AMD RDNA consumer
GPUs (Radeon RX 7000/8000 series) using Wave32 WMMA instructions.

Key differences from CDNA Attention:
- Wave size: 32 lanes (vs 64 for CDNA)
- MMA shape: 16x16x16 only (vs multiple shapes for CDNA)
- Fragment sizes: A/B = 16 elements, C/D = 8 elements per lane
- k_group_size = 1 (single MMA per K iteration)
"""

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e

from sys import size_of, simd_width_of

from algorithm.functional import unswitch
from gpu import barrier, block_idx, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from layout import Layout, LayoutTensor
from layout._utils import idx2crd, make_amd_buffer_resource
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import blocked_product
from layout.layout_tensor import (
    ThreadScope,
    copy_dram_to_local,
    copy_local_to_dram,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from memory import stack_allocation
from memory.pointer import AddressSpace as BaseAddressSpace
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import _online_softmax_iter_for_mma_output

from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf

from .attention import AttentionConfig
from .buffers import KVBuffer
from .buffers_rdna import (
    KBufferRDNA,
    VBufferRDNA,
    QRegisterBufferRDNA,
    OutputRegisterBufferRDNA,
    PRegisterBufferRDNA,
    RDNA_MMA_M,
    RDNA_MMA_N,
    RDNA_MMA_K,
    RDNA_WARP_SIZE,
    RDNA_AB_FRAG_SIZE,
    RDNA_CD_FRAG_SIZE,
    get_rdna_fragment_layout,
    get_rdna_warp_layout,
)
from .mma_rdna import mma_rdna
from .utils import (
    GlobalMemoryManager,
    LocalLayoutTensor,
    SharedLayoutTensor,
    SharedMemoryManager,
    copy_local_to_dram2,
    get_warp_coords,
)

# RDNA-specific constants
comptime RDNA_K_GROUP_SIZE = 1  # Always 1 for RDNA 16x16x16 WMMA


@always_inline
fn _mask_apply_rdna[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    mma_shape: IndexList[3],
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    frag_num_rows: Int,  # Physical fragment size (8 for RDNA)
    use_exp2: Bool = False,
](
    kv_tile_start_row: UInt32,
    kv_tile_num_rows: UInt32,
    start_pos: UInt32,
    seq_len: UInt32,
    num_keys: UInt32,
    mask_block_row: UInt32,
    mask_warp_row: UInt32,
    mask_warp_col: UInt32,
    scale: Float32,
    mask: mask_t,
    p_reg_vectorized: LayoutTensor[mut=True, accum_type, **_],
    not_last_iter: Bool,
    cache_start_pos: UInt32 = 0,
):
    """RDNA-specific mask application for WMMA fragments with swap_a_b=True.

    For RDNA WMMA 16x16 C/D register mapping:
    - Lane l, element v → D[row=v*2+l//16, col=l%16]
    - P^T[key, seq]: key = v*2 + l//16 (interleaved), seq = l%16
    - Lanes 0-15 hold even keys (0,2,4,...,14), lanes 16-31 hold odd keys (1,3,...,15)
    - All 8 elements in a lane are at the SAME seq position (l%16)
    """
    comptime output_frag_size = frag_num_rows  # 8 for RDNA

    var lane = lane_id()
    var scale_log2e: Scalar[accum_type] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )

    # RDNA WMMA C/D: lane l, elem v → D[row=v*2+l//16, col=l%16]
    # P^T[key, seq]: seq = l%16, key = v*2 + l//16 (interleaved)
    var lane_seq_offset = Int(lane % UInt(16))
    var lane_key_group = Int(lane // UInt(16))

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma
            p_reg_vectorized[mma_id, 0] = (
                p_reg_vectorized[mma_id, 0] * scale_log2e
            )

            # Base coordinates for this MMA tile in P matrix
            var mma_seq_base = mask_warp_row + UInt32(m_mma * mma_shape[0])
            var mma_key_base = (
                mask_warp_col
                + UInt32(n_mma * mma_shape[1])
                + (kv_tile_start_row if token_gen else 0)
            )

            var score_seq = (
                (num_keys - 1) if token_gen else mask_block_row
                + mma_seq_base
                + UInt32(lane_seq_offset)
            )
            var score_seq_with_start_pos = score_seq + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    # Interleaved key: elem j at lane group g → key = j*2 + g
                    var score_key = mma_key_base + UInt32(
                        j * 2 + lane_key_group
                    )
                    var score_key_with_cache_start_pos = (
                        score_key + cache_start_pos
                    )
                    # For RDNA, lane_key_group (lane//16) is the interleave
                    # group (even/odd keys), NOT the GQA group offset.
                    # Use lane % group to match MHAAttentionConfigRDNA.q_head_idx().
                    var group_idx = Int(lane % UInt(group))
                    var q_head_idx = (
                        block_idx.y * UInt(group) + UInt(group_idx)
                    ) if token_gen else block_idx.x
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_seq_with_start_pos),
                            Int(score_key_with_cache_start_pos),
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )

            @parameter
            if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            # Always mask out-of-bound keys on RDNA. Upstream, CausalMask sets
            # mask_out_of_bound=False on all AMD GPUs, but we do need this to
            # be masked for RDNA. To avoid a broader impact on CDNA
            # functionality, perform the masking here.
            var bound_seq = num_keys if token_gen else seq_len

            if score_seq >= bound_seq:

                @parameter
                for j in range(output_frag_size):
                    p_reg_vectorized[mma_id, 0][j] = Scalar[accum_type](-10000)
            elif not not_last_iter or token_gen:
                var bound_key = (
                    kv_tile_start_row
                    + kv_tile_num_rows if token_gen else num_keys
                )

                @parameter
                for j in range(output_frag_size):
                    var score_key = mma_key_base + UInt32(
                        j * 2 + lane_key_group
                    )
                    if score_key >= bound_key:
                        p_reg_vectorized[mma_id, 0][j] = Scalar[accum_type](
                            -10000
                        )


struct AttentionRDNA[
    attention_config_t: AttentionConfig,
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MHAConfig,
    group: Int,
    token_gen: Bool,
    sink: Bool,
    q_depth: Int = Int(config.depth),
    cache_depth: Int = Int(config.depth),
    output_depth: Int = Int(config.depth),
]:
    """RDNA-specific Attention implementation for Wave32 WMMA.

    This struct provides attention computation using RDNA's 16x16x16 WMMA
    operations with Wave32 execution. It uses RDNA-specific buffer types
    with 16-element A/B fragments and 8-element C/D fragments.
    """

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

    # RDNA always uses 16x16x16 WMMA
    comptime mma_shape = IndexList[3](RDNA_MMA_M, RDNA_MMA_N, RDNA_MMA_K)

    # RDNA fragment layouts
    comptime fragment_layout = get_rdna_fragment_layout()
    comptime output_frag_size = RDNA_CD_FRAG_SIZE
    comptime fragment_layout_nested = Layout.row_major(1, RDNA_CD_FRAG_SIZE)

    comptime num_m_mmas = ceildiv(Self.WM, UInt(Self.mma_shape[0]))
    comptime num_n_mmas = ceildiv(Self.WN, UInt(Self.mma_shape[1]))
    comptime num_n_mmas_output = ceildiv(
        Self.output_depth // Int(Self.num_warps_n), Self.mma_shape[1]
    )

    comptime swap_a_b = True
    comptime use_exp2 = True

    # RDNA k_group_size is always 1
    comptime k_group_size = RDNA_K_GROUP_SIZE
    comptime num_k_mmas2 = ceildiv(
        Self.BK, UInt(Self.mma_shape[2] * Self.k_group_size)
    )

    # RDNA warp layout: 16 rows x 2 columns = 32 threads
    comptime warp_layout = get_rdna_warp_layout()

    comptime num_stages = 2

    # RDNA-specific buffer types
    comptime OutputRegisterBufferType = OutputRegisterBufferRDNA[
        Self.accum_type,
        Int(Self.num_m_mmas),
        Self.num_n_mmas_output,
    ]

    comptime PRegisterBufferType = PRegisterBufferRDNA[
        Self.accum_type,
        Self.q_type,
        Int(Self.BM),
        Int(Self.BN),
        Int(Self.BK),
        Int(Self.WM),
        Int(Self.WN),
        Int(Self.num_m_mmas),
        Int(Self.num_n_mmas),
        Self.mma_shape,
        Self.k_group_size,
    ]

    comptime row_layout = Layout.row_major(
        Int(Self.num_m_mmas), Self.fragment_layout.shape[0].value()
    )

    comptime RowMaxTensorType = LocalLayoutTensor[
        Self.accum_type,
        Self.row_layout,
    ]

    comptime RowSumTensorType = Self.RowMaxTensorType

    comptime GlobalMemoryManagerType = GlobalMemoryManager[
        Self.q_type,
        UInt32(Self.BM),
        UInt32(Self.BN),
        UInt32(Self.BK),
        UInt32(Self.depth),
        UInt32(Self.num_heads),
        UInt32(Self.group),
        Self.token_gen,
        UInt32(Self.q_depth),
        UInt32(Self.output_depth),
    ]

    comptime SharedMemoryManagerType = SharedMemoryManager[
        Self.attention_config_t.shared_kv,
        Self.attention_config_t.full_kv,
        Self.attention_config_t.depth_padded,
        Self.attention_config_t.double_buffer,
        Self.q_type,
        Int(Self.BM),
        Int(Self.BN),
        Int(Self.BK),
        Int(Self.depth),
        Self.token_gen,
    ]

    comptime QRegisterBufferType = QRegisterBufferRDNA[
        dtype = Self.q_type,
        mma_shape = Self.mma_shape,
        k_group_size = Self.k_group_size,
        WM = Int(Self.WM),
        WN = Int(Self.WN),
        BN = Int(Self.BN),
        BK = Int(Self.BK),
        depth = Self.q_depth,
        thread_layout = Self.warp_layout,
    ]

    var out_reg_buffer: Self.OutputRegisterBufferType
    var p_reg_buffer: Self.PRegisterBufferType

    var rowmax: Self.RowMaxTensorType
    var rowsum: Self.RowSumTensorType

    var gmem_manager: Self.GlobalMemoryManagerType
    var smem_manager: Self.SharedMemoryManagerType

    var q_buffer: Self.QRegisterBufferType
    var output_ptr: UnsafePointer[Scalar[Self.output_type], MutAnyOrigin]

    var batch_idx: Int

    var k: Self.k_t
    var v: Self.v_t
    var mask: Self.mask_t

    var mask_block_row: UInt32
    var mask_warp_row: UInt32
    var mask_warp_col: UInt32

    var scale: Float32

    var seq_len: Int
    var num_keys: Int
    var start_pos: Int
    var cache_start_pos: Int

    var warp_scratch_tensor: SharedLayoutTensor[
        Self.accum_type,
        Layout.row_major(2 * Int(Self.num_warps_n), Int(Self.BM)),
    ]

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        return Self.attention_config_t.q_head_idx()

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return Self.attention_config_t.q_tile_idx()

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return Self.attention_config_t.kv_head_idx()

    @always_inline
    fn zero_p_buffer(self):
        self.p_reg_buffer.zero()

    @always_inline
    fn get_batch_idx(self) -> Int:
        return self.batch_idx

    @staticmethod
    @always_inline
    fn get_tensor_core_mma_qk(
        out result: TiledTensorCore[
            get_accum_type[Self.q_type](),
            Self.q_type,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=True,
        ],
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn get_tensor_core_mma_pv(
        out result: TiledTensorCore[
            get_accum_type[Self.q_type](),
            Self.q_type,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=False,
        ],
    ):
        return type_of(result)()

    @always_inline
    fn mma_qk[
        k_buffer_type: KVBuffer,
        //,
        prefetch_function: OptionalReg[fn() capturing -> None] = None,
        beg_iter: Int = 0,
        num_iters: Int = Int(Self.depth // Self.BK),
        prefetched_b_tile: Bool = False,
    ](mut self, mut k_buffer: k_buffer_type):
        mma_rdna[
            tensor_core_mma = Self.get_tensor_core_mma_qk(),
            BK = Int(Self.BK),
            prefetch_function=prefetch_function,
            swap_a_b = Self.swap_a_b,
            beg_iter=beg_iter,
            num_iters=num_iters,
            prefetched_b_tile=prefetched_b_tile,
        ](
            self.p_reg_buffer,
            self.q_buffer,
            k_buffer,
        )

    @always_inline
    fn mma_pv[
        v_buffer_type: KVBuffer,
        //,
        prefetch_function: OptionalReg[fn() capturing -> None] = None,
        prefetched_b_tile: Bool = True,
    ](mut self, mut v_buffer: v_buffer_type):
        # Create a callback that copies P chunk i to shared memory
        @parameter
        fn copy_p_chunk[i: Int]():
            self.p_reg_buffer.copy_to_shared[i]()

        mma_rdna[
            tensor_core_mma = Self.get_tensor_core_mma_pv(),
            BK = Int(Self.BK),
            prefetch_function=prefetch_function,
            swap_a_b = Self.swap_a_b,
            num_iters = Int(Self.BN // Self.BK),
            prefetched_b_tile=prefetched_b_tile,
            a_copy_fn=copy_p_chunk,
        ](
            self.out_reg_buffer,
            self.p_reg_buffer,
            v_buffer,
        )

    @always_inline
    fn mask_status(
        self,
        kv_tile_start_row: UInt32,
    ) -> TileMaskStatus:
        @parameter
        if Self.token_gen:
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(Self.BN)),
            )
        else:
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.mask_block_row + UInt32(self.start_pos)),
                    Int(kv_tile_start_row + UInt32(self.cache_start_pos)),
                ),
                Index[dtype = DType.uint32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn mask_advance(mut self):
        @parameter
        if not Self.token_gen:
            self.mask_warp_col += UInt32(Self.BN)

    @always_inline
    fn mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    fn mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        @parameter
        if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var status = self.mask_status(
                kv_tile_start_row,
            )
            if self.mask_skip_tile(status):
                self.mask_advance()
                return True
        return False

    @always_inline
    fn mask_apply(
        mut self,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        not_last_iter: Bool,
    ):
        @always_inline
        @parameter
        fn _mask_apply_impl[masked: Bool]():
            _mask_apply_rdna[
                masked=masked,
                accum_type = Self.accum_type,
                token_gen = Self.token_gen,
                mma_shape = Self.mma_shape,
                num_m_mmas = Int(Self.num_m_mmas),
                num_n_mmas = Int(Self.num_n_mmas),
                mask_t = Self.mask_t,
                group = Self.group,
                frag_num_rows=RDNA_CD_FRAG_SIZE,
                use_exp2 = Self.use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                UInt32(self.start_pos),
                UInt32(self.seq_len),
                UInt32(self.num_keys),
                self.mask_block_row,
                self.mask_warp_row,
                self.mask_warp_col,
                self.scale,
                self.mask,
                self.p_reg_buffer.vectorize(),
                not_last_iter,
                UInt32(self.cache_start_pos),
            )

        @parameter
        if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var mask_status = self.mask_status(
                kv_tile_start_row,
            )
            unswitch[_mask_apply_impl](
                mask_status == TileMaskStatus.PARTIAL_MASK
            )
        else:
            _mask_apply_impl[masked=True]()
        self.mask_advance()

    @always_inline
    fn __init__(
        out self,
        attention_config: Self.attention_config_t,
        output_ptr: UnsafePointer[Scalar[Self.output_type], MutAnyOrigin],
        q: UnsafePointer[Scalar[Self.q_type], ImmutAnyOrigin],
        k: Self.k_t,
        v: Self.v_t,
        mask: Self.mask_t,
        sink_weights: OptionalReg[
            LayoutTensor[
                Self.q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ],
        batch_idx: Int,
        scale: Float32,
        seq_len: Int,
        num_keys: Int,
        start_pos: Int,
        cache_start_pos: Int = 0,
    ):
        self.rowmax = Self.RowMaxTensorType.stack_allocation()
        self.rowsum = Self.RowSumTensorType.stack_allocation()
        self.out_reg_buffer = Self.OutputRegisterBufferType()
        self.out_reg_buffer.zero()

        self.gmem_manager = Self.GlobalMemoryManagerType(
            UInt32(Self.q_tile_idx()),
            UInt32(Self.kv_head_idx()),
            seq_len,
            Self.attention_config_t.get_q_offset[UInt(Self.q_depth)](),
            Self.attention_config_t.get_output_offset[
                UInt(Self.output_depth)
            ](),
        )
        self.smem_manager = Self.SharedMemoryManagerType()

        self.warp_scratch_tensor = type_of(self.warp_scratch_tensor)(
            self.smem_manager.get_warp_scratch_ptr[Self.accum_type]()
        )

        @parameter
        if not Self.token_gen:
            # In prefill mode, P needs BM*BK elements of shared memory.
            # K only has BN*BK elements, and BM can exceed BN (e.g. depth=64:
            # BM=128, BN=64), so we allocate a dedicated P buffer.
            var p_ptr = stack_allocation[
                Int(Self.BM) * Int(Self.BK),
                Self.q_type,
                address_space = AddressSpace.SHARED,
            ]()
            self.p_reg_buffer = Self.PRegisterBufferType(p_ptr)
        else:
            var p_ptr = self.smem_manager.get_p_ptr[Self.q_type]()
            self.p_reg_buffer = Self.PRegisterBufferType(p_ptr)

        var q_tile = self.gmem_manager.get_q_tensor(q)
        self.q_buffer = Self.QRegisterBufferType(q_tile)

        self.output_ptr = output_ptr

        self.k = k
        self.v = v
        self.mask = mask

        self.mask_block_row = UInt32(self.q_tile_idx() * Self.BM)
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]
        var warp_col = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[1]
        self.mask_warp_row = UInt32(warp_row * Int(Self.WM))
        self.mask_warp_col = UInt32(warp_col * Int(Self.WN))

        self.batch_idx = batch_idx
        self.scale = scale

        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos

        @parameter
        if Self.sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            var sink_weight = (
                sink_weights.value()[Int(self.q_head_idx())][0].cast[
                    Self.accum_type
                ]()
                * log2e
            )
            self.rowmax = self.rowmax.fill(sink_weight)
            self.rowsum = self.rowsum.fill(1)
        else:
            self.rowmax = self.rowmax.fill(min_or_neg_inf[Self.accum_type]())
            self.rowsum = self.rowsum.fill(0)

    @always_inline
    fn online_softmax(self):
        var warp_scratch = self.warp_scratch_tensor
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]

        _online_softmax_iter_for_mma_output[
            Self.accum_type,
            Layout.row_major(Int(Self.num_m_mmas), Int(Self.num_n_mmas)),
            Layout.row_major(Int(Self.num_warps_m), Int(Self.num_warps_n)),
            Self.warp_layout,
            use_exp2 = Self.use_exp2,
            fragment_layout = Self.fragment_layout,
        ](
            self.out_reg_buffer.vectorize(),
            self.p_reg_buffer.vectorize(),
            warp_scratch.tile[2 * Int(Self.num_warps_n), Int(Self.WM)](
                0, Int(warp_row)
            ),
            self.rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            self.rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

    @always_inline
    fn store_output(self):
        """Store output from registers to global memory."""
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]
        var warp_col = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[1]
        var output_tile = self.gmem_manager.get_output_tensor(self.output_ptr)

        var reg_tile = self.out_reg_buffer.get_reg_tile()
        var lane = Int(lane_id())
        var row_group = lane // 16
        var col_within_mma = lane % 16

        # For decoding, M-dim rows are heads in the GQA group (stride=output_depth).
        # For prefill, M-dim rows are seq positions (stride=num_heads*output_depth).
        comptime row_stride = Int(Self.output_depth) if Self.token_gen else Int(
            Self.num_heads
        ) * Int(Self.output_depth)
        var row_bound = Int(Self.group) if Self.token_gen else self.seq_len

        @parameter
        for depth_tile in range(Self.num_n_mmas_output):

            @parameter
            for seq_tile in range(Self.num_m_mmas):
                comptime mma_idx = Int(depth_tile) * Int(Self.num_m_mmas) + Int(
                    seq_tile
                )

                @parameter
                for elem in range(8):
                    # RDNA WMMA C/D: lane l, elem v → D[row=v*2+l//16, col=l%16]
                    # O^T[depth, seq]: depth = elem*2 + row_group (interleaved)
                    var seq_in_mma = col_within_mma
                    var depth_in_mma = elem * 2 + row_group

                    var global_seq = (
                        Int(warp_row) * Int(Self.WM)
                        + Int(seq_tile) * 16
                        + seq_in_mma
                    )
                    var global_depth = (
                        Int(warp_col)
                        * Int(Self.output_depth)
                        // Int(Self.num_warps_n)
                        + Int(depth_tile) * 16
                        + depth_in_mma
                    )

                    if global_seq < row_bound:
                        var output_offset = (
                            global_seq * row_stride + global_depth
                        )
                        comptime reg_offset = mma_idx * 8 + elem
                        var val = reg_tile.ptr[reg_offset]

                        output_tile.ptr[output_offset] = val.cast[
                            output_tile.dtype
                        ]()

    @always_inline
    fn copy_fragment_to_smem[chunk_idx: Int](self):
        """Copy one chunk of P to shared memory."""
        self.p_reg_buffer.copy_to_shared[chunk_idx]()

    @always_inline
    fn store_partition_info(
        self,
        num_partitions: Int,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
    ):
        @parameter
        if not Self.token_gen:
            return

        var q_head_idx = self.q_head_idx()
        if num_partitions > 1:
            if thread_idx.x < UInt(Self.group):
                var row_sum = self.rowsum[0, 0][0]
                var row_max = self.rowmax[0, 0][0]

                exp_sum_ptr[q_head_idx] = row_sum
                qk_max_ptr[q_head_idx] = row_max
