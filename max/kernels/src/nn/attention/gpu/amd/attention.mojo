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

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e

from sys import size_of, simd_width_of
from sys.info import _cdna_4_or_newer
from sys.intrinsics import _type_is_eq
from sys._assembly import inlined_assembly
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
from memory import bitcast
from sys.intrinsics import readfirstlane
from nn.mha_mask import CausalMask, MASK_VALUE, MaterializedMask
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore, num_matrix_reg
from memory.pointer import AddressSpace as BaseAddressSpace
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from .softmax import Softmax
from sys import _RegisterPackType
from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf

from .buffers import (
    KBuffer,
    KVBuffer,
    OutputRegisterBuffer,
    PRegisterBuffer,
    QRegisterBuffer,
    VBuffer,
    VBufferTransposeLoads,
)
from .mma import mma
from .utils import (
    GlobalMemoryManager,
    LocalLayoutTensor,
    SharedLayoutTensor,
    SharedMemoryManager,
    copy_local_to_dram2,
    get_fragment_layout,
    get_nested_fragment_layout,
    get_warp_coords,
    get_warp_layout,
)


trait AttentionConfig(ImplicitlyCopyable):
    # share shared memory for k and v
    comptime shared_kv: Bool
    # shared memory for the full tile vs BK blocks
    comptime full_kv: Bool
    # pad the depth for v smem
    comptime depth_padded: Bool
    # double buffer
    comptime double_buffer: Bool

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        ...

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        ...

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        ...

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        ...


@always_inline
fn _mask_apply[
    attention_config_t: AttentionConfig,
    accum_type: DType,
    token_gen: Bool,
    mma_shape: IndexList[3],
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
    use_exp2: Bool = False,
](
    masked: Bool,
    kv_tile_start_row: UInt32,
    kv_tile_num_rows: UInt32,
    start_pos: UInt32,
    seq_len: UInt32,
    num_keys: UInt32,
    mask_block_row: UInt32,
    mask_warp_row: UInt32,
    mask_warp_col: UInt32,
    scale: Scalar[accum_type],
    mask: mask_t,
    p_reg_vectorized: LayoutTensor[mut=True, accum_type, ...],
    not_last_iter: Bool,
    cache_start_pos: UInt32 = 0,
):
    comptime output_frag_size = fragment_layout.size()

    comptime rowwise_stride = fragment_layout.shape[0].value()
    comptime colwise_stride = fragment_layout.shape[1].value()
    comptime frag_is_row_vector = rowwise_stride == 1
    comptime assert frag_is_row_vector, "fragment layout is not a row vector"

    var lane = lane_id()

    var coords = idx2crd[warp_layout](Int(lane))
    var lane_row = coords[0] * rowwise_stride
    var lane_col = coords[1] * colwise_stride

    comptime if token_gen:
        if lane_row >= group:
            return

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            # Coordinates in mask for current mma tile.
            var mask_frag_row = mask_warp_row + UInt32(m_mma * mma_shape[0])
            var mask_frag_col = (
                mask_warp_col
                + UInt32(n_mma * mma_shape[1])
                + (kv_tile_start_row if token_gen else 0)
            )
            mask_frag_row += UInt32(lane_row)
            mask_frag_col += UInt32(lane_col)
            # The row in score matrix of shape seq_len x num_keys.
            # Mask col is score col since we don't partition in col.
            var score_row = (
                num_keys - 1
            ) if token_gen else mask_block_row + mask_frag_row
            var score_col = mask_frag_col
            var score_col_with_cache_start_pos = score_col + cache_start_pos
            var score_row_with_start_pos = score_row + start_pos
            comptime is_causal_mask = _type_is_eq[mask_t, CausalMask]()

            if masked:
                comptime for j in range(
                    0, output_frag_size, 2 if is_causal_mask else 1
                ):
                    var q_head_idx = attention_config_t.q_head_idx()

                    comptime if is_causal_mask:
                        var x_0 = Int32(
                            score_row_with_start_pos
                            - score_col_with_cache_start_pos
                        )
                        comptime y_0 = Int32(fragment_layout(j)) - 1
                        var val_0 = p_reg_vectorized[mma_id, 0][j]

                        comptime y_1 = Int32(fragment_layout(j + 1)) - 1
                        var val_1 = p_reg_vectorized[mma_id, 0][j + 1]
                        var val_inf = Scalar[accum_type](-10000.0)

                        # v_cmp writing to 2 sgprs or vcc requires waiting for 1 cycle
                        # before they can be used in another VALU instruction.
                        # Using 2 cmp and 2 cndmask instructions avoids this wait.
                        # Without using this inline asm, we would stall for a cycle after each comparison.
                        # the generated asm without this would be:
                        # v_cmp_lt_i32_e32 vcc, -1, v221
                        # s_nop 1
                        # v_cndmask_b32_e32 v82, v227, v82, vcc
                        # v_cmp_lt_i32_e32 vcc, 0, v221
                        # s_nop 1
                        # v_cndmask_b32_e32 v83, v227, v83, vcc

                        # this inline asm does the same thing as CausalMask.mask(),
                        # it just processes 2 elements at a time vs. 1.
                        comptime asm = """
                            v_cmp_lt_i32_e64 $2, $5, $4
                            v_cmp_lt_i32_e64 $3, $8, $7
                            v_cndmask_b32_e64 $0, $10, $6, $2
                            v_cndmask_b32_e64 $1, $10, $9, $3
                            """
                        var ret = inlined_assembly[
                            asm,
                            _RegisterPackType[
                                Scalar[accum_type],
                                Scalar[accum_type],
                                Int64,
                                Int64,
                            ],
                            constraints="=v,=v,=&s,=&s,v,n,v,v,n,v,v,~{vcc}",
                        ](x_0, y_0, val_0, x_0, y_1, val_1, val_inf)
                        p_reg_vectorized[mma_id, 0][j] = ret[0]
                        p_reg_vectorized[mma_id, 0][j + 1] = ret[1]

                        # p_reg_vectorized[mma_id, 0][j] = mask.mask[
                        #     element_type = DType.int32
                        # ](
                        #     IndexList[4, element_type = DType.int32](
                        #         Int(block_idx.z),
                        #         Int(q_head_idx),
                        #         Int(score_row_with_start_pos)
                        #         - Int(score_col_with_cache_start_pos),
                        #         Int(fragment_col),
                        #     ),
                        #     p_reg_vectorized[mma_id, 0][j],
                        # )
                    else:
                        comptime fragment_col = fragment_layout(j)
                        p_reg_vectorized[mma_id, 0][j] = mask.mask(
                            IndexList[4, element_type = DType.uint32](
                                Int(block_idx.z),
                                Int(q_head_idx),
                                Int(score_row_with_start_pos),
                                Int(
                                    score_col_with_cache_start_pos
                                    + UInt32(fragment_col)
                                ),
                            ),
                            p_reg_vectorized[mma_id, 0][j],
                        )

            comptime if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            if (not not_last_iter or token_gen) and mask_t.mask_out_of_bound:
                var bound_y = (
                    kv_tile_start_row
                    + kv_tile_num_rows if token_gen else num_keys
                )

                comptime for j in range(output_frag_size):
                    comptime fragment_col = fragment_layout(j)

                    var bound_x = num_keys if token_gen else seq_len

                    p_reg_vectorized[mma_id, 0][j] = _kernel_mask(
                        IndexList[2, element_type = DType.uint32](
                            Int(score_row),
                            Int(score_col + UInt32(fragment_col)),
                        ),
                        IndexList[2, element_type = DType.uint32](
                            Int(bound_x), Int(bound_y)
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )


struct Attention[
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

    comptime mma_shape = Self.attention_config_t.get_mma_shape()

    comptime fragment_layout = get_fragment_layout[Self.mma_shape]()
    comptime output_frag_size = Self.fragment_layout.size()
    comptime fragment_layout_nested = get_nested_fragment_layout[
        Self.mma_shape
    ]()

    comptime num_m_mmas = ceildiv(Self.WM, UInt(Self.mma_shape[0]))
    comptime num_n_mmas = ceildiv(Self.WN, UInt(Self.mma_shape[1]))
    comptime num_n_mmas_output = ceildiv(
        Self.output_depth // Int(Self.num_warps_n), Self.mma_shape[1]
    )

    comptime swap_a_b = True
    comptime use_exp2 = True
    # we want to load 16B of data for each fragment so k_group_size is set such that
    # k_group_size * num_matrix_fragments * size_of[Self.q_type]() = 16B
    comptime k_group_size = 16 // (
        num_matrix_reg[Self.mma_shape[0], Self.mma_shape[2]]()
        * size_of[Self.q_type]()
    )
    comptime num_k_mmas2 = ceildiv(
        Self.BK, UInt(Self.mma_shape[2] * Self.k_group_size)
    )

    comptime warp_layout = get_warp_layout[Self.mma_shape]()

    comptime num_stages = 2

    comptime OutputRegisterBufferType = OutputRegisterBuffer[
        Self.accum_type,
        Int(Self.num_m_mmas),
        Self.num_n_mmas_output,
        Self.output_frag_size,
    ]

    comptime PRegisterBufferType = PRegisterBuffer[
        Self.accum_type,
        Self.q_type,
        Int(Self.BM),
        Int(Self.BN),
        Int(Self.BK),
        Int(Self.WM),
        Int(Self.WN),
        Int(Self.num_m_mmas),
        Int(Self.num_n_mmas),
        Self.output_frag_size,
        Self.BN != Self.WN,
        Self.mma_shape,
        Self.k_group_size,
        # use double buffer as proxy for experimental kernel
        # need to find a better way to determine this
        tr_load_enabled = Self.attention_config_t.double_buffer,
        num_stages = 2 if Self.attention_config_t.double_buffer else 1,
    ]

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

    comptime QRegisterBufferType = QRegisterBuffer[
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
    var kv_start_row: UInt32

    var scale: Scalar[Self.accum_type]

    var seq_len: Int
    var num_keys: Int
    var start_pos: Int
    var cache_start_pos: Int

    var softmax: Softmax[
        Self.accum_type,
        Layout.row_major(Int(Self.num_m_mmas), Int(Self.num_n_mmas)),
        Layout.row_major(Int(Self.num_warps_m), Int(Self.num_warps_n)),
        Self.warp_layout,
        Self.fragment_layout,
        Self.use_exp2,
    ]

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
    fn zero_p_buffer[stage: Int = 0](self):
        self.p_reg_buffer.zero[stage]()

    @always_inline
    fn get_batch_idx(self) -> Int:
        return self.batch_idx

    @always_inline
    fn scale_p_reg[stage: Int = 0](self):
        var p_reg_vectorized = self.p_reg_buffer.vectorize[stage]()

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                p_reg_vectorized[mma_id, 0] *= self.scale

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
        mma[
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
        mma[
            tensor_core_mma = Self.get_tensor_core_mma_pv(),
            BK = Int(Self.BK),
            prefetch_function=prefetch_function,
            swap_a_b = Self.swap_a_b,
            num_iters = Int(Self.BN // Self.BK),
            prefetched_b_tile=prefetched_b_tile,
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
        comptime if Self.token_gen:
            # Decoding with mask checking: check single token at num_keys-1
            return self.mask.status(
                Index[dtype = DType.uint32](
                    self.num_keys - 1,
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(Self.BN)),
            )
        else:
            # Prefill or decoding without mask checking: check full tile
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.mask_block_row + UInt32(self.start_pos)),
                    Int(kv_tile_start_row + UInt32(self.cache_start_pos)),
                ),
                Index[dtype = DType.uint32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn mask_advance(mut self):
        comptime if not Self.token_gen:
            self.mask_warp_col += UInt32(Self.BN)

    @always_inline
    fn mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    fn mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var status = self.mask_status(
                kv_tile_start_row,
            )
            if self.mask_skip_tile(status):
                self.mask_advance()
                return True
        return False

    @always_inline
    fn mask_apply[
        stage: Int = 0
    ](
        mut self,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        not_last_iter: Bool,
    ):
        @always_inline
        @parameter
        fn _mask_apply_impl(masked: Bool):
            _mask_apply[
                attention_config_t = Self.attention_config_t,
                accum_type = Self.accum_type,
                token_gen = Self.token_gen,
                mma_shape = Self.mma_shape,
                num_m_mmas = Int(Self.num_m_mmas),
                num_n_mmas = Int(Self.num_n_mmas),
                mask_t = Self.mask_t,
                group = Self.group,
                fragment_layout = Self.fragment_layout_nested,
                warp_layout = Self.warp_layout,
                use_exp2 = Self.use_exp2,
            ](
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
                self.p_reg_buffer.vectorize[stage](),
                not_last_iter,
                UInt32(self.cache_start_pos),
            )

        # self.scale_p_reg[stage]()

        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var mask_status = self.mask_status(
                kv_tile_start_row,
            )
            _mask_apply_impl(mask_status == TileMaskStatus.PARTIAL_MASK)
        else:
            _mask_apply_impl(masked=True)
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
        self.softmax = type_of(self.softmax)()
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

        self.p_reg_buffer = Self.PRegisterBufferType(
            self.smem_manager.get_p_ptr[Self.q_type]()
        )

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

        comptime scaling_factor = (
            log2e if (
                Self.use_exp2 and (not Self.mask_t.apply_log2e_after_mask)
            ) else Scalar[Self.accum_type](1)
        )
        var scale_log2e: Scalar[Self.accum_type] = (
            scale.cast[Self.accum_type]() * scaling_factor
        )

        comptime is_causal_mask = _type_is_eq[Self.mask_t, CausalMask]()

        comptime if not is_causal_mask or (
            not Self.attention_config_t.double_buffer
        ):
            # using double buffer as proxy for the experimental kernel,
            # using readfirstlane for the gfx942 kernel leads to incorrect results.
            # This needs more investigation.
            # Disable inline asm for all other masks except causal mask,
            # Inline asm was generating bad code for the MaterializedMask,
            # and for now we only care about the performance of the causal mask.
            self.scale = scale_log2e
        else:
            self.scale = inlined_assembly[
                "v_readfirstlane_b32 $0, $1",
                Scalar[Self.accum_type],
                constraints="=s,v",
            ](scale_log2e)
            # readfirstlane does not work without inline asm
            # bitcast[Self.accum_type](
            #     readfirstlane(bitcast[DType.int32](Float32(scale_log2e)))
            # )

        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos
        self.kv_start_row = 0

        comptime if Self.sink:
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
            _ = self.softmax.rowmax_tensor.fill(sink_weight)
            _ = self.softmax.rowsum_tensor.fill(1)
        else:
            _ = self.softmax.rowmax_tensor.fill(
                min_or_neg_inf[Self.accum_type]()
            )
            _ = self.softmax.rowsum_tensor.fill(0)

    @always_inline
    fn online_softmax[stage: Int = 0](mut self):
        var warp_scratch = self.warp_scratch_tensor
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]

        self.softmax.full(
            self.out_reg_buffer.vectorize(),
            self.p_reg_buffer.vectorize[stage](),
            warp_scratch.tile[2 * Int(Self.num_warps_n), Int(Self.WM)](
                0, warp_row
            ),
        )

    @always_inline
    fn store_output(self):
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]
        var warp_col = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[1]
        var output_tile = self.gmem_manager.get_output_tensor(self.output_ptr)
        var output_warp_tile = output_tile.tile[
            Int(Self.WM), Self.output_depth // Int(Self.num_warps_n)
        ](warp_row, warp_col)

        comptime if Self.mma_shape[0] == 32:
            copy_local_to_dram2[
                dst_thread_layout = Self.warp_layout,
                thread_scope = ThreadScope.WARP,
            ](
                output_warp_tile.vectorize[
                    1,
                    4,
                ](),
                self.out_reg_buffer.reg_tile.vectorize[1, 4](),
                output_tile,
            )
        else:
            copy_local_to_dram[
                dst_thread_layout = Self.warp_layout,
                thread_scope = ThreadScope.WARP,
            ](
                output_warp_tile.vectorize[
                    Self.fragment_layout.shape[0].value(),
                    Self.fragment_layout.shape[1].value(),
                ](),
                self.out_reg_buffer.vectorize(),
                output_tile,
            )

    @always_inline
    fn copy_fragment_to_smem(self):
        comptime if not Self.token_gen:
            return

        self.p_reg_buffer.copy_to_shared()

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
        comptime if not Self.token_gen:
            return

        var q_head_idx = self.q_head_idx()
        if num_partitions > 1:
            if thread_idx.x < UInt(Self.group):
                var row_sum = self.softmax.rowsum_tensor[0, 0][0]
                var row_max = self.softmax.rowmax_tensor[0, 0][0]

                exp_sum_ptr[q_head_idx] = row_sum
                qk_max_ptr[q_head_idx] = row_max
