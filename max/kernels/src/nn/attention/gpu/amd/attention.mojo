# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from layout.tensor_core import TiledTensorCore, num_matrix_reg
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
    convert_f32_to_bf16,
    copy_local_to_dram2,
    get_fragment_layout,
    get_nested_fragment_layout,
    get_warp_coords,
    get_warp_layout,
)


trait AttentionConfig(ImplicitlyCopyable):
    # share shared memory for k and v
    alias shared_kv: Bool
    # shared memory for the full tile vs BK blocks
    alias full_kv: Bool
    # pad the depth for v smem
    alias depth_padded: Bool
    # double buffer
    alias double_buffer: Bool

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
    masked: Bool,
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
    p_reg_vectorized: LayoutTensor[accum_type, **_],
    not_last_iter: Bool,
    cache_start_pos: UInt32 = 0,
):
    alias output_frag_size = fragment_layout.size()

    alias rowwise_stride = fragment_layout.shape[0].value()
    alias colwise_stride = fragment_layout.shape[1].value()
    alias frag_is_row_vector = rowwise_stride == 1
    constrained[
        frag_is_row_vector,
        "fragment layout is not a row vector",
    ]()

    var lane = lane_id()
    var scale_log2e: Scalar[accum_type] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )

    var coords = idx2crd[warp_layout](Int(lane))
    var lane_row = coords[0] * rowwise_stride
    var lane_col = coords[1] * colwise_stride

    @parameter
    if token_gen:
        if lane_row >= group:
            return

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma
            p_reg_vectorized[mma_id, 0] = (
                p_reg_vectorized[mma_id, 0] * scale_log2e
            )
            # Coordinates in mask for current mma tile.
            var mask_frag_row = mask_warp_row + m_mma * mma_shape[0]
            var mask_frag_col = (
                mask_warp_col
                + n_mma * mma_shape[1]
                + (kv_tile_start_row if token_gen else 0)
            )
            mask_frag_row += lane_row
            mask_frag_col += lane_col
            # The row in score matrix of shape seq_len x num_keys.
            # Mask col is score col since we don't partition in col.
            var score_row = (
                num_keys - 1
            ) if token_gen else mask_block_row + mask_frag_row
            var score_col = mask_frag_col
            var score_col_with_cache_start_pos = score_col + cache_start_pos
            var score_row_with_start_pos = score_row + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)
                    var group_idx = lane_row
                    var q_head_idx = (
                        block_idx.y * UInt(group) + UInt(group_idx)
                    ) if token_gen else block_idx.y
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos),
                            Int(score_col_with_cache_start_pos + fragment_col),
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )

            @parameter
            if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            if (not not_last_iter or token_gen) and mask_t.mask_out_of_bound:
                var bound_y = (
                    kv_tile_start_row
                    + kv_tile_num_rows if token_gen else num_keys
                )

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)

                    var bound_x = num_keys if token_gen else seq_len

                    p_reg_vectorized[mma_id, 0][j] = _kernel_mask(
                        IndexList[2, element_type = DType.uint32](
                            Int(score_row),
                            Int(score_col + fragment_col),
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
    mask_t: MHAMask, //,
    config: MHAConfig,
    group: Int,
    token_gen: Bool,
    sink: Bool,
    q_depth: Int = Int(config.depth),
    cache_depth: Int = Int(config.depth),
    output_depth: Int = Int(config.depth),
]:
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_threads = config.num_threads()
    alias num_heads = config.num_heads
    alias num_warps_n = Self.BN // Self.WN
    alias num_warps_m = Self.BM // Self.WM
    alias depth = config.depth
    alias accum_type = get_accum_type[q_type]()

    alias mma_shape = Self.attention_config_t.get_mma_shape()

    alias fragment_layout = get_fragment_layout[Self.mma_shape]()
    alias output_frag_size = Self.fragment_layout.size()
    alias fragment_layout_nested = get_nested_fragment_layout[Self.mma_shape]()

    alias num_m_mmas = ceildiv(Self.WM, UInt(Self.mma_shape[0]))
    alias num_n_mmas = ceildiv(Self.WN, UInt(Self.mma_shape[1]))
    alias num_n_mmas_output = ceildiv(
        Self.output_depth // Int(Self.num_warps_n), Self.mma_shape[1]
    )

    alias swap_a_b = True
    alias use_exp2 = True
    # we want to load 16B of data for each fragment so k_group_size is set such that
    # k_group_size * num_matrix_fragments * size_of[Self.q_type]() = 16B
    alias k_group_size = 16 // (
        num_matrix_reg[Self.mma_shape[0], Self.mma_shape[2]]()
        * size_of[Self.q_type]()
    )
    alias num_k_mmas2 = ceildiv(
        Self.BK, UInt(Self.mma_shape[2] * Self.k_group_size)
    )

    alias warp_layout = get_warp_layout[Self.mma_shape]()

    alias num_stages = 2

    alias OutputRegisterBufferType = OutputRegisterBuffer[
        Self.accum_type,
        Int(Self.num_m_mmas),
        Self.num_n_mmas_output,
        Self.output_frag_size,
    ]

    alias PRegisterBufferType = PRegisterBuffer[
        Self.accum_type,
        q_type,
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
        tr_load_enabled = attention_config_t.double_buffer,
    ]

    alias row_layout = Layout.row_major(
        Int(Self.num_m_mmas), Self.fragment_layout.shape[0].value()
    )

    alias RowMaxTensorType = LocalLayoutTensor[
        Self.accum_type,
        Self.row_layout,
    ]

    alias RowSumTensorType = Self.RowMaxTensorType

    alias GlobalMemoryManagerType = GlobalMemoryManager[
        q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.depth,
        Self.num_heads,
        Self.group,
        Self.token_gen,
        Self.q_depth,
        Self.output_depth,
    ]

    alias SharedMemoryManagerType = SharedMemoryManager[
        attention_config_t.shared_kv,
        attention_config_t.full_kv,
        attention_config_t.depth_padded,
        attention_config_t.double_buffer,
        Self.q_type,
        Int(Self.BM),
        Int(Self.BN),
        Int(Self.BK),
        Int(Self.depth),
        Self.token_gen,
    ]

    alias QRegisterBufferType = QRegisterBuffer[
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
    var output_ptr: UnsafePointer[Scalar[Self.output_type],]

    var batch_idx: Int

    var k: k_t
    var v: v_t
    var mask: mask_t

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
            get_accum_type[q_type](),
            q_type,
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
            get_accum_type[q_type](),
            q_type,
            Self.mma_shape,
            group_size = Self.k_group_size,
            transpose_b=False,
        ],
    ):
        return type_of(result)()

    @always_inline
    fn mma_qk[
        k_buffer_type: KVBuffer, //,
        prefetch_function: OptionalReg[fn () capturing -> None] = None,
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
        v_buffer_type: KVBuffer, //,
        prefetch_function: OptionalReg[fn () capturing -> None] = None,
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
        @parameter
        if token_gen:
            # Decoding with mask checking: check single token at num_keys-1
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(Self.BN)),
            )
        else:
            # Prefill or decoding without mask checking: check full tile
            return self.mask.status(
                Index[dtype = DType.uint32](
                    Int(self.mask_block_row + self.start_pos),
                    Int(kv_tile_start_row + self.cache_start_pos),
                ),
                Index[dtype = DType.uint32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn mask_advance(mut self):
        @parameter
        if not token_gen:
            self.mask_warp_col += Self.BN

    @always_inline
    fn mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    fn mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        @parameter
        if not token_gen or mask_t.check_mask_during_decoding:
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
            _mask_apply[
                masked=masked,
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
                kv_tile_start_row,
                kv_tile_num_rows,
                self.start_pos,
                self.seq_len,
                self.num_keys,
                Int(self.mask_block_row),
                Int(self.mask_warp_row),
                self.mask_warp_col,
                self.scale,
                self.mask,
                self.p_reg_buffer.vectorize(),
                not_last_iter,
                self.cache_start_pos,
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
        attention_config: attention_config_t,
        output_ptr: UnsafePointer[Scalar[Self.output_type],],
        q: UnsafePointer[Scalar[Self.q_type]],
        k: k_t,
        v: v_t,
        mask: mask_t,
        sink_weights: OptionalReg[
            LayoutTensor[
                Self.q_type, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
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
            Self.q_tile_idx(),
            Self.kv_head_idx(),
            seq_len,
            Self.attention_config_t.get_q_offset[UInt(q_depth)](),
            Self.attention_config_t.get_output_offset[UInt(output_depth)](),
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

        self.mask_block_row = self.q_tile_idx() * Self.BM
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]
        var warp_col = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[1]
        self.mask_warp_row = warp_row * Int(Self.WM)
        self.mask_warp_col = warp_col * Int(Self.WN)

        self.batch_idx = batch_idx
        self.scale = scale

        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos

        @parameter
        if sink:
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
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(Int(Self.num_m_mmas), Int(Self.num_n_mmas)),
            # threads layout by warp
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
        var warp_row = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[0]
        var warp_col = get_warp_coords[Int(Self.BN), Int(Self.WN)]()[1]
        var output_tile = self.gmem_manager.get_output_tensor(self.output_ptr)
        var output_warp_tile = output_tile.tile[
            Int(Self.WM), Self.output_depth // Int(Self.num_warps_n)
        ](warp_row, warp_col)

        @parameter
        if Self.mma_shape[0] == 32:
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
        @parameter
        if not Self.token_gen:
            return

        self.p_reg_buffer.copy_to_shared()

    @always_inline
    fn store_partition_info(
        self,
        num_partitions: Int,
        exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        qk_max_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
    ):
        @parameter
        if not Self.token_gen:
            return

        var q_head_idx = self.q_head_idx()
        if num_partitions > 1:
            if thread_idx.x < UInt(group):
                var row_sum = self.rowsum[0, 0][0]
                var row_max = self.rowmax[0, 0][0]

                exp_sum_ptr[q_head_idx] = row_sum
                qk_max_ptr[q_head_idx] = row_max
