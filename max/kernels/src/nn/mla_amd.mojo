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
from sys import simd_width_of
from sys.info import _cdna_4_or_newer
from sys.intrinsics import readfirstlane

from algorithm.functional import unswitch
from gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.memory import AddressSpace
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
)
from memory import AddressSpace as BaseAddressSpace
from layout import IntTuple, Layout, LayoutTensor
from layout.layout import blocked_product
from layout._utils import idx2crd

from layout.element import Element
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_local_to_shared,
    copy_dram_to_local,
    copy_local_to_dram,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle
from layout.tensor_core import (
    TensorCore,
    get_mma_shape,
    num_matrix_reg,
    TiledTensorCore,
)
from layout.int_tuple import UNKNOWN_VALUE
from memory import bitcast, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_iter_for_mma_output,
    softmax,
)
from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from .mha_gfx942 import (
    QRegisterBuffer,
    PRegisterBuffer,
    VBuffer,
    copy_local_to_dram2,
    KBuffer,
    pad,
    convert_f32_to_bf16,
    apply_softmax_denominator,
    SharedMemoryManager,
    _apply_mask,
    mma,
    copy_fragment_to_smem,
)


struct GlobalMemoryManager[
    dtype: DType,
    BM: UInt32,
    BN: UInt32,
    BK: UInt32,
    depth: UInt32,
    q_depth: UInt32,
    rope_depth: UInt32,
    num_heads: UInt32,
    group: UInt32,
    token_gen: Bool,
    cache_depth: UInt32,
    cache_num_heads: UInt32,
]:
    alias kv_num_heads = num_heads // group
    alias nope_depth = depth - rope_depth
    # BHSD layout for q and kv cache
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(q_depth)),
        IntTuple(Int(num_heads * q_depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(q_depth))

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)),
        IntTuple(Int(num_heads * depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(Self.nope_depth))

    alias kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(Self.kv_num_heads * depth), 1),
    )

    alias k_rope_gmem_layout = Layout(
        IntTuple(Int(BN), Int(cache_depth)),
        IntTuple(Int(cache_num_heads * cache_depth), 1),
    )

    var q_offset: UInt32
    var q_runtime_layout: RuntimeLayout[
        Self.q_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    var output_offset: UInt32
    var output_runtime_layout: RuntimeLayout[
        Self.output_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    @always_inline
    fn __init__(
        out self, q_tile_idx: UInt32, kv_head_idx: UInt32, seq_len: Int
    ):
        var q_tile_num_rows = min(
            BM, UInt(seq_len) - q_tile_idx * BM
        ) if not token_gen else group

        self.q_offset = (
            q_depth
            * (
                block_idx.y + num_heads * q_tile_idx * BM
            ) if not token_gen else q_depth
            * q_tile_idx
            * BM
        )

        self.output_offset = (
            depth
            * (
                block_idx.y + num_heads * q_tile_idx * BM
            ) if not token_gen else Self.nope_depth
            * q_tile_idx
            * BM
        )

        self.q_runtime_layout = type_of(self.q_runtime_layout)(
            RuntimeTuple[
                Self.q_gmem_layout.shape,
                element_type = type_of(self.q_runtime_layout).element_type,
            ](Int(q_tile_num_rows), Int(q_depth)),
            RuntimeTuple[
                Self.q_gmem_layout.stride,
                element_type = type_of(self.q_runtime_layout).linear_idx_type,
            ](Int(num_heads * q_depth if not token_gen else q_depth), 1),
        )

        self.output_runtime_layout = type_of(self.output_runtime_layout)(
            RuntimeTuple[
                Self.output_gmem_layout.shape,
                element_type = type_of(self.output_runtime_layout).element_type,
            ](
                Int(q_tile_num_rows),
                Int(Self.depth) if not token_gen else Int(Self.nope_depth),
            ),
            RuntimeTuple[
                Self.output_gmem_layout.stride,
                element_type = type_of(
                    self.output_runtime_layout
                ).linear_idx_type,
            ](Int(num_heads * depth if not token_gen else Self.nope_depth), 1),
        )

    @always_inline
    fn get_q_tensor[
        qtype: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[qtype]],
        out result: LayoutTensor[
            qtype,
            Self.q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return type_of(result)(
            ptr + Int(self.q_offset),
            self.q_runtime_layout,
        )

    @always_inline
    fn get_output_tensor[
        out_type: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[out_type]],
        out result: LayoutTensor[
            out_type,
            Self.output_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return type_of(result)(
            ptr + Int(self.output_offset),
            self.output_runtime_layout,
        )

    @always_inline
    fn get_kv_tensor[
        kvtype: DType, //,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], **_],
        kv_tile_num_rows: UInt32,
        out result: LayoutTensor[
            kvtype,
            Self.kv_gmem_layout,
            ptr.origin,
            masked=True,
            address_space = ptr.address_space,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * depth), 1
            ),
        )

        return type_of(result)(
            ptr,
            kv_runtime_layout,
        )

    @always_inline
    fn get_krope_tensor[
        kvtype: DType, //,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], **_],
        kv_tile_num_rows: UInt32,
        out result: LayoutTensor[
            kvtype,
            Self.k_rope_gmem_layout,
            ptr.origin,
            masked=True,
            address_space = ptr.address_space,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var k_rope_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(cache_depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(cache_num_heads * cache_depth), 1
            ),
        )

        return type_of(result)(
            ptr,
            k_rope_runtime_layout,
        )


@always_inline
fn mla_prefill_single_batch_amd[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    seq_len: Int,
    num_keys: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: UInt32,
    cache_start_pos: UInt32,
    mask: mask_t,
):
    alias token_gen = False
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias rope_depth = q_depth - depth
    alias num_heads = config.num_heads
    alias BK = config.block_k()
    alias simd_width = simd_width_of[q_type]()

    alias mma_shape = IndexList[3](32, 32, 16) if (
        _cdna_4_or_newer()
        and depth != 64
        # will deal with 64 later
    ) else IndexList[3](32, 32, 8)
    alias cache_num_heads = num_heads // UInt(group)

    alias fragment_layout = Layout.row_major(1, 16)
    alias fragment_layout_nested = Layout(
        IntTuple(1, IntTuple(4, 4)), IntTuple(1, IntTuple(1, 8))
    )
    alias warp_layout = Layout.col_major(32, 2)
    alias swap_a_b = True
    alias k_group_size = 16 // mma_shape[2]

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, UInt(mma_shape[0]))
    alias num_n_mmas = ceildiv(WN, UInt(mma_shape[1]))
    alias num_k_mmas2 = ceildiv(BK, UInt(mma_shape[2] * k_group_size))
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    alias WN_O = depth
    alias num_n_mmas_output = WN_O // UInt(mma_shape[1])

    var out_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas_output, output_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = block_idx.y  # // group

    var q_tile_idx = block_idx.x

    var gmem_manager = GlobalMemoryManager[
        q_type,
        BM,
        BN,
        BK,
        depth,
        q_depth,
        0,
        num_heads,
        1,
        token_gen,
        cache_depth,
        cache_num_heads,
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas, fragment_layout.shape[0].value()),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(min_or_neg_inf[accum_type]())
    )

    var rowsum = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas, fragment_layout.shape[0].value()),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen
    ]()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    var q_buffer = QRegisterBuffer[
        mma_shape=mma_shape,
        k_group_size=k_group_size,
        WM=WM,
        WN=WN,
        BN=BN,
        BK=BK,
        depth=q_depth,
        thread_layout=warp_layout,
    ](q_tile)

    q_buffer.load_from_dram()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int
    ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
        var mask_status = mask.status(
            Index[dtype = DType.uint32](
                Int(q_tile_idx * BM + start_pos),
                Int(kv_tile_start_row + cache_start_pos),
            ),
            Index[dtype = DType.uint32](Int(BM), Int(BN)),
        )

        if mask_status == TileMaskStatus.FULL_MASK:
            mask_warp_col += BN
            return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var k_rope_tile = gmem_manager.get_krope_tensor(
            k_rope.block_paged_ptr[BN](
                batch_idx,
                kv_tile_start_row + cache_start_pos,
                Int(kv_head_idx // UInt(group)),
                cache_depth - rope_depth,
            ),
            kv_tile_num_rows,
        )

        var p_buffer = PRegisterBuffer[
            accum_type,
            q_type,
            num_m_mmas,
            num_n_mmas,
            output_frag_size,
        ]()

        var num_b_rows = Int(kv_tile_num_rows)
        alias num_threads = config.num_threads()

        var v_buffer = VBuffer[
            mma_shape=mma_shape,
            k_group_size=k_group_size,
            BN=BN,
            BK=BK,
            depth=depth,
            num_threads=num_threads,
        ](v_tile, smem_manager.get_kv_ptr[v_tile.dtype]())

        var k_buffer = KBuffer[
            mma_shape=mma_shape,
            k_group_size=k_group_size,
            swizzle=None,
            BN=BN,
            WN=BN,
            BK=BK,
            num_threads=num_threads,
        ](
            k_tile,
            num_b_rows,
            smem_manager.get_kv_ptr[k_tile.dtype](),
        )

        var k_rope_buffer = KBuffer[
            mma_shape=mma_shape,
            k_group_size=k_group_size,
            swizzle=None,
            BN=BN,
            WN=BN,
            BK=BK,
            num_threads=num_threads,
        ](
            k_rope_tile,
            num_b_rows,
            smem_manager.get_kv_ptr[k_rope_tile.dtype](),
        )

        alias tensor_core_mma = TiledTensorCore[
            accum_type,
            q_type,
            mma_shape,
            group_size=k_group_size,
            transpose_b=True,
        ]()

        # calculate k q ^T
        k_buffer.load_from_dram()

        @parameter
        for i in range(depth // BK):
            k_buffer.copy_to_shared()

            barrier()

            @parameter
            if i < depth // BK - 1:
                k_buffer.load_from_dram()

                @parameter
                if i == depth // BK - 2:
                    # prefetch k_rope from dram
                    k_rope_buffer.load_from_dram()

            @parameter
            for k_mma in range(num_k_mmas2):
                var q_mma_tile = q_buffer.get_mma_tile[i, k_mma]()
                k_buffer.load_from_shared[
                    # TODO: I should be able to use tensor_core_mma here
                    # but getting compiler errors
                    accum_type,
                    q_type,
                    mma_shape,
                    k_group_size,
                    True,
                    k_mma,
                ]()
                var k_mma_tile = k_buffer.get_mma_tile()
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    q_mma_tile, k_mma_tile, p_buffer.reg_tile
                )

            barrier()

        # I tried this and previous block as a lambda but compiler was crashing
        # so duplicating code for now
        @parameter
        for i in range(rope_depth // BK):
            k_rope_buffer.copy_to_shared()

            barrier()

            @parameter
            if i < rope_depth // BK - 1:
                k_rope_buffer.load_from_dram()

                @parameter
                if i == rope_depth // BK - 2:
                    # prefetch v from dram
                    v_buffer.load_from_dram()

            @parameter
            for k_mma in range(num_k_mmas2):
                var q_mma_tile = q_buffer.get_mma_tile[i + depth // BK, k_mma]()
                k_rope_buffer.load_from_shared[
                    # TODO: I should be able to use tensor_core_mma here
                    # but getting compiler errors
                    accum_type,
                    q_type,
                    mma_shape,
                    k_group_size,
                    True,
                    k_mma,
                ]()
                var k_mma_tile = k_rope_buffer.get_mma_tile()
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    q_mma_tile, k_mma_tile, p_buffer.reg_tile
                )

            barrier()

        var p_reg_vectorized = p_buffer.reg_tile.vectorize[
            1, output_frag_size
        ]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M = mma_shape[0],
                MMA_N = mma_shape[1],
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=1,
                fragment_layout=fragment_layout_nested,
                warp_layout=warp_layout,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
                not_last_iter,
                cache_start_pos,
            )

        unswitch[_apply_mask_impl](mask_status == TileMaskStatus.PARTIAL_MASK)

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )
        # don't know why we need this barrier but i get random failures without it
        barrier()
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_buffer.reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        barrier()

        # calculate v^T p^T

        @parameter
        for i in range(BN // BK):
            # v has been prefetched from dram during the last mma
            v_buffer.copy_to_shared()

            @parameter
            if i < (BN // BK) - 1:
                v_buffer.load_from_dram()

            # ensure that shared memory is filled
            barrier()

            v_buffer.load_from_shared()

            var p_mma_tile_interleaved = p_buffer.interleave[i]()

            @parameter
            for k_mma_idx in range(v_buffer.num_k_tiles):
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    p_mma_tile_interleaved.tile[1, simd_width](0, k_mma_idx),
                    v_buffer.mma_tile.tile[
                        depth // UInt(mma_shape[0]), simd_width
                    ](k_mma_idx, 0),
                    out_reg_tile,
                )

            barrier()

    for i in range(UInt32(0), UInt32(num_keys), UInt32(BN)):
        var end = min(i + BN, num_keys)
        loop_over_kvcache[BN](i, end, end != num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    var output_warp_tile = output_tile.tile[WM, WN](warp_row, warp_col)

    copy_local_to_dram2[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            1,
            4,
        ](),
        out_reg_tile.vectorize[1, 4](),
        output_tile,
    )


@always_inline
fn mla_decoding_single_batch_amd[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask, //,
    depth_v: Int,
    group: Int,
    config: MHAConfig,
    sink: Bool = False,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    seq_len: Int,
    num_keys: Int,
    num_partitions: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
):
    alias token_gen = True
    alias rope_depth = depth - UInt(depth_v)
    alias nope_depth = UInt(depth_v)
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias kv_num_heads = num_heads // UInt(group)
    alias BK = config.block_k()
    # constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simd_width_of[q_type]()

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias use_transposed_layout = True
    alias fragment_layout = Layout.row_major(
        1, 4
    ) if use_transposed_layout else Layout.row_major(4, 1)
    alias warp_layout = Layout.col_major(
        16, 4
    ) if use_transposed_layout else Layout.row_major(4, 16)
    alias swap_a_b = use_transposed_layout
    alias k_group_size = 2

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, UInt(MMA_M))
    alias num_n_mmas = ceildiv(WN, UInt(MMA_N))
    alias num_n_mmas_depth = UInt(
        ceildiv(nope_depth // num_warps_n, UInt(mma_shape[1]))
    )

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var out_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas_depth, output_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = 0
    var q_head_idx = block_idx.y

    alias rowwise_stride = fragment_layout.shape[0].value()
    var q_tile_idx = q_head_idx
    var lane = lane_id()

    alias q_depth = depth

    constrained[q_depth == depth, "q_depth must be equal to depth"]()

    var gmem_manager = GlobalMemoryManager[
        q_type,
        BM,
        BN,
        BK,
        depth,
        q_depth,
        rope_depth,
        num_heads,
        group,
        token_gen,
        0,
        0,
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, fragment_layout.shape[0].value()),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    var rowsum = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, fragment_layout.shape[0].value()),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    @parameter
    if sink:
        debug_assert(
            Bool(sink_weights),
            "expect sink_weights to be non-null when sink=true",
        )
        var sink_weight = (
            sink_weights.value()[Int(q_head_idx)][0].cast[accum_type]() * log2e
        )
        rowmax = rowmax.fill(sink_weight)
        rowsum = rowsum.fill(1)
    else:
        rowmax = rowmax.fill(min_or_neg_inf[accum_type]())
        rowsum = rowsum.fill(0)

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen, nope_depth
    ]()

    var p_smem_iter = smem_manager.get_p_iter()
    var k_smem_iter = smem_manager.get_k_iter()
    var v_smem_iter = smem_manager.get_v_iter()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    # the following assumes BK == 32, i.e. simd_width = 2*frag_size
    alias q_reg_size = (depth // BK) * num_m_mmas * UInt(simd_width)

    var q_reg_data = stack_allocation[
        q_reg_size,
        q_type,
        address_space = AddressSpace.LOCAL,
    ]()

    var q_reg_tile_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(num_m_mmas, simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ](q_reg_data, q_reg_size)

    var q_gmem_warp_iter = q_tile.tiled_iterator[WM, BK, axis=1](warp_row, 0)

    @parameter
    for i in range(depth // BK):
        var q_reg_tile = q_reg_tile_iter.next_unsafe(i)[]
        copy_dram_to_local[
            src_thread_layout = Layout.col_major(16, 4),
            thread_scope = ThreadScope.WARP,
        ](
            q_reg_tile.vectorize[1, simd_width](),
            q_gmem_warp_iter,
            q_tile.dim[0]() * q_tile.stride[0](),
        )
        q_gmem_warp_iter._incr()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int
    ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
        @parameter
        if mask_t.check_mask_during_decoding:
            var mask_status = mask.status(
                Index[dtype = DType.uint32](
                    Int(num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(BN)),
            )

            if mask_status == TileMaskStatus.FULL_MASK:
                return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )
        var k_global_iterator = k_tile.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_global_iterator = v_tile.slice[
            :, : Int(nope_depth)
        ]().tiled_iterator[BK, nope_depth, axis=0](0, 0)

        var p_reg_tile = (
            LayoutTensor[
                accum_type,
                Layout.row_major(num_m_mmas * num_n_mmas, output_frag_size),
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ]
            .stack_allocation()
            .fill(0)
        )

        alias swizzle = Swizzle(2, 0, 2)

        var num_b_rows = OptionalReg[Int](
            kv_tile_num_rows
        ) if not not_last_iter else None

        # TODO (KERN-1708):this is just a dummy iterator to satisfy the interface
        # will fix it with better interface later
        var q_smem_iter = LayoutTensorIter[
            q_type,
            Layout.row_major(num_m_mmas, simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](
            UnsafePointer[
                Scalar[q_type], address_space = AddressSpace.SHARED
            ](),
            q_reg_size,
        )

        @parameter
        @always_inline
        fn prefetch_function[tile_id: Int]():
            ...

        mma[
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            transpose_b=True,
            k_group_size=k_group_size,
            config=config,
            prefetch_function=prefetch_function,
            swizzle=swizzle,
            swap_a_b=swap_a_b,
            num_iters = Int(depth // BK),
            token_gen=token_gen,
        ](
            p_reg_tile,
            q_reg_tile_iter,
            q_smem_iter,
            k_global_iterator,
            k_smem_iter,
            num_b_rows,
        )

        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=group,
                fragment_layout=fragment_layout,
                warp_layout=warp_layout,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
                not_last_iter,
            )

        @parameter
        if mask_t.check_mask_during_decoding:
            var mask_status = mask.status(
                Index[dtype = DType.uint32](
                    Int(num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(1), Int(BN)),
            )
            unswitch[_apply_mask_impl](
                mask_status == TileMaskStatus.PARTIAL_MASK
            )
        else:
            _apply_mask_impl[masked=True]()

        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )
        alias reg_layout_by_mma_unit_depth = Layout.row_major(
            num_m_mmas * num_n_mmas_depth, output_frag_size
        )

        # Not sure why we need this barrier here, but the code hangs without it
        barrier()

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit_depth]().vectorize[
                1, output_frag_size
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        copy_fragment_to_smem[
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_N,
            num_m_mmas,
            num_n_mmas,
            fragment_layout,
            warp_layout,
        ](
            p_smem_iter,
            p_reg_vectorized,
            warp_row,
            warp_col,
        )

        barrier()

        mma[
            BM=BM,
            BN=nope_depth,
            BK=BK,
            WM=WM,
            WN = nope_depth // num_warps_n,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            transpose_b=False,
            k_group_size=k_group_size,
            config=config,
            prefetch_function=prefetch_function,
            swizzle=None,
            swap_a_b=swap_a_b,
            num_iters = Int(BN // BK),
            token_gen=token_gen,
        ](
            out_reg_tile,
            p_smem_iter,
            p_smem_iter,
            v_global_iterator,
            v_smem_iter,
            num_b_rows,
        )

        # ensure that smem for v is not required anymore
        barrier()

    start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    for i in range(start, end, BN):
        var end_ = min(i + BN, end)
        loop_over_kvcache[BN](i, end_, end_ != end)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas_depth,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    if num_partitions > 1:
        if thread_idx.x < UInt(group):
            var row_sum = rowsum[0, 0][0]
            var row_max = rowmax[0, 0][0]

            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    var output_warp_tile = output_tile.tile[WM, nope_depth // num_warps_n](
        warp_row, warp_col
    )

    copy_local_to_dram[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            fragment_layout.shape[0].value(),
            fragment_layout.shape[1].value(),
        ](),
        out_reg_tile.vectorize[1, output_frag_size](),
        output_tile,
    )
