# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from gpu import (
    barrier,
    lane_id,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
    WARP_SIZE,
    grid_dim,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from nn.mha_operand import KVCacheMHAOperand, MHAOperand, NDBufferMHAOperand

from math.constants import log2e

from math import align_down, align_up, ceildiv, exp, recip

from memory import UnsafePointer, stack_allocation

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.sync import (
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
)
import gpu.warp as warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, IntTuple
from layout.layout_tensor import (
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_dram_to_local,
    copy_local_to_sram,
    copy_sram_to_dram,
    ThreadScope,
)
from nn.softmax import (
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax,
)
from nn.mha_utils import _kernel_mask

from nn.mha_mask import MHAMask, NullMask, TileMaskStatus

from layout.runtime_tuple import RuntimeTuple
from utils.numerics import min_or_neg_inf, neg_inf
from pathlib import Path
from algorithm.functional import tile_and_unswitch, unswitch, vectorize

from layout.runtime_layout import RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb, static
from layout.tensor_core import TensorCore, get_mma_shape
from linalg.utils import GemmShape
from math import align_down, ceildiv, align_up
from memory import UnsafePointer
from sys import simdwidthof, alignof
from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type
from linalg.utils_gpu import MatmulConfig
from linalg.utils import apply_epilogue, elementwise_epilogue_type
from layout.swizzle import Swizzle
from math import exp
from nn.mha_utils import MHAConfig


@always_inline
fn mma[
    transpose_b: Bool,
    k_group_size: Int,
    config: MHAConfig,
    swizzle: OptionalReg[Swizzle] = None,
](
    c: LayoutTensor,
    a: LayoutTensor,
    a_smem: LayoutTensor[a.dtype, address_space = AddressSpace.SHARED],
    b: LayoutTensor,
    b_smem: LayoutTensor[b.dtype, address_space = AddressSpace.SHARED],
):
    alias simd_width = simdwidthof[a.dtype]()
    alias mma_shape = get_mma_shape[a.dtype, get_accum_type[a.dtype]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias accum_type = get_accum_type[a.dtype]()
    alias WM = config.warp_m()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    var warp_id = thread_idx.x // WARP_SIZE
    alias num_warps = config.num_threads() // WARP_SIZE

    @parameter
    if a.address_space != AddressSpace.SHARED:
        copy_dram_to_sram[
            thread_layout = Layout.row_major(16, 4), swizzle=swizzle
        ](
            a_smem.tile[WM, depth](warp_id, 0).vectorize[1, simd_width](),
            a.tile[WM, depth](warp_id, 0).vectorize[1, simd_width](),
            a,
        )

    @parameter
    if b.address_space != AddressSpace.SHARED:
        copy_dram_to_sram[
            thread_layout = Layout.row_major(16, 4), swizzle=swizzle
        ](
            b_smem.tile[BN // num_warps, depth](warp_id, 0).vectorize[
                1, simd_width
            ](),
            b.tile[BN // num_warps, depth](warp_id, 0).vectorize[
                1, simd_width
            ](),
            b,
        )

    barrier()
    var mma_op = TensorCore[
        accum_type,
        a.dtype,
        (MMA_M, MMA_N, MMA_K),
        transpose_b=transpose_b,
    ]()

    var a_warp_tile = a_smem.tile[WM, depth](warp_id, 0)

    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(BN, MMA_N)
    alias num_k_mmas2 = ceildiv(depth, (MMA_K * k_group_size))

    var a_reg_tile = tb[a.dtype]().row_major[
        num_m_mmas * num_k_mmas2, simd_width
    ]().local().alloc()
    var b_reg_tile = tb[b.dtype]().row_major[
        num_n_mmas * num_k_mmas2, simd_width
    ]().local().alloc()

    @parameter
    for k_mma in range(Int(num_k_mmas2)):
        mma_op.load_a[swizzle= True if swizzle else False](
            a_warp_tile,
            a_reg_tile.tile[num_m_mmas, simd_width](k_mma, 0).vectorize[
                1, simd_width
            ](),
            k_mma,
        )
        mma_op.load_b[swizzle=swizzle](
            b_smem,
            b_reg_tile.tile[num_n_mmas, simd_width](k_mma, 0).vectorize[
                1, simd_width
            ](),
            k_mma,
        )

    @parameter
    for k_mma in range(Int(num_k_mmas2)):

        @parameter
        for k in range(k_group_size):
            alias elements_per_thread = simd_width // k_group_size
            var a_reg_k = a_reg_tile.tile[num_m_mmas, elements_per_thread](
                k_mma, k
            ).vectorize[1, elements_per_thread]()
            var b_reg_k = b_reg_tile.tile[num_n_mmas, elements_per_thread](
                k_mma, k
            ).vectorize[1, elements_per_thread]()
            mma_op.mma(a_reg_k, b_reg_k, c.vectorize[1, 4]())


@always_inline
fn _apply_mask[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
    mask_t: MHAMask,
    not_last_iter: Bool,
    group: Int,
](
    kv_tile_start_row: Int,
    kv_tile_num_rows: Int,
    start_pos: Int,
    seq_len: Int,
    num_keys: Int,
    mask_block_row: Int,
    mask_warp_row: Int,
    mask_warp_col: Int,
    scale: Float32,
    mask: mask_t,
    p_reg_vec2: LayoutTensor[accum_type, **_],
):
    var scale_log2e: SIMD[accum_type, 1] = scale.cast[accum_type]() * log2e
    var lane = lane_id()

    @parameter
    if token_gen:
        if lane >= 16:
            return

    @parameter
    for m_mma in range(Int(num_m_mmas)):

        @parameter
        for n_mma in range(Int(num_n_mmas)):
            alias mma_id = n_mma * num_m_mmas + m_mma
            # Coordinates in mask for current mma tile.
            var mask_frag_row = mask_warp_row + m_mma * MMA_M
            var mask_frag_col = mask_warp_col + n_mma * MMA_N
            mask_frag_row += (lane // MMA_N) * output_frag_size
            mask_frag_col += lane % MMA_N
            # The row in score matrix of shape seq_len x num_keys.
            # Mask col is score col since we don't partition in col.
            var score_row = (
                num_keys - 1
            ) if token_gen else mask_block_row + mask_frag_row
            var score_col = mask_frag_col
            var score_row_with_start_pos = score_row + start_pos
            alias i = 0

            @parameter
            if masked:

                @parameter
                for j in range(group if token_gen else output_frag_size):
                    var group_idx = (lane // 16) * output_frag_size + j
                    var q_head_idx = (
                        block_idx.y * group + group_idx
                    ) if token_gen else block_idx.y
                    p_reg_vec2[mma_id, i][j] = mask.mask(
                        IndexList[4, element_bitwidth=32, unsigned=True,](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos + j),
                            Int(score_col),
                        ),
                        p_reg_vec2[mma_id, i][j] * scale_log2e,
                    )
            else:
                p_reg_vec2[mma_id, i] = p_reg_vec2[mma_id, i] * scale_log2e
            if not not_last_iter or token_gen:
                var bound_y = kv_tile_start_row + kv_tile_num_rows if token_gen else num_keys

                @parameter
                for j in range(group if token_gen else output_frag_size):
                    var bound_x = num_keys + j if token_gen else seq_len

                    p_reg_vec2[mma_id, i][j] = _kernel_mask(
                        IndexList[2, element_bitwidth=32, unsigned=True,](
                            Int(score_row + j),
                            Int(score_col),
                        ),
                        IndexList[
                            2,
                            element_bitwidth=32,
                            unsigned=True,
                        ](bound_x, bound_y),
                        p_reg_vec2[mma_id, i][j],
                    )


@always_inline
fn apply_softmax_denominator[
    accum_type: DType, //, num_m_mmas: Int, num_n_mmas: Int
](
    out_reg_tile: LayoutTensor[accum_type, **_],
    rowsum: LayoutTensor[accum_type, **_],
):
    @parameter
    for m_mma in range(Int(num_m_mmas)):

        @parameter
        for n_mma in range(Int(num_n_mmas)):

            @parameter
            for i in range(out_reg_tile.shape[1]()):
                var rowsum_inv = recip(rowsum[m_mma, i])
                out_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rebind[
                    out_reg_tile.element_type
                ](rowsum_inv)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mha_single_batch[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
    token_gen: Bool = False,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    seq_len: Int,
    num_keys: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
):
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias kv_num_heads = num_heads // group

    constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simdwidthof[q_type]()

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias output_frag_size = (MMA_M * MMA_N) // WARP_SIZE
    alias k_group_size = 16 // simd_width
    alias num_k_mmas2 = ceildiv(depth, (MMA_K * k_group_size))
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.WM
    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(BN, MMA_N)
    var out_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, output_frag_size
    ]().local().alloc().fill(0)

    var warp_id = thread_idx.x // WARP_SIZE

    var kv_head_idx = block_idx.y if token_gen else block_idx.y // group

    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)),
        IntTuple(Int(num_heads * depth), 1),
    ) if not token_gen else Layout.row_major(BM, depth)

    var q_tile_num_rows = min(
        BM, UInt(seq_len) - q_tile_idx * BM
    ) if not token_gen else group
    var q_offset = depth * (
        (kv_head_idx * group if token_gen else block_idx.y)
        + num_heads * q_tile_idx * BM
    )

    var q_gmem_runtime_layout = RuntimeLayout(
        RuntimeTuple[q_gmem_layout.shape, unsigned=True](
            Int(q_tile_num_rows), depth
        ),
        RuntimeTuple[q_gmem_layout.stride, unsigned=True](
            num_heads * depth if not token_gen else depth, 1
        ),
    )

    var q_tile = LayoutTensor[q_type, q_gmem_layout, masked=True](
        q + Int(q_offset),
        q_gmem_runtime_layout,
    )

    var output_tile = LayoutTensor[output_type, q_gmem_layout, masked=True,](
        output + Int(q_offset),
        q_gmem_runtime_layout,
    )

    alias row_alignment = alignof[SIMD[accum_type, simdwidthof[accum_type]()]]()

    var rowmax = tb[accum_type]().row_major[
        num_m_mmas, output_frag_size
    ]().local().alloc()
    var rowsum = tb[accum_type]().row_major[
        num_m_mmas, output_frag_size
    ]().local().alloc()

    _ = rowmax.vectorize[1, output_frag_size]().fill(
        min_or_neg_inf[accum_type]()
    )
    _ = rowsum.vectorize[1, output_frag_size]().fill(0)

    @parameter
    fn get_smem_layout(
        tile_size: Int, block_size_m: Int, block_size_n: Int
    ) -> Layout:
        return Layout(
            IntTuple(
                IntTuple(tile_size, block_size_m // tile_size),
                IntTuple(
                    k_group_size * MMA_K, block_size_n // (k_group_size * MMA_K)
                ),
            ),
            IntTuple(
                IntTuple(k_group_size * MMA_K, block_size_n * tile_size),
                IntTuple(1, k_group_size * MMA_K * tile_size),
            ),
        )

    var q_smem = LayoutTensor[
        q_type,
        get_smem_layout(MMA_M, BM, depth),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var k_smem = LayoutTensor[
        k_t.type,
        get_smem_layout(MMA_N, BN, depth),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var p_smem = LayoutTensor[
        q_type,
        get_smem_layout(MMA_N, BM, BN),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var v_smem = LayoutTensor[
        v_t.type,
        Layout.row_major(BN, depth),
        address_space = AddressSpace.SHARED,
    ](k_smem.ptr.bitcast[Scalar[v_t.type]]())

    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2, BM),
        address_space = AddressSpace.SHARED,
    ](p_smem.ptr.bitcast[Scalar[accum_type]]())

    alias swizzle = Swizzle(2, 0, 2)

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_id * WM
    var mask_warp_col = 0

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout(
            RuntimeTuple[kv_gmem_layout.shape, unsigned=True](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, unsigned=True](
                kv_num_heads * depth, 1
            ),
        )

        var k_tile = LayoutTensor[
            k_t.type,
            kv_gmem_layout,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_runtime_layout,
        )
        var v_tile = LayoutTensor[
            v_t.type,
            kv_gmem_layout,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_runtime_layout,
        )

        var p_reg_tile = tb[accum_type]().row_major[
            num_m_mmas * num_n_mmas, output_frag_size
        ]().local().alloc().fill(0)

        if kv_tile_start_row == 0:
            mma[
                transpose_b=True,
                k_group_size=k_group_size,
                config=config,
                swizzle=swizzle,
            ](p_reg_tile, q_tile, q_smem, k_tile, k_smem)
        else:
            mma[
                transpose_b=True,
                k_group_size=k_group_size,
                config=config,
                swizzle=swizzle,
            ](p_reg_tile, q_smem, q_smem, k_tile, k_smem)

        var p_reg_vec2 = p_reg_tile.vectorize[1, output_frag_size]()

        # unswitch[_apply_mask](
        #     mask.status(
        #         Index[element_bitwidth=32, unsigned=True](
        #             Int(q_tile_idx * BM + start_pos), kv_tile_start_row
        #         ),
        #         Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
        #     )
        #     == TileMaskStatus.PARTIAL_MASK
        # )
        # ^ this somehow does not give correct results

        _apply_mask[
            masked=True,
            accum_type=accum_type,
            token_gen=token_gen,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            output_frag_size=output_frag_size,
            mask_t=mask_t,
            not_last_iter=not_last_iter,
            group=group,
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
            p_reg_vec2,
        )

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )

        # Not sure why we need this barrier here, but the code hangs without it
        barrier()

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(BM // WM, 1),
            Layout.row_major(4, 16),
            use_exp2=True,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[1, WM](0, Int(warp_id)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        copy_local_to_sram[thread_layout = Layout.row_major(4, 16)](
            p_smem.tile[WM, BN](warp_id, 0).vectorize[output_frag_size, 1](),
            p_reg_tile.vectorize[1, output_frag_size](),
        )
        # ensure that write to p_smem is finished
        barrier()

        # not using swizzle here as I need to figure out how to swizzle copy_local_to_sram correctly
        mma[transpose_b=False, k_group_size=k_group_size, config=config](
            out_reg_tile, p_smem, p_smem, v_tile, v_smem
        )

        # ensure that smem for v is not required anymore
        barrier()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[num_m_mmas=num_m_mmas, num_n_mmas=num_n_mmas](
        out_reg_tile, rowsum
    )

    var output_smem = LayoutTensor[
        output_type,
        Layout.row_major(BM, depth),
        address_space = AddressSpace.SHARED,
    ](q_smem.ptr.bitcast[Scalar[output_type]]())

    var output_smem_warp_tile = output_smem.tile[WM, depth](warp_id, 0)

    copy_local_to_sram[
        thread_layout = Layout.row_major(4, 16),
        thread_scope = ThreadScope.WARP,
    ](
        output_smem_warp_tile.vectorize[output_frag_size, 1](),
        out_reg_tile.vectorize[1, output_frag_size](),
    )

    barrier()

    alias num_threads = config.num_threads()

    alias copy_thread_layout = Layout.row_major(
        num_threads * simd_width // depth,
        depth // simd_width,
    )

    # TODO(KERN-1649): if swizzle is None, copy_sram_to_dram does not work
    # correctly with masked tensor
    copy_sram_to_dram[
        thread_layout=copy_thread_layout,
        num_threads=num_threads,
        swizzle = Swizzle(0, 0, 1),
    ](
        output_tile.vectorize[1, simd_width](),
        output_smem.vectorize[1, simd_width](),
    )
