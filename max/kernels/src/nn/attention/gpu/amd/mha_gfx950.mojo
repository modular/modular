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

from itertools import product
from math import ceildiv
from sys import simd_width_of, llvm_intrinsic
from sys.intrinsics import readfirstlane, _type_is_eq
from sys.info import _cdna_4_or_newer
from gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
    s_waitcnt,
)
from memory.pointer import AddressSpace as BaseAddressSpace
from layout import IntTuple, Layout, LayoutTensor
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import blocked_product
from layout._utils import make_amd_buffer_resource, idx2crd
from layout.element import Element
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_layout import RuntimeLayout
from layout.swizzle import Swizzle
from layout.tensor_core import (
    TensorCore,
    get_mma_shape,
    num_matrix_reg,
    TiledTensorCore,
)
from memory import bitcast, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus, CausalMask
from nn.mha_operand import MHAOperand
from nn.mha_utils import (
    MHAConfig,
    _kernel_mask,
    get_start_and_end_for_partitions,
)

from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from .mha_gfx942 import Attention
from .utils import load_b, load_b_tr, copy_dram_to_sram_lds


# Note: this is a experimental implementation of MHA for gfx950.


@always_inline
fn set_priority[priority: Int]():
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](Int16(priority))


@always_inline
fn scheduling_hints_qk[group: Int]():
    comptime for i in range(4):
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(group))

        comptime for j in range(4):
            schedule_group_barrier(AMDScheduleBarrierMask.VALU, 1, Int32(group))
            schedule_group_barrier(
                AMDScheduleBarrierMask.TRANS, 1, Int32(group)
            )

    comptime for i in range(12):
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(group))
        schedule_group_barrier(AMDScheduleBarrierMask.VALU, 6, Int32(group))


@always_inline
fn scheduling_hints_pv[group: Int]():
    comptime for i in range(12):
        schedule_group_barrier(AMDScheduleBarrierMask.VALU, 4, Int32(group))
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(group))
        schedule_group_barrier(AMDScheduleBarrierMask.VALU, 10, Int32(group))

    comptime for i in range(4):
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(group))

        comptime for i in range(4):
            schedule_group_barrier(AMDScheduleBarrierMask.VALU, 1, Int32(group))
            schedule_group_barrier(
                AMDScheduleBarrierMask.TRANS, 1, Int32(group)
            )


@always_inline
fn barrier[
    *, schedule_barrier_before: Bool = True, schedule_barrier_after: Bool = True
]():
    comptime if schedule_barrier_before:
        schedule_barrier()

    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

    comptime if schedule_barrier_after:
        schedule_barrier()


@always_inline
fn block_sync_lds[
    *,
    lgkmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS (local data share) with waitcnt barrier.
    """

    s_waitcnt[lgkmcnt=lgkmcnt]()


@always_inline
fn block_sync_lds_direct_load[
    *,
    vmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS for direct load with waitcnt barrier.
    """
    s_waitcnt[vmcnt=vmcnt]()


struct KVCacheIterator[
    cache_t: MHAOperand, tile_size: Int, kv_num_heads: Int, depth: Int
]:
    comptime kv_gmem_layout = Layout(
        IntTuple(Self.tile_size, Self.depth),
        IntTuple(Self.kv_num_heads * Self.depth, 1),
    )
    var cache: Self.cache_t
    var end: Int
    var tile_start_row: Int
    var batch_idx: Int
    var kv_head_idx: Int

    @always_inline
    fn __init__(
        out self,
        cache: Self.cache_t,
        batch_idx: Int,
        kv_head_idx: Int,
        end: Int,
    ):
        self.cache = cache
        self.end = end
        self.tile_start_row = 0
        self.batch_idx = batch_idx
        self.kv_head_idx = kv_head_idx

    @always_inline
    fn next_unsafe(
        mut self,
        out result: LayoutTensor[
            Self.cache_t.dtype,
            Self.kv_gmem_layout,
            MutAnyOrigin,
            masked=True,
        ],
    ):
        var kv_tile_num_rows = min(
            Self.tile_size,
            self.end - self.tile_start_row,
        )
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(kv_tile_num_rows, Self.depth),
            type_of(result.runtime_layout.stride)(
                Self.kv_num_heads * Self.depth, 1
            ),
        )
        var out = type_of(result)(
            self.cache.block_paged_ptr[Self.tile_size](
                UInt32(self.batch_idx),
                UInt32(self.tile_start_row),
                UInt32(self.kv_head_idx),
                0,
            ),
            kv_runtime_layout,
        )
        self.tile_start_row += Self.tile_size
        return out

    @always_inline
    fn increment(mut self):
        self.tile_start_row += Self.tile_size


struct KVBuffer[
    kv_t: MHAOperand,
    //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
]:
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth, Self.MMA_N
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime num_k_tiles = ceildiv(
        Self.depth if Self.transpose else Self.WN, Self.BK
    )

    comptime num_repeats = Self.depth // Self.BK
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime base_layout = Layout.row_major(Self.BN, Self.BK)
    comptime smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)

    comptime MMATileType = LayoutTensor[
        Self.kv_t.dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_mmas2 * Self.num_k_tiles, Self.simd_width
        ),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SharedIterType = LayoutTensorIter[
        Self.kv_t.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    comptime SharedTileType = Self.SharedIterType.LayoutTensorType
    comptime SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var kv_cache_iter: KVCacheIterator[
        Self.kv_t, Self.BN, Self.kv_num_heads, Self.depth
    ]

    var lds_base_ptrs: InlineArray[UInt32, 2]

    var warp_id: UInt32

    @always_inline
    fn __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype],
            address_space = AddressSpace.SHARED,
            ...,
        ],
        end: UInt,
        warp_id: UInt32,
    ):
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, Int(batch_idx), Int(head_idx), Int(end)
        )

        self.warp_id = warp_id

        var warp_row = self.warp_id // UInt32(4)
        var warp_col = self.warp_id % UInt32(4)

        self.lds_base_ptrs = type_of(self.lds_base_ptrs)(uninitialized=True)

        comptime for i in range(2):
            var smem_tile = self.smem_iter.next_unsafe(
                self.smem_iter.linear_uint_type(i)
            )[]
            var smem_warp_tile = smem_tile.tile[32, Self.BK](
                Int(warp_row), Int(warp_col)
            )
            self.lds_base_ptrs[i] = UInt32(
                readfirstlane(Int32(Int(smem_warp_tile.ptr)))
            )

    @always_inline
    fn load_from_dram[buffer_idx: Int](mut self):
        var global_tile = self.kv_cache_iter.next_unsafe()

        var smem_tile = self.smem_iter.next_unsafe(
            self.smem_iter.linear_uint_type(buffer_idx)
        )[]

        comptime if Self.depth == 64:
            var smem_warp_tile = smem_tile.tile[32, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            var gmem_warp_tile = global_tile.tile[32, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            # load from dram to sram directly
            copy_dram_to_sram_lds[swizzle = Self.swizzle,](
                smem_warp_tile,
                gmem_warp_tile,
            )
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE

            comptime for depth_tile in range(Self.depth // 128):
                var warp_row = self.warp_id // UInt32(4)
                var warp_col = self.warp_id % UInt32(4)
                var smem_warp_tile = smem_tile.tile[32, Self.BK](
                    Int(warp_row), Int(warp_col) + num_warps * depth_tile
                )
                var gmem_warp_tile = global_tile.tile[32, Self.BK](
                    Int(warp_row), Int(warp_col) + num_warps * depth_tile
                )
                # load from dram to sram directly
                copy_dram_to_sram_lds[swizzle = Self.swizzle,](
                    smem_warp_tile,
                    gmem_warp_tile,
                    self.lds_base_ptrs[buffer_idx],
                )

    @always_inline
    fn get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> Self.MMATileType.SplitElementType[
        Self.num_k_tiles
    ].SplitElementType[Self.num_k_mmas2]:
        return self.mma_tile.split[Self.num_k_tiles]()[bk_tile_idx].split[
            self.num_k_mmas2
        ]()[k_mma_tile_idx]

    @always_inline
    fn copy_to_shared(
        self,
    ):
        ...

    @always_inline
    fn load_from_shared(self, buffer: UInt):
        comptime for bk_tile in range(Self.num_k_tiles):
            self.load_from_shared[bk_tile](buffer)

    @always_inline
    fn load_from_shared[bk_tile: Int](self, buffer: UInt):
        comptime if Self.transpose:
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = get_warp_id() % UInt(num_warps_n)
            var smem_tile = self.smem_iter.next_unsafe(
                self.smem_iter.layout_uint_type(buffer)
            )[].tile[Self.BN, Self.BK](0, bk_tile)

            var wtile_coord0 = Int(warp_col)
            var wtile_coord1 = 0
            var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
                wtile_coord0, wtile_coord1
            )
            var load_b_tile = load_b[Self.mma_shape, swizzle = Self.swizzle](
                warp_tile
            )

            self.mma_tile.split[Self.num_k_tiles]()[bk_tile].vectorize[
                1, Self.simd_width
            ]().copy_from(load_b_tile.vectorize[1, Self.simd_width]())

        else:
            comptime MMA_M = Self.mma_shape[0]
            comptime MMA_K = Self.mma_shape[2]

            comptime for k in range(Self.BK // MMA_K):
                var smem_tile = (
                    self.smem_iter.next_unsafe(
                        self.smem_iter.layout_uint_type(buffer)
                    )[]
                    .tile[Self.BK, Self.depth](bk_tile, 0)
                    .tile[MMA_K, Self.depth](k, 0)
                )
                var frags = (
                    type_of(
                        self.mma_tile.split[Self.num_k_tiles]()[bk_tile].split[
                            Self.num_k_mmas2
                        ]()[k]
                    )
                    .stack_allocation()
                    .vectorize[1, Self.simd_width]()
                )

                comptime for i in range(Self.depth // MMA_M):
                    comptime tile_layout = type_of(
                        smem_tile.tile[MMA_K, MMA_M](0, i)
                    ).layout
                    # TODO: KERN-2173, the offset calculation is a workaround
                    # a bug in tile, remove this once the bug is fixed
                    comptime tiles_per_bk = Self.BK // MMA_M
                    comptime stride = self.base_layout.size()
                    comptime offset = (
                        MMA_M * (i % tiles_per_bk)
                        + (i // tiles_per_bk) * stride
                    )
                    var tile = LayoutTensor[
                        smem_tile.dtype,
                        tile_layout,
                        MutAnyOrigin,
                        address_space = smem_tile.address_space,
                    ](smem_tile.ptr + offset)
                    frags[i, 0] = rebind[frags.element_type](
                        load_b_tr[Self.mma_shape](tile)
                    )
                var mma_tile = self.get_mma_tile[k, bk_tile]()
                mma_tile.vectorize[1, Self.simd_width]().copy_from(frags)


__extension Attention:
    @always_inline
    fn get_num_rows(self) -> UInt32:
        var end = min(
            self.kv_start_row + UInt32(Self.BN), UInt32(self.num_keys)
        )
        var num_rows = max(
            min(Int32(end - self.kv_start_row), Int32(UInt32(Self.BN))), 0
        )
        return UInt32(num_rows)

    @always_inline
    fn apply_mask[stage: Int](mut self, not_last_iter: Bool = False):
        self.scale_p_reg[stage]()
        var num_rows = self.get_num_rows()
        self.mask_apply[stage](self.kv_start_row, num_rows, not_last_iter)
        self.kv_start_row += UInt32(Self.BN)

    @always_inline
    fn online_softmax_step_0[stage: Int, mask: Bool = True](mut self):
        comptime if mask:
            self.apply_mask[stage]()
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.calculate_qk_max(score_reg_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_reg_tile)

    @always_inline
    fn online_softmax_step_1[stage: Int](mut self):
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.exp[start=1, stride=2](score_reg_tile)
        self.softmax.calculate_qk_sum(score_reg_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

    @always_inline
    fn online_softmax_update_output(mut self):
        var output_reg_tile = self.out_reg_buffer.vectorize()
        self.softmax.update_output(output_reg_tile)

    @always_inline
    fn online_softmax_full[stage: Int](mut self):
        self.apply_mask[stage]()
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.calculate_qk_max(score_reg_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_reg_tile)

        self.softmax.exp[start=1, stride=2](score_reg_tile)
        self.softmax.calculate_qk_sum(score_reg_tile, warp_scratch)
        self.softmax.calculate_correction()
        self.softmax.update_max()
        self.softmax.update_sum()

        var output_reg_tile = self.out_reg_buffer.vectorize()
        self.softmax.update_output(output_reg_tile)

    @always_inline
    fn mha_prefill_experimental(mut self):
        comptime assert Self.BK == 32, "BK must be 32"
        comptime assert Self.depth == 128, "depth must be 128"

        comptime num_threads = config.num_threads()
        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )
        var high_warps = warp_id // 4
        var k_buffer = KVBuffer[
            mma_shape = Self.mma_shape,
            k_group_size = Self.k_group_size,
            swizzle = Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None),
            BN = Int(Self.BN),
            WN = Int(Self.WN),
            BK = Int(Self.BK),
            num_threads = Int(Self.num_threads),
            depth = Int(Self.depth),
            kv_num_heads = Int(Self.num_heads) // Self.group,
            transpose=True,
        ](
            self.k,
            UInt(self.batch_idx),
            self.kv_head_idx(),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            UInt(self.num_keys),
            warp_id,
        )

        var v_buffer = KVBuffer[
            mma_shape = Self.mma_shape,
            k_group_size = Self.k_group_size,
            swizzle=None,
            BN = Int(Self.BN),
            WN = Int(Self.WN),
            BK = Int(Self.BK),
            num_threads = Int(Self.num_threads),
            depth = Int(Self.depth),
            kv_num_heads = Int(Self.num_heads) // Self.group,
            transpose=False,
        ](
            self.v,
            UInt(self.batch_idx),
            self.kv_head_idx(),
            self.smem_manager.get_v_ptr[type_of(self.v).dtype](),
            UInt(self.num_keys),
            warp_id,
        )

        comptime accum_type = get_accum_type[type_of(self.k).dtype]()
        comptime simd_width = simd_width_of[Self.q_type]()

        @always_inline
        @parameter
        fn mma_qk[stage: Int]():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size = Self.k_group_size,
                transpose_b=True,
            ]()
            self.zero_p_buffer[stage]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    var q_mma_tile = self.q_buffer.get_mma_tile[
                        Int(i), Int(k_mma)
                    ]()

                    var k_mma_tile = k_buffer.get_mma_tile[Int(k_mma), Int(i)]()
                    tensor_core_mma.mma[swap_a_b = Self.swap_a_b](
                        q_mma_tile,
                        k_mma_tile,
                        self.p_reg_buffer.get_reg_tile[stage](),
                    )

        @always_inline
        @parameter
        fn mma_pv[stage: Int]():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size = Self.k_group_size,
                transpose_b=True,
            ]()

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    tensor_core_mma.mma[swap_a_b = Self.swap_a_b](
                        self.p_reg_buffer.get_mma_tile[Int(i), k_mma, stage](),
                        v_buffer.get_mma_tile[k_mma, Int(i)](),
                        self.out_reg_buffer.reg_tile,
                    )

        # The pipeline follows the scheduling pattern in the paper "HipKittens: Fast and Furious AMD Kernels"
        # paper:https://arxiv.org/abs/2511.08083
        # code reference: https://github.com/HazyResearch/HipKittens

        _ = k_buffer.load_from_dram[0]()
        block_sync_lds_direct_load[vmcnt=0]()
        barrier[schedule_barrier_after=False]()

        _ = k_buffer.load_from_dram[1]()
        _ = v_buffer.load_from_dram[0]()
        k_buffer.load_from_shared(0)
        schedule_barrier()
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=2]()
        barrier[schedule_barrier_after=False]()

        mma_qk[0]()
        self.online_softmax_step_0[0]()
        schedule_barrier()

        # stagger warps
        if high_warps == 1:
            barrier[schedule_barrier_after=False]()

        k_buffer.load_from_shared(1)
        _ = k_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[1]()
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=4]()
        barrier[schedule_barrier_after=False]()

        comptime break_mask = False

        @always_inline
        @parameter
        fn loop_over_kvcache[tile_size: Int](end: UInt32):
            # barrier()
            # TODO: enable skipping this for other masks, this is not required for the causal mask but will help with other masks
            # if self.mask_skip_and_advance(self.kv_tile_start_row):
            #     k_buffer.kv_cache_iter.increment()
            #     v_buffer.kv_cache_iter.increment()
            #     return
            mma_qk[1]()
            self.online_softmax_step_1[0]()
            self.online_softmax_update_output()
            scheduling_hints_qk[1]()

            barrier()

            _ = k_buffer.load_from_dram[1]()
            v_buffer.load_from_shared(0)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[vmcnt=4]()

            barrier()

            set_priority[1]()

            comptime if break_mask:
                self.apply_mask[1]()
            mma_pv[0]()
            self.online_softmax_step_0[1, mask = not break_mask]()
            scheduling_hints_pv[2]()
            set_priority[0]()
            barrier()

            _ = v_buffer.load_from_dram[0]()
            k_buffer.load_from_shared(0)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[vmcnt=4]()
            barrier()

            mma_qk[0]()
            self.online_softmax_step_1[1]()
            self.online_softmax_update_output()
            scheduling_hints_qk[3]()
            barrier()

            _ = k_buffer.load_from_dram[0]()
            v_buffer.load_from_shared(1)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[vmcnt=4]()

            barrier()

            set_priority[1]()

            comptime if break_mask:
                self.apply_mask[0]()
            mma_pv[1]()
            self.online_softmax_step_0[0, mask = not break_mask]()

            scheduling_hints_pv[4]()
            set_priority[0]()
            barrier()

            _ = v_buffer.load_from_dram[1]()
            k_buffer.load_from_shared(1)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[vmcnt=4]()
            barrier()

        var iter_end: Int

        comptime is_causal_mask = _type_is_eq[Self.mask_t, CausalMask]()

        comptime if is_causal_mask:
            # for causal mask we can exit early depending on the q_tile_idx
            var num_tiles_causal = ceildiv(
                Int((self.q_tile_idx() + 1) * Self.BM) + self.start_pos,
                Int(Self.BN),
            )
            var num_tiles = ceildiv(self.num_keys, Int(Self.BN))
            num_tiles_causal = min(num_tiles_causal, num_tiles)
            iter_end = max((num_tiles_causal - 1) * Int(Self.BN), 0)
        else:
            iter_end = max(self.num_keys - Int(Self.BN), 0)

        for _ in range(
            UInt32(3 * Self.BN),
            UInt32(iter_end),
            UInt32(Self.BN * 2),
        ):
            var end = min(
                self.kv_start_row + UInt32(2 * Self.BN), UInt32(self.num_keys)
            )
            loop_over_kvcache[Int(Self.BN)](end)

        mma_qk[1]()
        self.online_softmax_step_1[0]()
        self.online_softmax_update_output()
        self.online_softmax_step_0[1]()
        self.online_softmax_step_1[1]()
        barrier()

        _ = k_buffer.load_from_dram[1]()
        v_buffer.load_from_shared(0)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=4]()
        barrier()

        mma_pv[0]()
        self.online_softmax_update_output()
        barrier()

        _ = v_buffer.load_from_dram[0]()
        k_buffer.load_from_shared(0)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=4]()
        barrier()

        mma_qk[0]()
        self.online_softmax_step_0[0]()
        self.online_softmax_step_1[0]()
        barrier()

        v_buffer.load_from_shared(1)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=2]()
        barrier()

        mma_pv[1]()
        self.online_softmax_update_output()
        barrier()

        _ = v_buffer.load_from_dram[1]()
        k_buffer.load_from_shared(1)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=2]()
        barrier()

        mma_qk[1]()
        barrier()

        v_buffer.load_from_shared(0)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=0]()
        barrier()

        mma_pv[0]()
        self.online_softmax_full[1]()
        barrier()

        v_buffer.load_from_shared(1)
        block_sync_lds[lgkmcnt=0]()
        barrier()

        mma_pv[1]()

        barrier()

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )

        if high_warps == 0:
            barrier[
                schedule_barrier_after=False, schedule_barrier_before=False
            ]()
        self.store_output()
