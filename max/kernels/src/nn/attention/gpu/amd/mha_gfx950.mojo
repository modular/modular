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

from std.math import ceildiv
from std.sys import simd_width_of, llvm_intrinsic, get_defined_bool
from std.sys.intrinsics import readfirstlane, _type_is_eq
from std.gpu import WARP_SIZE
from std.gpu import warp_id_uint as get_warp_id
from std.gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
    s_waitcnt,
)
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
)
from layout.layout import blocked_product
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from std.memory import bitcast
from nn.attention.mha_mask import TileMaskStatus, CausalMask
from nn.attention.mha_operand import MHAOperand

from std.utils import IndexList
from std.utils.numerics import get_accum_type
from .mha_gfx942 import Attention
from .utils import load_b, load_b_tr, copy_dram_to_sram_lds


# Note: this is a experimental implementation of MHA for gfx950.


@always_inline
def set_priority[priority: Int]():
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](Int16(priority))


@always_inline
def scheduling_hints_qk[group: Int]():
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
def scheduling_hints_pv[group: Int]():
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
def barrier[
    *, schedule_barrier_before: Bool = True, schedule_barrier_after: Bool = True
]():
    comptime if schedule_barrier_before:
        schedule_barrier()

    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

    comptime if schedule_barrier_after:
        schedule_barrier()


@always_inline
def block_sync_lds[
    *,
    lgkmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS (local data share) with waitcnt barrier.
    """

    s_waitcnt[lgkmcnt=lgkmcnt]()


@always_inline
def block_sync_lds_direct_load[
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
    def __init__(
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
    def next_unsafe(
        mut self,
        out result: LayoutTensor[
            Self.cache_t.dtype,
            Self.kv_gmem_layout,
            ImmutAnyOrigin,
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
    def increment(mut self):
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

    comptime warp_tile_rows = 32
    comptime num_repeats = Self.depth // Self.BK
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime base_layout = Layout.row_major(Self.BN, Self.BK)

    # Number of VM (vector memory) instructions each warp issues per
    # load_from_dram call.  Used to compute correct vmcnt values for
    # block_sync_lds_direct_load -- the count must reflect how many
    # rocdl.raw.ptr.buffer.load.lds instructions are still in-flight.
    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _total_tiles = (Self.BN // Self.warp_tile_rows) * (
        Self.depth // Self.BK
    )
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    # Each 32×32 warp-tile produces 2 VM instructions inside
    # copy_dram_to_sram_lds (loop: M/32 * N/32 * 32/16 = 2).
    comptime vm_instrs_per_load = UInt32(Self._tiles_per_warp * 2)
    comptime smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)

    comptime MMATileType = LayoutTensor[
        Self.kv_t.dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_mmas2 * Self.num_k_tiles, Self.simd_width
        ),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SharedTileType = LayoutTensor[
        Self.kv_t.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    comptime SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var smem_ptr: UnsafePointer[
        Scalar[Self.kv_t.dtype],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    comptime smem_stage_size = Self.smem_layout.size()

    var kv_cache_iter: KVCacheIterator[
        Self.kv_t, Self.BN, Self.kv_num_heads, Self.depth
    ]

    var lds_base_ptrs: InlineArray[UInt32, 2]

    var warp_id: UInt32

    @always_inline
    def __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        end: UInt,
        warp_id: UInt32,
    ):
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_ptr = shared_ptr

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, Int(batch_idx), Int(head_idx), Int(end)
        )

        self.warp_id = warp_id

        var warp_row, warp_col = divmod(self.warp_id, UInt32(4))

        self.lds_base_ptrs = type_of(self.lds_base_ptrs)(uninitialized=True)

        comptime for i in range(2):
            var smem_tile = Self.SharedTileType(
                self.smem_ptr + i * Self.smem_stage_size
            )
            var smem_warp_tile = smem_tile.tile[Self.warp_tile_rows, Self.BK](
                Int(warp_row), Int(warp_col)
            )
            self.lds_base_ptrs[i] = UInt32(
                readfirstlane(Int32(Int(smem_warp_tile.ptr)))
            )

    @always_inline
    def load_from_dram[buffer_idx: Int](mut self):
        var global_tile = self.kv_cache_iter.next_unsafe()

        var smem_tile = Self.SharedTileType(
            self.smem_ptr + buffer_idx * Self.smem_stage_size
        )

        comptime if Self.depth == 64:
            var smem_warp_tile = smem_tile.tile[Self.warp_tile_rows, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            var gmem_warp_tile = global_tile.tile[Self.warp_tile_rows, Self.BK](
                Int(self.warp_id) // 2, Int(self.warp_id) % 2
            )
            var lds_base = UInt32(readfirstlane(Int32(Int(smem_warp_tile.ptr))))
            # load from dram to sram directly
            copy_dram_to_sram_lds[swizzle=Self.swizzle,](
                smem_warp_tile,
                gmem_warp_tile,
                lds_base,
            )
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE
            # Number of row groups and depth groups (warp_tile_rows×BK tiles)
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime num_col_groups = Self.depth // Self.BK
            # Total tiles to cover: num_row_groups × num_col_groups
            comptime total_tiles = num_row_groups * num_col_groups
            # Each warp handles ceil(total_tiles / num_warps) tiles
            comptime tiles_per_warp = ceildiv(total_tiles, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                # Each warp's tile index
                var warp_tile = UInt32(tile_idx) + self.warp_id
                var warp_row, warp_col = divmod(
                    warp_tile, UInt32(num_col_groups)
                )
                var smem_warp_tile = smem_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                var gmem_warp_tile = global_tile.tile[
                    Self.warp_tile_rows, Self.BK
                ](Int(warp_row), Int(warp_col))
                # Compute LDS base pointer for this specific tile
                var lds_base = UInt32(
                    readfirstlane(Int32(Int(smem_warp_tile.ptr)))
                )
                # load from dram to sram directly
                copy_dram_to_sram_lds[swizzle=Self.swizzle,](
                    smem_warp_tile,
                    gmem_warp_tile,
                    lds_base,
                )

    @always_inline
    def get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> Self.MMATileType.SplitElementType[
        Self.num_k_tiles
    ].SplitElementType[Self.num_k_mmas2]:
        return self.mma_tile.split[Self.num_k_tiles]()[bk_tile_idx].split[
            self.num_k_mmas2
        ]()[k_mma_tile_idx]

    @always_inline
    def copy_to_shared(
        self,
    ):
        ...

    @always_inline
    def load_from_shared(self, buffer: UInt):
        comptime for bk_tile in range(Self.num_k_tiles):
            self.load_from_shared[bk_tile](buffer)

    @always_inline
    def load_from_shared[bk_tile: Int](self, buffer: UInt):
        comptime if Self.transpose:
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = get_warp_id() % UInt(num_warps_n)
            var smem_base = Self.SharedTileType(
                self.smem_ptr + Int(buffer) * Self.smem_stage_size
            )
            var smem_tile = smem_base.tile[Self.BN, Self.BK](0, bk_tile)

            var wtile_coord0 = Int(warp_col)
            var wtile_coord1 = 0
            var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
                wtile_coord0, wtile_coord1
            )
            var load_b_tile = load_b[Self.mma_shape, swizzle=Self.swizzle](
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
                    Self.SharedTileType(
                        self.smem_ptr + Int(buffer) * Self.smem_stage_size
                    )
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
                        address_space=smem_tile.address_space,
                    ](smem_tile.ptr + offset)
                    frags[i, 0] = rebind[frags.element_type](
                        load_b_tr[Self.mma_shape](tile)
                    )
                var mma_tile = self.get_mma_tile[k, bk_tile]()
                mma_tile.vectorize[1, Self.simd_width]().copy_from(frags)


__extension Attention:
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

    @always_inline
    def online_softmax_step_0[stage: Int, mask: Bool = True](mut self):
        comptime if mask:
            self.apply_mask[stage]()
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.calculate_qk_max(score_reg_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_reg_tile)

    @always_inline
    def online_softmax_step_0_fma[stage: Int, mask: Bool = True](mut self):
        """Step 0 with deferred scaling: mask (no scale), max, exp_scaled.

        Avoids pre-scaling all scores by deferring the scale multiply into the
        exp computation. Uses exp_scaled which subtracts the unscaled max first
        (exact for the maximum element), then scales inside exp2.
        score_frag_rowmax remains unscaled after this call — scale_rowmax is
        deferred to step_1_fma (before calculate_correction needs it).
        """
        comptime if mask:
            self.apply_mask[stage, scale=False]()

        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.calculate_qk_max(score_reg_tile, warp_scratch)
        self.softmax.exp_scaled[start=0, stride=2](score_reg_tile, self.scale)

    @always_inline
    def online_softmax_step_1[stage: Int](mut self):
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
    def online_softmax_step_1_fma[stage: Int](mut self):
        """Step 1 with deferred scaling for odd-indexed tiles.

        Processes remaining score tiles with exp_scaled, then scales the
        rowmax before calculate_correction (which compares against the
        previous iteration's scaled rowmax_tensor).
        """
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.exp_scaled[start=1, stride=2](score_reg_tile, self.scale)
        self.softmax.scale_rowmax(self.scale)
        self.softmax.calculate_qk_sum(score_reg_tile, warp_scratch)
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
        var warp_scratch = self.warp_scratch_tensor.tile[
            2 * Int(Self.num_warps_n), Int(Self.WM)
        ](0, 0)
        var score_reg_tile = self.p_reg_buffer.vectorize[stage]()
        self.softmax.calculate_qk_max(score_reg_tile, warp_scratch)
        self.softmax.exp[start=0, stride=2](score_reg_tile)

    @always_inline
    def online_softmax_step_1_prescaled[stage: Int](mut self):
        """Softmax step 1 for pre-scaled Q: no scale needed on scores."""
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
    def online_softmax_update_output(mut self):
        var output_reg_tile = self.out_reg_buffer.vectorize()
        self.softmax.update_output(output_reg_tile)

    @always_inline
    def online_softmax_full[stage: Int](mut self):
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
    def mha_prefill_experimental_old(mut self):
        comptime assert Self.BK == 32, "BK must be 32"
        comptime assert (
            Self.depth == 64 or Self.depth == 128 or Self.depth == 256
        ), "depth must be 64, 128, or 256"

        comptime num_threads = config.num_threads()
        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )
        var high_warps = warp_id // 4
        var k_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None),
            BN=Int(Self.BN),
            WN=Int(Self.WN),
            BK=Int(Self.BK),
            num_threads=Int(Self.num_threads),
            depth=Int(Self.depth),
            kv_num_heads=Int(Self.num_heads) // Self.group,
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
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Int(Self.BN),
            WN=Int(Self.WN),
            BK=Int(Self.BK),
            num_threads=Int(Self.num_threads),
            depth=Int(Self.depth),
            kv_num_heads=Int(Self.num_heads) // Self.group,
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
        def mma_qk[stage: Int]():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]()
            self.zero_p_buffer[stage]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    var q_mma_tile = self.q_buffer.get_mma_tile[
                        Int(i), Int(k_mma)
                    ]()

                    var k_mma_tile = k_buffer.get_mma_tile[Int(k_mma), Int(i)]()
                    tensor_core_mma.mma[swap_a_b=Self.swap_a_b](
                        q_mma_tile,
                        k_mma_tile,
                        self.p_reg_buffer.get_reg_tile[stage](),
                    )

        @always_inline
        @parameter
        def mma_pv[stage: Int]():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]()

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    tensor_core_mma.mma[swap_a_b=Self.swap_a_b](
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
        block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
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
        block_sync_lds_direct_load[
            vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
        ]()
        barrier[schedule_barrier_after=False]()

        comptime break_mask = False

        @always_inline
        @parameter
        def loop_over_kvcache[tile_size: Int](end: UInt32):
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
            block_sync_lds_direct_load[
                vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
            ]()

            barrier()

            set_priority[1]()

            comptime if break_mask:
                self.apply_mask[1]()
            mma_pv[0]()
            self.online_softmax_step_0[1, mask=not break_mask]()
            scheduling_hints_pv[2]()
            set_priority[0]()
            barrier()

            _ = v_buffer.load_from_dram[0]()
            k_buffer.load_from_shared(0)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[
                vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
            ]()
            barrier()

            mma_qk[0]()
            self.online_softmax_step_1[1]()
            self.online_softmax_update_output()
            scheduling_hints_qk[3]()
            barrier()

            _ = k_buffer.load_from_dram[0]()
            v_buffer.load_from_shared(1)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[
                vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
            ]()

            barrier()

            set_priority[1]()

            comptime if break_mask:
                self.apply_mask[0]()
            mma_pv[1]()
            self.online_softmax_step_0[0, mask=not break_mask]()

            scheduling_hints_pv[4]()
            set_priority[0]()
            barrier()

            _ = v_buffer.load_from_dram[1]()
            k_buffer.load_from_shared(1)
            block_sync_lds[lgkmcnt=0]()
            block_sync_lds_direct_load[
                vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
            ]()
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
        block_sync_lds_direct_load[
            vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
        ]()
        barrier()

        mma_pv[0]()
        self.online_softmax_update_output()
        barrier()

        _ = v_buffer.load_from_dram[0]()
        k_buffer.load_from_shared(0)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[
            vmcnt=k_buffer.vm_instrs_per_load + v_buffer.vm_instrs_per_load
        ]()
        barrier()

        mma_qk[0]()
        self.online_softmax_step_0[0]()
        self.online_softmax_step_1[0]()
        barrier()

        v_buffer.load_from_shared(1)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
        barrier()

        mma_pv[1]()
        self.online_softmax_update_output()
        barrier()

        _ = v_buffer.load_from_dram[1]()
        k_buffer.load_from_shared(1)
        block_sync_lds[lgkmcnt=0]()
        block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
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

    @always_inline
    def mha_prefill_gfx950(mut self):
        """Double-buffered gfx950 MHA with V-load-later optimization.

        Uses both LDS slots for K and V: compute from one slot while
        prefetching the next tile into the other slot. V registers are
        loaded just before PV matmul (not during K load) to reduce peak
        VGPR usage by ~21, enabling double-buffer without spills.
        """
        comptime assert Self.BK == 32, "BK must be 32"
        comptime assert (
            Self.depth == 64 or Self.depth == 128 or Self.depth == 256
        ), "depth must be 64, 128, or 256"
        # Pre-scale Q by default for depth<=128 to eliminate per-element
        # scale multiply from the softmax hot loop. Disabled for depth>128
        # to work around LLVM Machine Instruction Scheduler crash (isReg
        # assertion in RewriteMFMAFormStage).
        comptime prescale_q = get_defined_bool[
            "PRESCALE_Q", True
        ]() and Self.depth <= 128

        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )
        var k_buffer = KVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None),
            BN=Int(Self.BN),
            WN=Int(Self.WN),
            BK=Int(Self.BK),
            num_threads=Int(Self.num_threads),
            depth=Int(Self.depth),
            kv_num_heads=Int(Self.num_heads) // Self.group,
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
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Int(Self.BN),
            WN=Int(Self.WN),
            BK=Int(Self.BK),
            num_threads=Int(Self.num_threads),
            depth=Int(Self.depth),
            kv_num_heads=Int(Self.num_heads) // Self.group,
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

        @always_inline
        @parameter
        def mma_qk():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]()
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    var q_mma_tile = self.q_buffer.get_mma_tile[
                        Int(i), Int(k_mma)
                    ]()
                    var k_mma_tile = k_buffer.get_mma_tile[Int(k_mma), Int(i)]()
                    tensor_core_mma.mma[swap_a_b=Self.swap_a_b](
                        q_mma_tile,
                        k_mma_tile,
                        self.p_reg_buffer.get_reg_tile[0](),
                    )

        @always_inline
        @parameter
        def mma_pv():
            comptime tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size=Self.k_group_size,
                transpose_b=True,
            ]()

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    tensor_core_mma.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.get_mma_tile[Int(i), k_mma, 0](),
                        v_buffer.get_mma_tile[k_mma, Int(i)](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Calculate iteration bounds using mask helpers.
        # start_column returns the first non-fully-masked column,
        # aligned down to BN.  last_masked_set_end returns the total
        # number of BN-wide tiles to process (a count, not an index).
        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Int(Self.BM), Int(Self.BN), 1](
            score_row
        )
        var num_tiles = Int(
            self.mask.last_masked_set_end[Int(Self.BM), Int(Self.BN), 1](
                score_row, UInt32(self.num_keys)
            )
        )

        # Advance KV iterators and mask tracking to start_col.
        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        var num_pairs = num_tiles // 2

        # Pre-scale Q registers so QK matmul produces already-scaled scores.
        # This eliminates the per-element scale multiply from the hot loop
        # at the cost of bf16 quantization of (Q * scale * log2e).
        comptime if prescale_q:
            self.scale_q_buffer()

        # Pre-load first tile K+V into LDS slot 0
        _ = k_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        # For CausalMask, start_column/last_masked_set_end already
        # guarantee no FULL_MASK tiles in the iteration range. For other
        # masks (e.g. ChunkedCausalMask), FULL_MASK tiles can appear in
        # the middle, so we need per-tile status checks.
        comptime has_interior_full_mask = not _type_is_eq[
            Self.mask_t, CausalMask
        ]()

        @always_inline
        @parameter
        def process_tile[slot: Int, has_next: Bool]():
            """Process a single KV tile from the given LDS slot.

            Waits for DRAM→LDS to complete, runs QK→softmax→PV. If has_next,
            prefetches the next tile into the other slot before softmax.
            For non-causal masks, checks tile status and skips FULL_MASK.
            """
            comptime next_slot = 1 - slot

            # Wait for K+V loads to this slot.
            # When has_next, the previous iteration's prefetch for the
            # *other* slot's V is still in-flight.
            comptime if has_next:
                block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            else:
                block_sync_lds_direct_load[vmcnt=0]()
            barrier[
                schedule_barrier_before=False, schedule_barrier_after=False
            ]()

            # For masks that can have FULL_MASK in the middle (e.g.
            # ChunkedCausalMask), check status and skip if fully masked.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        _ = k_buffer.load_from_dram[next_slot]()
                        _ = v_buffer.load_from_dram[next_slot]()
                        barrier[
                            schedule_barrier_before=False,
                            schedule_barrier_after=False,
                        ]()
                    return

            k_buffer.load_from_shared(UInt(slot))
            mma_qk()

            # Prefetch next tile before softmax to hide page table
            # s_load latency during softmax VALU.
            comptime if has_next:
                _ = k_buffer.load_from_dram[next_slot]()
                _ = v_buffer.load_from_dram[next_slot]()

            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            elif Self.depth > 128 or Self.mask_t.apply_log2e_after_mask:
                self.online_softmax_step_0[0]()
                self.online_softmax_step_1[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Wait for V loads from current tile before reading.
            # When has_next, the next slot's K+V prefetch (2 loads) is
            # still in-flight.
            comptime if has_next:
                block_sync_lds_direct_load[
                    vmcnt=k_buffer.vm_instrs_per_load
                    + v_buffer.vm_instrs_per_load
                ]()
            # Barrier ensures all waves' V DMA is visible before any
            # wave reads V from LDS (cross-wave coherence).
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()
            v_buffer.load_from_shared(UInt(slot))

            self.online_softmax_update_output()
            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        # Even tile: compute from slot 0, prefetch to slot 1.
        # Odd tile: compute from slot 1, prefetch to slot 0.
        for _ in range(num_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Remainder: last odd tile from slot 0 (already prefetched).
        if num_tiles % 2 != 0:
            process_tile[0, False]()

        # Apply final softmax denominator and store
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()
