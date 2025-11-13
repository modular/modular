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
from itertools import product
from math import ceildiv, recip
from math.constants import log2e
from memory import LegacyUnsafePointer as UnsafePointer
from sys import align_of, simd_width_of, size_of, llvm_intrinsic
from sys.intrinsics import readfirstlane
from sys.info import _cdna_4_or_newer
from algorithm.functional import unswitch
from gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    # barrier,
)
from gpu import warp_id as get_warp_id
from gpu.intrinsics import ds_read_tr16_b64
from gpu.memory import CacheOperation
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
    s_waitcnt_barrier,
)
from memory.pointer import AddressSpace as BaseAddressSpace
from layout import IntTuple, Layout, LayoutTensor
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import blocked_product
from layout._utils import make_amd_buffer_resource, idx2crd
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
    load_b_tr,
)
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
from .mha_gfx942 import Attention
from .utils import SharedMemoryManager

# Note: this is a experimental implementation of MHA for gfx950.


@always_inline
fn block_sync_lds[
    *,
    lgkmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS (local data share) with waitcnt barrier.
    """
    s_waitcnt_barrier[lgkmcnt=lgkmcnt]()


@always_inline
fn block_sync_lds_direct_load[
    *,
    vmcnt: UInt32 = 0,
]():
    """
    Synchronize LDS for direct load with waitcnt barrier.
    """
    s_waitcnt_barrier[vmcnt=vmcnt]()


struct KVCacheIterator[
    cache_t: MHAOperand, tile_size: Int, kv_num_heads: Int, depth: Int
]:
    alias kv_gmem_layout = Layout(
        IntTuple(Int(tile_size), Int(depth)),
        IntTuple(Int(kv_num_heads * depth), 1),
    )
    var cache: cache_t
    var end: Int
    var tile_start_row: Int
    var batch_idx: Int
    var kv_head_idx: Int

    @always_inline
    fn __init__(
        out self, cache: cache_t, batch_idx: Int, kv_head_idx: Int, end: Int
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
            cache_t.dtype,
            Self.kv_gmem_layout,
            MutAnyOrigin,
            masked=True,
        ],
    ):
        var kv_tile_num_rows = min(
            Int(tile_size), Int(self.end - self.tile_start_row)
        )
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = type_of(result.runtime_layout)(
            type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(Self.depth)
            ),
            type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * Self.depth), 1
            ),
        )
        var out = type_of(result)(
            self.cache.block_paged_ptr[tile_size](
                self.batch_idx, self.tile_start_row, self.kv_head_idx, 0
            ),
            kv_runtime_layout,
        )
        self.tile_start_row += tile_size
        return out

    @always_inline
    fn increment(mut self):
        self.tile_start_row += tile_size


@always_inline
fn copy_dram_to_sram_lds[
    swizzle: OptionalReg[Swizzle] = OptionalReg[Swizzle](),
](dst: LayoutTensor, src: LayoutTensor,) -> Int:
    alias thread_layout = Layout.row_major(16, 4)
    var worker_idx = lane_id()

    var bc = make_amd_buffer_resource(src)

    alias M = src.shape[0]()
    alias N = src.shape[1]()
    var num_loads = 0
    # we use 16x4 thread layout to load 16x32 tile from dram to sram
    # but we need to load 32x16 tiles from sram for mma, so swizzle need to be applied to
    # the whole 32x32 tile.
    alias BM = 32
    alias BN = 32
    alias BM_SUB = thread_layout.shape[0].value()

    @parameter
    for n_tile, m_tile, m_sub_tile in product(
        range(N // BN), range(M // BM), range(BM // BM_SUB)
    ):
        var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        alias dst_layout = dst_partitions.layout
        # dst need to be contiguous
        constrained[dst_layout.stride[1].value() == 1, String(dst_layout)]()
        constrained[dst_layout.stride[0].value() == 32, String(dst_layout)]()
        var worker_idx_with_offset = worker_idx + UInt(m_sub_tile * WARP_SIZE)
        var src_dist = src_partitions.vectorize[
            1, simd_width_of[src.dtype]()
        ]().distribute[thread_layout](
            (
                UInt(
                    swizzle.value()(Int(worker_idx_with_offset))
                ) if swizzle else worker_idx_with_offset
            )
            % UInt(WARP_SIZE)
        )
        alias dtype = src.dtype
        var src_offset = (Int(src_dist.ptr) - Int(src.ptr)) // size_of[dtype]()

        alias src_load_offset = src_dist.layout(0)
        var ptr = dst_partitions.ptr
        var dst_ptr = ptr.address_space_cast[AddressSpace.SHARED]()
        bc.load_to_lds[width = simd_width_of[src.dtype]()](
            Int32(src_offset + src_load_offset),
            dst_ptr,
            scalar_offset=0,
        )
        num_loads += 1
    return num_loads


@always_inline
fn load_b_[
    mma_shape: IndexList[3], swizzle: OptionalReg[Swizzle], k_tile_idx: Int
](src: LayoutTensor) -> SIMD[src.dtype, simd_width_of[src.dtype]()]:
    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    constrained[src.shape[0]() == MMA_M]()
    alias simd_width = simd_width_of[src.dtype]()
    var tile = src.tile[MMA_M, MMA_K](0, k_tile_idx)
    alias thread_layout = Layout.col_major(32, 2) if mma_shape[
        0
    ] == 32 else Layout.col_major(16, 4)
    var dist = tile.vectorize[1, simd_width]().distribute[thread_layout,](
        lane_id()
    )
    var offset = dist.distance(src.ptr)

    @parameter
    if swizzle:
        offset = swizzle.value()(offset // simd_width) * simd_width
    var value = src.ptr.offset(offset).load[width=simd_width]()
    return value


@always_inline
fn load_b[
    mma_shape: IndexList[3], swizzle: OptionalReg[Swizzle]
](
    src: LayoutTensor,
    out res: LayoutTensor[
        src.dtype,
        Layout.row_major(src.layout.size() // (WARP_SIZE * 8), 8),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
):
    var output = type_of(res).stack_allocation()
    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias M = src.shape[0]() // MMA_M
    alias N = src.shape[1]() // MMA_K
    var output_vectorized = output.vectorize[1, 8]()

    @parameter
    for i, j in product(range(M), range(N)):
        var out_reg = load_b_[mma_shape, swizzle, j](
            src.tile[MMA_M, src.shape[1]()](i, 0)
        )
        output_vectorized[i + j * M, 0] = rebind[
            type_of(output_vectorized[i + j * M, 0])
        ](out_reg)

    return output


struct KVBuffer[
    kv_t: MHAOperand, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    swizzle: OptionalReg[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
]:
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WN if transpose else depth, Self.MMA_N)
    alias num_k_mmas2 = ceildiv(BK, Int(Self.MMA_K * k_group_size))
    alias simd_width = simd_width_of[kv_t.dtype]()

    # Shared memory layout
    # Layout construction for standard memory access:
    # - base_layout: Layout.row_major(BN, simd_width) -> BN×simd_width tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
    #
    # Resulting shape: BN×(simd_width × num_repeats) = BN×BK tensor
    # Where BK = simd_width × num_repeats, typically simd_width=8, num_repeats=BK/8
    #
    # This creates num_repeats blocks of BN×simd_width arranged horizontally:
    # Within each simd_width-column block, elements are consecutive (stride 1)
    # Between blocks: stride = BN × simd_width
    #
    # ASCII diagram for BN=128, simd_width=8, BK=32 (showing first 2 of 4 blocks):
    # ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    # │        Block 0 (128×8)                     │        Block 1 (128×8)                     │     ... 2 more blocks           │
    # ├────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
    # │   0    1    2    3    4    5    6    7     │ 1024 1025 1026 1027 1028 1029 1030 1031    │ (Block 2: 2048-3071)            │
    # │   8    9   10   11   12   13   14   15     │ 1032 1033 1034 1035 1036 1037 1038 1039    │ (Block 3: 3072-4095)            │
    # │  16   17   18   19   20   21   22   23     │ 1040 1041 1042 1043 1044 1045 1046 1047    │                                 │
    # │  24   25   26   27   28   29   30   31     │ 1048 1049 1050 1051 1052 1053 1054 1055    │                                 │
    # │ ...                                        │  ...                                       │                                 │
    # │1016 1017 1018 1019 1020 1021 1022 1023     │ 2040 2041 2042 2043 2044 2045 2046 2047    │                                 │
    # └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = BN × simd_width = 128 × 8 = 1024

    alias num_repeats = depth // BK
    # this shoul be 2, num_repeats x 32, BK but it gives error in tiling
    alias tiler_layout = Layout.row_major(1, Self.num_repeats)
    alias base_layout = Layout.row_major(BN, BK)
    alias smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)
    # alias smem_layout = Layout.row_major(BN, depth)

    # alias thread_layout = Layout.row_major(num_threads // 16, 16)

    alias MMATileType = LayoutTensor[
        kv_t.dtype,
        Layout.row_major(Self.num_mmas * Self.num_k_mmas2, Self.simd_width),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    alias wtile_dim0 = WN
    alias wtile_dim1 = BK

    alias SharedIterType = LayoutTensorIter[
        kv_t.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    alias SharedTileType = Self.SharedIterType.LayoutTensorType
    alias SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var kv_cache_iter: KVCacheIterator[kv_t, BN, kv_num_heads, depth]
    var buffer_idx: Int

    @always_inline
    fn __init__(
        out self,
        k_cache: kv_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[kv_t.dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
        end: UInt,
    ):
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, Int(batch_idx), Int(head_idx), Int(end)
        )
        self.buffer_idx = 0

    @always_inline
    fn load_from_dram(
        mut self,
    ) -> Int:
        var global_tile = self.kv_cache_iter.next_unsafe()

        var smem_tile = self.smem_iter.next_unsafe(self.buffer_idx)[]

        var num_loads = 0

        @parameter
        if depth == 64:
            var smem_warp_tile = smem_tile.tile[32, BK](
                Int(get_warp_id()) // 2, Int(get_warp_id()) % 2
            )
            var gmem_warp_tile = global_tile.tile[32, BK](
                Int(get_warp_id()) // 2, Int(get_warp_id()) % 2
            )
            # load from dram to sram directly
            num_loads = copy_dram_to_sram_lds[swizzle=swizzle,](
                smem_warp_tile,
                gmem_warp_tile,
            )
        else:
            alias num_warps = num_threads // WARP_SIZE

            @parameter
            for depth_tile in range(depth // 128):
                var smem_warp_tile = smem_tile.tile[BN, BK](
                    0, Int(get_warp_id()) + Int(num_warps * depth_tile)
                )
                var gmem_warp_tile = global_tile.tile[BN, BK](
                    0, Int(get_warp_id()) + Int(num_warps * depth_tile)
                )
                # load from dram to sram directly
                num_loads += copy_dram_to_sram_lds[swizzle=swizzle,](
                    smem_warp_tile,
                    gmem_warp_tile,
                )
        self.buffer_idx = self.buffer_idx ^ 1
        return num_loads

    @always_inline
    fn get_mma_tile[
        k_mma_tile_idx: Int
    ](self) -> Self.MMATileType.SplitElementType[Self.num_k_mmas2]:
        return self.mma_tile.split[self.num_k_mmas2]()[k_mma_tile_idx]

    @always_inline
    fn copy_to_shared(
        self,
    ):
        ...

    @always_inline
    fn load_from_shared(self, buffer: UInt, bk_tile: UInt):
        @parameter
        if transpose:
            alias num_warps_n = BN // WN
            var warp_col = get_warp_id() % UInt(num_warps_n)
            var smem_tile = self.smem_iter.next_unsafe(buffer)[].tile[BN, BK](
                0, Int(bk_tile)
            )

            var wtile_coord0 = Int(warp_col)
            var wtile_coord1 = 0
            var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
                wtile_coord0, wtile_coord1
            )
            var load_b_tile = load_b[Self.mma_shape, swizzle=swizzle](warp_tile)

            self.mma_tile.vectorize[1, Self.simd_width]().copy_from(
                load_b_tile.vectorize[1, Self.simd_width]()
            )

        else:
            alias MMA_M = mma_shape[0]
            alias MMA_K = mma_shape[2]

            @parameter
            for k in range(BK // MMA_K):
                var smem_tile = (
                    self.smem_iter.next_unsafe(buffer)[]
                    .tile[BK, depth](Int(bk_tile), 0)
                    .tile[MMA_K, depth](k, 0)
                )
                var frags = (
                    type_of(self.mma_tile.split[Self.num_k_mmas2]()[k])
                    .stack_allocation()
                    .vectorize[1, Self.simd_width]()
                )

                @parameter
                for i in range(depth // MMA_M):
                    alias tile_layout = type_of(
                        smem_tile.tile[MMA_K, MMA_M](0, i)
                    ).layout
                    # TODO: KERN-2173, the offset calculation is a workaround
                    # a bug in tile, remove this once the bug is fixed
                    alias tiles_per_bk = BK // MMA_M
                    alias stride = self.base_layout.size()
                    alias offset = (
                        MMA_M * (i % tiles_per_bk)
                        + (i // tiles_per_bk) * stride
                    )
                    var tile = LayoutTensor[
                        smem_tile.dtype,
                        tile_layout,
                        MutAnyOrigin,
                        address_space = smem_tile.address_space,
                    ](smem_tile.ptr.offset(offset))
                    frags[i, 0] = rebind[frags.element_type](
                        load_b_tr[Self.mma_shape](tile)
                    )
                self.mma_tile.split[Self.num_k_mmas2]()[k].vectorize[
                    1, Self.simd_width
                ]().copy_from(frags)


__extension Attention:
    fn mha_prefill_experimental(mut self):
        constrained[Self.BK == 32, "BK must be 32"]()

        alias num_threads = config.num_threads()
        var k_buffer = KVBuffer[
            mma_shape = Self.mma_shape,
            k_group_size = Self.k_group_size,
            swizzle = Swizzle(4, 0, 4) if Self.mma_shape[0]
            == 32 else OptionalReg[Swizzle](None),
            BN = Int(Self.BN),
            WN = Int(Self.WN),
            BK = Int(Self.BK),
            num_threads = Int(Self.num_threads),
            depth = Int(Self.depth),
            kv_num_heads = Int(Self.num_heads) // Int(Self.group),
            transpose=True,
        ](
            self.k,
            UInt(self.batch_idx),
            self.kv_head_idx(),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            UInt(self.num_keys),
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
            kv_num_heads = Int(Self.num_heads) // Int(Self.group),
            transpose=False,
        ](
            self.v,
            UInt(self.batch_idx),
            self.kv_head_idx(),
            self.smem_manager.get_v_ptr[type_of(self.v).dtype](),
            UInt(self.num_keys),
        )

        alias num_v_loads = 4
        alias num_k_loads = 4
        alias k_lds_loads = 4
        alias v_lds_loads = 8

        alias accum_type = get_accum_type[type_of(self.k).dtype]()

        @always_inline
        @parameter
        fn prologue():
            _ = k_buffer.load_from_dram()
            _ = v_buffer.load_from_dram()
            _ = k_buffer.load_from_dram()
            block_sync_lds_direct_load[vmcnt = num_v_loads + num_k_loads]()
            # barrier()
            k_buffer.load_from_shared(0, 0)

        prologue()
        schedule_barrier()

        var loop_counter = 0

        alias simd_width = simd_width_of[Self.q_type]()

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
            block_sync_lds[lgkmcnt=k_lds_loads]()
            # barrier()
            if self.mask_skip_and_advance(kv_tile_start_row):
                k_buffer.kv_cache_iter.increment()
                v_buffer.kv_cache_iter.increment()
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)
            alias tensor_core_mma = TiledTensorCore[
                accum_type,
                q_type,
                Self.mma_shape,
                group_size = Self.k_group_size,
                transpose_b=True,
            ]()

            _ = v_buffer.load_from_dram()

            self.zero_p_buffer()

            @parameter
            for i in range(Self.depth // Self.BK):

                @parameter
                if i != 0:
                    # i == 0 is either loaded in the prologue or end of last iteration
                    k_buffer.load_from_shared(UInt(loop_counter % 2), i)

                @parameter
                for k_mma in range(Self.num_k_mmas2):
                    var q_mma_tile = self.q_buffer.get_mma_tile[
                        Int(i), Int(k_mma)
                    ]()

                    var k_mma_tile = k_buffer.get_mma_tile[Int(k_mma)]()
                    tensor_core_mma.mma[swap_a_b = Self.swap_a_b](
                        q_mma_tile, k_mma_tile, self.p_reg_buffer.reg_tile
                    )

            block_sync_lds_direct_load[vmcnt = num_k_loads + num_v_loads]()
            # barrier()
            v_buffer.load_from_shared(UInt(loop_counter % 2), 0)

            # @parameter
            # for i in range(12):
            #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
            #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)

            # @parameter
            # for i in range(4):
            #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
            #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 2, 0)

            self.mask_apply(kv_tile_start_row, kv_tile_num_rows, not_last_iter)

            self.online_softmax()

            block_sync_lds[lgkmcnt=v_lds_loads]()
            # barrier()
            # calculate v^T p^T
            _ = k_buffer.load_from_dram()

            @parameter
            for i in range(Self.BN // Self.BK):

                @parameter
                if i != 0:
                    v_buffer.load_from_shared(UInt(loop_counter % 2), i)

                @parameter
                for k_mma in range(v_buffer.num_k_mmas2):
                    tensor_core_mma.mma[swap_a_b = Self.swap_a_b](
                        self.p_reg_buffer.get_mma_tile[Int(i), Int(k_mma)](),
                        v_buffer.get_mma_tile[k_mma](),
                        self.out_reg_buffer.reg_tile,
                    )
            block_sync_lds_direct_load[vmcnt = num_k_loads + num_v_loads]()
            # barrier()
            loop_counter = loop_counter ^ 1
            k_buffer.load_from_shared(UInt(loop_counter % 2), 0)

            # @parameter
            # for i in range(12):
            #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
            #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 2, 0)

            # @parameter
            # for i in range(4):
            #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
            #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)

        for i in range(UInt32(0), UInt32(self.num_keys), UInt32(Self.BN)):
            var end = min(i + Self.BN, self.num_keys)
            loop_over_kvcache[Int(Self.BN)](i, end, end != self.num_keys)

        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)

        self.store_output()
