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
from gpu.memory import AddressSpace, CacheOperation
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from memory import AddressSpace as BaseAddressSpace
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

# Note: this is a experimental implementation of MHA for gfx950.

# reference for waitcnt_arg and related synchronization utilities:
# https://github.com/ROCm/rocm-libraries/blob/5ba64a9aa8557e465d1a5937609e23f34adb601e/projects/composablekernel/include/ck_tile/core/arch/arch.hpp#L140


struct WaitCountArg:
    """
    Mojo struct to encapsulate waitcnt argument bitfields and helpers.
    """

    # bit numbers (hex) -------------------------> FE'DC'BA98'7'654'3210
    # [V]M [E]XP [L]GKM counters and [U]NUSED ---> VV'UU'LLLL'U'EEE'VVVV

    # Constants
    alias MAX: UInt32 = 0b1100111101111111
    alias MAX_VM_CNT: UInt32 = 0b111111
    alias MAX_EXP_CNT: UInt32 = 0b111
    alias MAX_LGKM_CNT: UInt32 = 0b1111

    @staticmethod
    fn from_vmcnt(cnt: UInt32) -> UInt32:
        debug_assert(cnt >= 0 and (cnt >> 6) == 0, "valid range is [0..63]")
        return WaitCountArg.MAX & ((cnt & 0b1111) | ((cnt & 0b110000) << 10))

    @staticmethod
    fn from_expcnt(cnt: UInt32) -> UInt32:
        debug_assert(cnt >= 0 and (cnt >> 3) == 0, "valid range is [0..7]")
        return WaitCountArg.MAX & (cnt << 4)

    @staticmethod
    fn from_lgkmcnt(cnt: UInt32) -> UInt32:
        debug_assert(cnt >= 0 and (cnt >> 4) == 0, "valid range is [0..15]")
        return WaitCountArg.MAX & (cnt << 8)


@always_inline
fn s_waitcnt(
    vmcnt: UInt32 = WaitCountArg.MAX_VM_CNT,
    expcnt: UInt32 = WaitCountArg.MAX_EXP_CNT,
    lgkmcnt: UInt32 = WaitCountArg.MAX_LGKM_CNT,
):
    """
    Issues an s_waitcnt with the specified counters.
    """
    # Compose the waitcnt argument
    var waitcnt_val = (
        WaitCountArg.from_vmcnt(vmcnt)
        | WaitCountArg.from_expcnt(expcnt)
        | WaitCountArg.from_lgkmcnt(lgkmcnt)
    )
    # Call the intrinsic (assuming a Mojo binding exists)
    llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](waitcnt_val)


@always_inline
fn s_waitcnt_barrier(
    vmcnt: UInt32 = WaitCountArg.MAX_VM_CNT,
    expcnt: UInt32 = WaitCountArg.MAX_EXP_CNT,
    lgkmcnt: UInt32 = WaitCountArg.MAX_LGKM_CNT,
):
    """
    Issues an s_waitcnt followed by a barrier.
    """
    s_waitcnt(vmcnt, expcnt, lgkmcnt)
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


@always_inline
fn block_sync_lds(lgkmcnt: UInt32 = 0):
    """
    Synchronize LDS (local data share) with waitcnt barrier.
    """
    s_waitcnt_barrier(
        WaitCountArg.MAX_VM_CNT, WaitCountArg.MAX_EXP_CNT, lgkmcnt
    )


@always_inline
fn block_sync_lds_direct_load(vmcnt: UInt32 = 0):
    """
    Synchronize LDS for direct load with waitcnt barrier.
    """
    s_waitcnt_barrier(
        vmcnt, WaitCountArg.MAX_EXP_CNT, WaitCountArg.MAX_LGKM_CNT
    )


@always_inline
fn copy_local_to_dram2[
    dst_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    # TODO: use copy_local_to_dram instead. This is a hack for hackathon :|.

    var worker_idx = (
        thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    )
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var offset = (Int(dst.ptr) - Int(dst_base.ptr)) // size_of[dst.dtype]()
    var descriptor = make_amd_buffer_resource(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset
    alias num_stores_per_thread = dst_fragments.layout.size()

    alias M = src.layout.shape[0].value()
    alias N = src.layout.shape[1].value()

    @parameter
    for n in range(N):

        @parameter
        for m in range(M):
            alias src_idx = 4 * n + 16 * m
            alias i = 4 * m + n

            alias dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx += dst_static_idx
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr.offset(src_idx),
                src.runtime_element_layout,
            )

            alias element_stride = dst_fragments.element_layout.stride[
                1
            ].value()

            @parameter
            if element_stride == 1:
                descriptor.store(
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:

                @parameter
                for i in range(dst_fragments.element_layout.size()):
                    alias element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    descriptor.store(
                        Int32(dst_idx + element_offset),
                        src,
                    )


@always_inline
fn convert_f32_to_bf16[dtype: DType](x: SIMD, out res: SIMD[dtype, x.size]):
    # CK uses truncation for f32 to bf16 conversion but it's not accurate,
    # we only use it when benchmarking against CK otherwise in practice
    # we use the accurate conversion.
    alias use_truncation = False

    @parameter
    if use_truncation:
        res = __type_of(res)(from_bits=(x.to_bits() >> 16).cast[DType.uint16]())
    else:
        res = x.cast[dtype]()


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
            MutableAnyOrigin,
            masked=True,
        ],
    ):
        var kv_tile_num_rows = min(
            Int(tile_size), Int(self.end - self.tile_start_row)
        )
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = __type_of(result.runtime_layout)(
            __type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(Self.depth)
            ),
            __type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * Self.depth), 1
            ),
        )
        var out = __type_of(result)(
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
    alias load_width = 8
    alias thread_layout = Layout.row_major(16, 4)
    var worker_idx = lane_id()

    var bc = make_amd_buffer_resource(src)

    alias BN = src.shape[0]()
    var num_loads = 0

    @parameter
    for tile in range(BN // 32):

        @parameter
        for i in range(32 // 16):
            var dst_partitions = dst.tile[32, 32](tile, 0).tile[16, 32](i, 0)
            var src_partitions = src.tile[32, 32](tile, 0).tile[16, 32](i, 0)
            var worker_idx_with_offset = worker_idx + i * WARP_SIZE
            var src_dist = src_partitions.vectorize[1, load_width]().distribute[
                thread_layout
            ](
                (
                    swizzle.value()(worker_idx_with_offset) if swizzle else Int(
                        worker_idx_with_offset
                    )
                )
                % WARP_SIZE
            )
            alias dtype = src.dtype
            var src_offset = (Int(src_dist.ptr) - Int(src.ptr)) // size_of[
                dtype
            ]()

            alias src_load_offset = src_dist.layout(0)
            var ptr = dst_partitions.ptr
            var dst_ptr = ptr.address_space_cast[AddressSpace.SHARED]()
            bc.load_to_lds[width=load_width](
                Int32(src_offset + src_load_offset), dst_ptr, scalar_offset=0
            )
            num_loads += 1
    return num_loads


@always_inline
fn load_b_[
    swizzle: OptionalReg[Swizzle], k_tile_idx: Int
](src: LayoutTensor) -> SIMD[src.dtype, simd_width_of[src.dtype]()]:
    constrained[src.shape[0]() == 32]()
    alias simd_width = simd_width_of[src.dtype]()
    var tile = src.tile[32, 16](0, k_tile_idx)
    var dist = tile.vectorize[1, simd_width]().distribute[
        Layout.col_major(32, 2)
    ](lane_id())
    var offset = dist.distance(src.ptr) // simd_width

    @parameter
    if swizzle:
        offset = swizzle.value()(offset) * simd_width
    var value = src.ptr.offset(offset).load[width=simd_width]()
    return value


@always_inline
fn load_b[
    swizzle: OptionalReg[Swizzle]
](
    src: LayoutTensor,
    out res: LayoutTensor[
        src.dtype,
        Layout.row_major(src.layout.size() // (WARP_SIZE * 8), 8),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
):
    var output = __type_of(res).stack_allocation()

    alias M = src.shape[0]() // 32
    alias N = src.shape[1]() // 16
    constrained[src.shape[1]() == 32, "src.shape[1]() == 32"]()
    var output_vectorized = output.vectorize[1, 8]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            var out_reg = load_b_[swizzle, j](src.tile[32, 32](i, 0))
            output_vectorized[i + j * M, 0] = rebind[
                __type_of(output_vectorized[i + j * M, 0])
            ](out_reg)

    return output


struct KBuffer[
    k_t: MHAOperand, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    swizzle: OptionalReg[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
]:
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WN, Self.MMA_N)
    alias num_k_mmas2 = ceildiv(BK, UInt(Self.MMA_K * k_group_size))
    alias simd_width = simd_width_of[k_t.dtype]()

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
        k_t.dtype,
        Layout.row_major(Self.num_mmas * Self.num_k_mmas2, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var mma_tile: Self.MMATileType

    alias wtile_dim0 = WN
    alias wtile_dim1 = BK

    alias SharedIterType = LayoutTensorIter[
        k_t.dtype,
        Self.smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    alias SharedTileType = Self.SharedIterType.LayoutTensorType
    alias SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var kv_cache_iter: KVCacheIterator[k_t, BN, kv_num_heads, depth]
    var buffer_idx: Int

    @always_inline
    fn __init__(
        out self,
        k_cache: k_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[k_t.dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
        end: UInt,
    ):
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()
        self.mma_tile = __type_of(self.mma_tile).stack_allocation()
        self.smem_iter = __type_of(self.smem_iter)(shared_ptr, 0)

        self.kv_cache_iter = __type_of(self.kv_cache_iter)(
            k_cache, batch_idx, head_idx, end
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
                get_warp_id() // 2, get_warp_id() % 2
            )
            var gmem_warp_tile = global_tile.tile[32, BK](
                get_warp_id() // 2, get_warp_id() % 2
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
                    0, get_warp_id() + num_warps * depth_tile
                )
                var gmem_warp_tile = global_tile.tile[BN, BK](
                    0, get_warp_id() + num_warps * depth_tile
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
    fn load_from_shared[
        accum_type: DType,
        mma_input_type: DType,
        transpose_b: Bool,
    ](self, buffer: UInt, bk_tile: UInt):
        alias num_warps_n = BN // WN
        var warp_col = get_warp_id() % num_warps_n
        var smem_tile = self.smem_iter.next_unsafe(buffer)[].tile[BN, BK](
            0, bk_tile
        )

        var wtile_coord0 = Int(warp_col)
        var wtile_coord1 = 0
        # constrained[False, ]()
        var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )
        # constrained[
        #     False,
        #     String(warp_tile.shape[0](), " ", warp_tile.shape[1]())
        #     + " "
        #     + String(Self.wtile_dim0, " ", Self.wtile_dim1, " ", BN),
        # ]()

        # alias tensor_core_mma = TiledTensorCore[
        #     accum_type,
        #     mma_input_type,
        #     mma_shape,
        #     group_size=k_group_size,
        #     transpose_b=transpose_b,
        # ]()

        # constrained[False, String(warp_tile.shape[0](), warp_tile.shape[1]())]()
        var load_b_tile = load_b[swizzle=swizzle](warp_tile)

        self.mma_tile.vectorize[1, Self.simd_width]().copy_from(
            load_b_tile.vectorize[1, Self.simd_width]()
        )

        # @parameter
        # for k_mma_tile_idx in range(Self.num_k_mmas2):
        #     tensor_core_mma.mma_op.load_b[swizzle = Self.swizzle](
        #         warp_tile,
        #         self.get_mma_tile[k_mma_tile_idx]().vectorize[
        #             1, Self.simd_width
        #         ](),
        #         UInt(k_mma_tile_idx),
        #     )


@always_inline
fn load_4x16(tile: LayoutTensor, var lane_id_16: Int) -> SIMD[tile.dtype, 4]:
    constrained[tile.dtype == DType.bfloat16, "tile.dtype == DType.bfloat16"]()
    constrained[tile.shape[0]() == 4, "tile.shape[0]() == 4"]()
    constrained[tile.shape[1]() == 16, "tile.shape[1]() == 16"]()
    debug_assert(lane_id_16 < 16, "lane_id_16 < 16")
    alias thread_layout = Layout.row_major(4, 4)
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_id_16)
    var offset = dist_result[2]
    var ptr = tile.ptr.address_space_cast[AddressSpace.SHARED]() + Int(offset)
    return ds_read_tr16_b64(ptr)


@always_inline
fn load_8x32(tile: LayoutTensor, var lane_id: Int) -> SIMD[tile.dtype, 4]:
    constrained[tile.dtype == DType.bfloat16, "tile.dtype == DType.bfloat16"]()
    constrained[tile.shape[0]() == 8, "tile.shape[0]() == 8"]()
    constrained[tile.shape[1]() == 32, "tile.shape[1]() == 32"]()
    var lane_id_32 = lane_id % 32
    var shared_b_tile = tile.tile[4, 16](lane_id // 32, lane_id_32 // 16)
    var lane_16 = lane_id % 16
    return load_4x16(shared_b_tile, lane_16)


@always_inline
fn load_16x32(tile: LayoutTensor, var lane_id: Int) -> SIMD[tile.dtype, 8]:
    constrained[tile.dtype == DType.bfloat16, "tile.dtype == DType.bfloat16"]()
    constrained[tile.shape[0]() == 16, "tile.shape[0]() == 16"]()
    constrained[tile.shape[1]() == 32, "tile.shape[1]() == 32"]()
    var part_1 = load_8x32(tile.tile[8, 32](0, 0), lane_id)
    var part_2 = load_8x32(tile.tile[8, 32](1, 0), lane_id)
    return part_1.join(part_2)


struct VBuffer[
    v_t: MHAOperand, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    kv_num_heads: Int,
]:
    alias simd_width = simd_width_of[v_t.dtype]()
    # alias base_layout = Layout.row_major(BN, depth)
    # alias tiler_layout = Layout.row_major(1, 1)
    # alias smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)

    alias base_layout = Layout.row_major(32, 32)
    alias tiler_layout = Layout.row_major(BN // 32, depth // 32)
    alias smem_layout = blocked_product(Self.base_layout, Self.tiler_layout)

    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)

    alias MMATileType = LayoutTensor[
        v_t.dtype,
        Layout.row_major(
            depth // Self.MMA_M * Self.num_k_tiles, Self.simd_width
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var mma_tile: Self.MMATileType

    alias SharedIterType = LayoutTensorIter[
        v_t.dtype,
        Self.smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    alias SharedTileType = Self.SharedIterType.LayoutTensorType

    var kv_cache_iter: KVCacheIterator[v_t, BN, kv_num_heads, depth]
    var buffer_idx: Int

    @always_inline
    fn __init__(
        out self,
        v_cache: v_t,
        batch_idx: UInt,
        head_idx: UInt,
        shared_ptr: UnsafePointer[
            Scalar[v_t.dtype],
            address_space = AddressSpace.SHARED, **_,
        ],
        end: UInt,
    ):
        constrained[depth in (64, 128, 256), "depth must be 64, 128, or 256"]()
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()

        self.mma_tile = __type_of(self.mma_tile).stack_allocation()
        self.smem_iter = __type_of(self.smem_iter)(shared_ptr, 0)
        self.kv_cache_iter = __type_of(self.kv_cache_iter)(
            v_cache, batch_idx, head_idx, end
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
                get_warp_id() // 2, get_warp_id() % 2
            )
            var gmem_warp_tile = global_tile.tile[32, BK](
                get_warp_id() // 2, get_warp_id() % 2
            )
            # load from dram to sram directly

            num_loads = copy_dram_to_sram_lds[swizzle=None,](
                smem_warp_tile,
                gmem_warp_tile,
            )
        else:
            alias num_warps = num_threads // WARP_SIZE

            @parameter
            for depth_tile in range(depth // 128):
                var smem_warp_tile = smem_tile.tile[BN, BK](
                    0, get_warp_id() + num_warps * depth_tile
                )
                var gmem_warp_tile = global_tile.tile[BN, BK](
                    0, get_warp_id() + num_warps * depth_tile
                )
                # load from dram to sram directly

                num_loads += copy_dram_to_sram_lds[swizzle=None,](
                    smem_warp_tile,
                    gmem_warp_tile,
                )

        # var num_loads = copy_dram_to_sram_lds[Self.thread_layout](
        #     smem_tile.vectorize[1, Self.simd_width](),
        #     global_tile,
        # )
        self.buffer_idx = self.buffer_idx ^ 1
        return num_loads

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared(
        self,
    ):
        ...

    @always_inline
    fn load_from_shared(self, buffer: UInt, bk_tile: UInt):
        @parameter
        for k in range(BK // 16):
            var smem_tile = (
                self.smem_iter.next_unsafe(buffer)[]
                .tile[BK, depth](bk_tile, 0)
                .tile[16, depth](k, 0)
            )
            var frags = (
                __type_of(self.mma_tile.split[Self.num_k_tiles]()[k])
                .stack_allocation()
                .vectorize[1, Self.simd_width]()
            )

            @parameter
            for i in range(depth // 32):
                frags[i, 0] = rebind[frags.element_type](
                    load_16x32(smem_tile.tile[16, 32](0, i), lane_id())
                )

            self.mma_tile.split[Self.num_k_tiles]()[k].vectorize[
                1, Self.simd_width
            ]().copy_from(frags)
        # if (
        #     thread_idx.x == 0
        #     and block_idx.y == 0
        #     and block_idx.z == 0
        #     and block_idx.x == 0
        # ):
        #     print(self.mma_tile)


struct QRegisterBuffer[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType, //,
    mma_shape: IndexList[3],
    k_group_size: Int,
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    thread_layout: Layout,
]:
    alias simd_width = simd_width_of[dtype]()
    alias MMA_M = mma_shape[0]
    alias MMA_K = mma_shape[2]
    alias num_mmas = ceildiv(WM, Self.MMA_M)
    alias num_k_tiles = ceildiv(BK, Self.MMA_K * k_group_size)

    alias GlobalTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        masked=masked,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]
    var gmem_tensor: Self.GlobalTensorType

    alias num_tiles = depth // BK
    alias RegisterTileType = LayoutTensor[
        dtype,
        Layout.row_major(
            Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.simd_width
        ),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var mma_tile: Self.RegisterTileType

    alias TiledIteratorType = Self.RegisterTileType.TiledIteratorType[
        Self.num_mmas * Self.num_k_tiles, Self.simd_width, axis=0
    ]

    # TODO: This is expensive, dereferencing q_gmem_warp_iter[] is expensive and
    # using its dim() is also expensive. Need to find a better way to do this.

    @always_inline
    fn __init__(
        out self,
        tensor: LayoutTensor[
            dtype,
            layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            masked=masked,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
        ],
    ):
        constrained[
            mma_shape[2] * k_group_size == 16,
            "mma_shape[2] * k_group_size must be 16",
        ]()
        self.gmem_tensor = tensor
        self.mma_tile = __type_of(self.mma_tile).stack_allocation()

    @always_inline
    fn load_from_dram(mut self):
        alias num_warps_n = BN // WN
        var warp_row = get_warp_id() // num_warps_n
        var bounds = max(
            min(Int32(WM), Int32(self.gmem_tensor.dim[0]() - WM * warp_row))
            * self.gmem_tensor.stride[0](),
            0,
        )
        var gmem_warp_iter = self.gmem_tensor.tiled_iterator[WM, BK, axis=1](
            warp_row, 0
        )
        var mma_tiles = self.mma_tile.split[Self.num_tiles]()

        @parameter
        for i in range(Self.num_tiles):
            var reg_tile = mma_tiles[i]
            copy_dram_to_local[
                src_thread_layout=thread_layout,
                thread_scope = ThreadScope.WARP,
            ](
                reg_tile.vectorize[1, Self.simd_width](),
                gmem_warp_iter,
                Int(readfirstlane(Int32(bounds))),
            )
            gmem_warp_iter._incr()

    @always_inline
    fn get_mma_tile[
        tile_idx: Int, k_idx: Int
    ](self) -> Self.RegisterTileType.SplitElementType[
        Self.num_tiles
    ].SplitElementType[Self.num_k_tiles]:
        return self.mma_tile.split[Self.num_tiles]()[tile_idx].split[
            Self.num_k_tiles
        ]()[k_idx]


struct PRegisterBuffer[
    accum_type: DType,
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    output_frag_size: Int,
]:
    alias RegisterTileType = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, output_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var reg_tile: Self.RegisterTileType

    alias OutputTileType = LayoutTensor[
        dtype,
        Layout.row_major(num_m_mmas, output_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    @always_inline
    fn __init__(out self):
        self.reg_tile = Self.RegisterTileType.stack_allocation().fill(0)

    @always_inline
    fn interleave[tile_idx: Int](self) -> Self.OutputTileType:
        var out = Self.OutputTileType.stack_allocation()

        @parameter
        for j in range(16):
            out[0, j] = convert_f32_to_bf16[dtype](self.reg_tile[tile_idx, j])
        return out


@always_inline
fn _apply_mask[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    MMA_M: Int,
    MMA_N: Int,
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
    var scale_log2e: SIMD[accum_type, 1] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )

    var coords = idx2crd[warp_layout](lane)
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
            var mask_frag_row = mask_warp_row + m_mma * MMA_M
            var mask_frag_col = (
                mask_warp_col
                + n_mma * MMA_N
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
            var score_row_with_start_pos = score_row + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    alias fragment_col = fragment_layout(j)
                    var group_idx = lane_row
                    var q_head_idx = (
                        block_idx.y * group + group_idx
                    ) if token_gen else block_idx.y
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos),
                            Int(score_col + fragment_col),
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


@always_inline
fn apply_softmax_denominator[
    accum_type: DType, //,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
](
    out_reg_tile: LayoutTensor[accum_type, **_],
    rowsum: LayoutTensor[accum_type, **_],
):
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv = recip(rowsum[m_mma, 0])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(fragment_layout.size()):

                @parameter
                if fragment_layout.shape[0].value() > 1:
                    rowsum_inv = recip(rowsum[m_mma, i])
                out_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rebind[
                    out_reg_tile.element_type
                ](rowsum_inv)


struct SharedMemoryManager[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_rowwise_warps: Int,
    token_gen: Bool,
](Defaultable):
    var p_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # p_smem is used for p
    var k_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # v_smem is used for v, and scratch
    var v_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    alias alignment = align_of[SIMD[dtype, simd_width_of[dtype]()]]()
    alias accum_type = get_accum_type[dtype]()
    alias p_smem_size = BM * BN if token_gen else 0
    alias simd_width = simd_width_of[dtype]()
    alias k_smem_size = BN * depth * 2
    alias v_smem_size = BN * depth * 2

    @always_inline
    fn __init__(out self):
        self.p_smem = stack_allocation[
            Self.p_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()
        self.k_smem = stack_allocation[
            Self.k_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()
        self.v_smem = stack_allocation[
            Self.v_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self.alignment,
        ]()

    @always_inline
    fn get_kv_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        address_space = AddressSpace.SHARED,
        # alignment2 = Self.alignment,
    ]:
        return self.k_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_k_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        address_space = AddressSpace.SHARED,
        # alignment2 = Self.alignment,
    ]:
        return self.k_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_v_ptr[
        _dtype: DType
    ](
        self,
    ) -> UnsafePointer[
        Scalar[_dtype],
        address_space = AddressSpace.SHARED,
        # alignment2 = Self.alignment,
    ]:
        return self.v_smem.bitcast[Scalar[_dtype]]()

    @always_inline
    fn get_p_ptr(
        self,
    ) -> UnsafePointer[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        # alignment2 = Self.alignment,
    ]:
        return self.p_smem.bitcast[Scalar[dtype]]()

    @always_inline
    fn get_k_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BN, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        constrained[token_gen, "this function is only used for token_gen"]()
        return __type_of(result)(self.k_smem, BN * depth)

    @always_inline
    fn get_v_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BK, BN),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        constrained[token_gen, "this function is only used for token_gen"]()
        return __type_of(result)(self.v_smem, BN * depth)

    @always_inline
    fn get_p_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        return __type_of(result)(
            self.p_smem,
            BM * BN,
        )

    @always_inline
    fn get_warp_scratch_tensor(
        self,
        out result: LayoutTensor[
            Self.accum_type,
            Layout.row_major(2 * num_rowwise_warps, BM),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        constrained[
            result.layout.size()
            * (size_of[Self.accum_type]() // size_of[dtype]())
            <= Self.k_smem_size,
            "warp_scratch_tile is too large",
        ]()
        var ptr = self.k_smem.bitcast[Scalar[Self.accum_type]]()
        return __type_of(result)(ptr if token_gen else __type_of(ptr)())


struct GlobalMemoryManager[
    dtype: DType,
    BM: UInt32,
    BN: UInt32,
    BK: UInt32,
    depth: UInt32,
    num_heads: UInt32,
    group: UInt32,
    token_gen: Bool,
]:
    alias kv_num_heads = num_heads // group
    # BHSD layout for q and kv cache
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)),
        IntTuple(Int(num_heads * depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(depth))

    alias kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(Self.kv_num_heads * depth), 1),
    )

    var q_offset: UInt32
    var q_runtime_layout: RuntimeLayout[
        Self.q_gmem_layout,
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

        self.q_offset = depth * (
            (kv_head_idx * group if token_gen else block_idx.y)
            + num_heads * q_tile_idx * BM
        )

        self.q_runtime_layout = __type_of(self.q_runtime_layout)(
            RuntimeTuple[
                Self.q_gmem_layout.shape,
                element_type = __type_of(self.q_runtime_layout).element_type,
            ](Int(q_tile_num_rows), Int(depth)),
            RuntimeTuple[
                Self.q_gmem_layout.stride,
                element_type = __type_of(self.q_runtime_layout).linear_idx_type,
            ](Int(num_heads * depth if not token_gen else depth), 1),
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
        return __type_of(result)(
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
            Self.q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return self.get_q_tensor(ptr)

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
            # alignment = ptr.alignment2,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = __type_of(result.runtime_layout)(
            __type_of(result.runtime_layout.shape)(
                Int(kv_tile_num_rows), Int(depth)
            ),
            __type_of(result.runtime_layout.stride)(
                Int(Self.kv_num_heads * depth), 1
            ),
        )

        return __type_of(result)(
            ptr,
            kv_runtime_layout,
        )


@always_inline
fn mha_single_batch_gfx950[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
    sink: Bool = False,
    sink_type: DType = output_type,
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
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
):
    alias token_gen = False
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    # constrained[depth == 128, "depth must be 128"]()
    alias num_heads = config.num_heads
    alias BK = config.block_k()
    alias simd_width = simd_width_of[q_type]()

    alias mma_shape = IndexList[3](32, 32, 16)

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
    alias num_n_mmas_depth = ceildiv(depth, UInt(mma_shape[1]))
    alias num_k_mmas2 = ceildiv(BK, UInt(mma_shape[2] * k_group_size))
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

    var kv_head_idx = block_idx.y // group

    var q_tile_idx = block_idx.x

    var q_head_idx = block_idx.y

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
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
        depth=depth,
        thread_layout=warp_layout,
    ](q_tile)

    q_buffer.load_from_dram()

    alias num_threads = config.num_threads()
    var k_buffer = KBuffer[
        mma_shape=mma_shape,
        k_group_size=k_group_size,
        swizzle = Swizzle(4, 0, 4),
        BN=BN,
        WN=WN,
        BK=BK,
        num_threads=num_threads,
        depth=depth,
        kv_num_heads = num_heads // group,
    ](
        k,
        batch_idx,
        kv_head_idx,
        smem_manager.get_k_ptr[k_t.dtype](),
        num_keys,
    )

    var v_buffer = VBuffer[
        mma_shape=mma_shape,
        k_group_size=k_group_size,
        BN=BN,
        BK=BK,
        depth=depth,
        num_threads=num_threads,
        kv_num_heads = num_heads // group,
    ](v, batch_idx, kv_head_idx, smem_manager.get_v_ptr[v_t.dtype](), num_keys)

    alias num_v_loads = 4
    alias num_k_loads = 4
    alias k_lds_loads = 4
    alias v_lds_loads = 8

    @always_inline
    @parameter
    fn prologue():
        _ = k_buffer.load_from_dram()
        _ = v_buffer.load_from_dram()
        _ = k_buffer.load_from_dram()
        block_sync_lds_direct_load(num_v_loads + num_k_loads)
        # barrier()
        k_buffer.load_from_shared[accum_type, q_type, True](0, 0)

    prologue()
    schedule_barrier()

    var loop_counter = 0

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int
    ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
        block_sync_lds(k_lds_loads)
        # barrier()
        var mask_status = mask.status(
            Index[dtype = DType.uint32](
                Int(q_tile_idx * BM + start_pos),
                Int(kv_tile_start_row),
            ),
            Index[dtype = DType.uint32](Int(BM), Int(BN)),
        )

        if mask_status == TileMaskStatus.FULL_MASK:
            k_buffer.kv_cache_iter.increment()
            v_buffer.kv_cache_iter.increment()
            mask_warp_col += BN
            return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var p_buffer = PRegisterBuffer[
            accum_type,
            q_type,
            num_m_mmas,
            num_n_mmas,
            output_frag_size,
        ]()

        alias tensor_core_mma = TiledTensorCore[
            accum_type,
            q_type,
            mma_shape,
            group_size=k_group_size,
            transpose_b=True,
        ]()

        _ = v_buffer.load_from_dram()

        @parameter
        for i in range(depth // BK):

            @parameter
            if i != 0:
                # i == 0 is either loaded in the prologue or end of last iteration
                k_buffer.load_from_shared[
                    # TODO: I should be able to use tensor_core_mma here
                    # but getting compiler errors
                    accum_type,
                    q_type,
                    True,
                ](loop_counter % 2, i)

            @parameter
            for k_mma in range(num_k_mmas2):
                var q_mma_tile = q_buffer.get_mma_tile[i, k_mma]()

                var k_mma_tile = k_buffer.get_mma_tile[k_mma]()
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    q_mma_tile, k_mma_tile, p_buffer.reg_tile
                )

        block_sync_lds_direct_load(num_k_loads + num_v_loads)
        # barrier()
        v_buffer.load_from_shared(loop_counter % 2, 0)

        # @parameter
        # for i in range(12):
        #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
        #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)

        # @parameter
        # for i in range(4):
        #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
        #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 2, 0)
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
                group=group,
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
            )

        unswitch[_apply_mask_impl](mask_status == TileMaskStatus.PARTIAL_MASK)

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )

        alias reg_layout_by_mma_unit_depth = Layout.row_major(
            num_m_mmas * num_n_mmas_depth, output_frag_size
        )
        # don't know why we need this barrier but i get random failures without it
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
            p_buffer.reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        block_sync_lds(v_lds_loads)
        # barrier()
        # calculate v^T p^T
        _ = k_buffer.load_from_dram()

        @parameter
        for i in range(BN // BK):

            @parameter
            if i != 0:
                v_buffer.load_from_shared(loop_counter % 2, i)

            var p_mma_tile_interleaved = p_buffer.interleave[i]()

            @parameter
            for k_mma_idx in range(v_buffer.num_k_tiles):
                tensor_core_mma.mma[swap_a_b=swap_a_b](
                    p_mma_tile_interleaved.tile[1, simd_width](0, k_mma_idx),
                    v_buffer.mma_tile.tile[depth // mma_shape[0], simd_width](
                        k_mma_idx, 0
                    ),
                    out_reg_tile,
                )
        block_sync_lds_direct_load(num_k_loads + num_v_loads)
        # barrier()
        loop_counter = loop_counter ^ 1
        k_buffer.load_from_shared[accum_type, q_type, True](loop_counter % 2, 0)

        # @parameter
        # for i in range(12):
        #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
        #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 2, 0)

        # @parameter
        # for i in range(4):
        #     schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, 0)
        #     schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)

    for i in range(UInt32(0), UInt32(num_keys), UInt32(BN)):
        var end = min(i + BN, num_keys)
        loop_over_kvcache[BN](i, end, end != num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas_depth,
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
