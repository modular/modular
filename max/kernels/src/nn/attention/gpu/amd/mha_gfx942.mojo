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
from math import ceildiv
from memory import LegacyUnsafePointer as UnsafePointer
from sys.info import _cdna_4_or_newer
from sys import env_get_bool

from gpu import barrier, block_idx, lane_id
from layout import LayoutTensor
from layout.swizzle import Swizzle
from nn.mha_utils import MHAConfig, get_start_and_end_for_partitions

from utils import IndexList
from utils.numerics import get_accum_type

from .attention import Attention, AttentionConfig
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


@fieldwise_init
struct MHAAttentionConfig[token_gen: Bool, config: MHAConfig, group: Int](
    AttentionConfig
):
    comptime USE_EXPERIMENTAL_CDNA4_MHA_KERNEL = _cdna_4_or_newer() and env_get_bool[
        "USE_EXPERIMENTAL_CDNA4_MHA_KERNEL", False
    ]() and not Self.token_gen

    # share shared memory for k and v
    comptime shared_kv = False if Self.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL else True
    # shared memory for the full tile vs BK blocks
    comptime full_kv = True if Self.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL else False
    # pad the depth for v smem
    comptime depth_padded = False if Self.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL else True
    # double shared memory for k and v
    comptime double_buffer = True if Self.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL else False

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        @parameter
        if Self.token_gen:
            comptime mma_shape = Self.get_mma_shape()
            var group_idx = lane_id() % UInt(mma_shape[0])
            return block_idx.y * UInt(Self.group) + UInt(group_idx)
        else:
            return block_idx.y

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return block_idx.x if not Self.token_gen else 0

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return block_idx.y if Self.token_gen else block_idx.y // UInt(
            Self.group
        )

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        var mma_shape = (
            IndexList[3](32, 32, 16) if (
                (_cdna_4_or_newer() and Self.config.depth != 64)
                # will deal with 64 later
                or Self.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL
            ) else IndexList[3](32, 32, 8)
        ) if not Self.token_gen else IndexList[3](16, 16, 16)
        return mma_shape

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        return q_depth * (
            (
                Self.kv_head_idx()
                * UInt(Self.group) if Self.token_gen else block_idx.y
            )
            + Self.config.num_heads * Self.q_tile_idx() * Self.config.block_m()
        )

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        return output_depth * (
            (
                Self.kv_head_idx()
                * UInt(Self.group) if Self.token_gen else block_idx.y
            )
            + Self.config.num_heads * Self.q_tile_idx() * Self.config.block_m()
        )


__extension Attention:
    @always_inline
    fn mha_prefill(
        mut self,
    ):
        constrained[Self.BK == 32, "BK must be 32"]()

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Int(Self.BN)](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    Self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Int(Self.BN)](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    Self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            var num_b_rows = Int(kv_tile_num_rows)

            var k_buffer = KBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_qk(),
                swizzle=None,
                BN = Int(Self.BN),
                WN = Int(Self.WN),
                BK = Int(Self.BK),
                depth = Int(Self.depth),
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )

            var v_buffer = VBufferTransposeLoads[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                BN = Int(Self.BN),
                BK = Int(Self.BK),
                depth = Int(Self.depth),
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
            ](v_tile, self.smem_manager.get_v_ptr[v_tile.dtype]())

            @parameter
            @always_inline
            fn prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.mask_apply(
                kv_tile_start_row,
                kv_tile_num_rows,
                not_last_iter,
            )
            # don't know why we need this barrier but i get random failures without it
            barrier()
            self.online_softmax()
            barrier()

            self.mma_pv(v_buffer)

        for i in range(UInt32(0), UInt32(self.num_keys), UInt32(Self.BN)):
            var end = min(i + Self.BN, self.num_keys)
            loop_over_kvcache[Int(Self.BN)](i, end, end != self.num_keys)

        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)

        self.store_output()

    @always_inline
    fn mha_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        qk_max_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        num_partitions: Int,
    ):
        constrained[Self.BK == 32, "BK must be 32"]()

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Int(Self.BN)](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Int(Self.BN)](
                    self.get_batch_idx(),
                    kv_tile_start_row,
                    self.kv_head_idx(),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            comptime swizzle = Swizzle(2, 0, 2)

            var num_b_rows = OptionalReg[Int](
                kv_tile_num_rows
            ) if not not_last_iter else None

            var k_buffer = KBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_qk(),
                swizzle=swizzle,
                BN = Int(Self.BN),
                WN = Int(Self.WN),
                BK = Int(Self.BK),
                depth = Int(Self.depth),
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
                token_gen = Self.token_gen,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )
            var v_tile_slice = v_tile.slice[:, : Int(Self.output_depth)]()
            var v_buffer = VBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                swizzle=None,
                BN = Int(Self.BN),
                WN = Int(Self.WN),
                BK = Int(Self.BK),
                depth = Self.output_depth,
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
                token_gen = Self.token_gen,
            ](
                v_tile_slice,
                num_b_rows,
                self.smem_manager.get_v_ptr[v_tile.dtype](),
            )

            @parameter
            @always_inline
            fn prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.mask_apply(
                kv_tile_start_row,
                kv_tile_num_rows,
                not_last_iter,
            )

            # Not sure why we need this barrier here, but the code hangs without it
            barrier()

            self.online_softmax()

            # warp scratch and p_smem are using the same smem space
            barrier()

            self.copy_fragment_to_smem()

            barrier()

            self.mma_pv(v_buffer)
            # ensure that smem for v is not required anymore
            barrier()

        start, end = get_start_and_end_for_partitions[Int(Self.BN)](
            self.num_keys, num_partitions, Int(block_idx.x)
        )

        for i in range(start, end, Self.BN):
            var end_ = min(i + Int(Self.BN), end)
            loop_over_kvcache[Int(Self.BN)](i, end_, end_ != end)

        # Apply softmax denominator.
        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
