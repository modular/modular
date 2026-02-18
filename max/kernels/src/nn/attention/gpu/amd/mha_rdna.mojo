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
"""RDNA-specific MHA kernel configurations and entry points.

This module provides attention computation configurations optimized for AMD RDNA
consumer GPUs (Radeon RX 7000/8000 series, gfx11xx/gfx12xx).

Key features:
- Wave32 execution model
- 16x16x16 WMMA shape (only supported shape for RDNA)
- Optimized shared memory management with K/P reuse
"""

from collections import OptionalReg

from gpu import barrier, block_idx, lane_id
from layout import LayoutTensor
from layout.swizzle import Swizzle
from nn.mha_utils import MHAConfig, get_start_and_end_for_partitions

from utils import IndexList
from utils.numerics import get_accum_type

from .attention import AttentionConfig
from .attention_rdna import AttentionRDNA
from .buffers_rdna import (
    KBufferRDNA,
    VBufferRDNA,
    RDNA_MMA_M,
    RDNA_MMA_N,
    RDNA_MMA_K,
    RDNA_WARP_SIZE,
)
from .utils import get_warp_coords


@fieldwise_init
struct MHAAttentionConfigRDNA[token_gen: Bool, config: MHAConfig, group: Int](
    AttentionConfig
):
    """RDNA-specific attention configuration for Wave32 WMMA.

    This config always uses:
    - MMA shape: 16x16x16 (only shape supported by RDNA WMMA)
    - k_group_size: 1
    - shared_kv: False (P reuses K's shared memory)
    - full_kv: False
    - depth_padded: True
    - double_buffer: False
    """

    # RDNA-specific configuration
    # P matrix reuses K's shared memory slot in prefill mode
    comptime shared_kv = False
    # Don't need full kv tile in shared memory
    comptime full_kv = False
    # Pad depth dimension to avoid bank conflicts
    comptime depth_padded = True
    # No double buffering for simpler memory management
    comptime double_buffer = False

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        @parameter
        if Self.token_gen:
            var group_idx = lane_id() % UInt(Self.group)
            return block_idx.y * UInt(Self.group) + UInt(group_idx)
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return block_idx.y if not Self.token_gen else 0

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return block_idx.y if Self.token_gen else Self.q_head_idx() // UInt(
            Self.group
        )

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        """Get MMA shape - always 16x16x16 for RDNA."""
        return IndexList[3](RDNA_MMA_M, RDNA_MMA_N, RDNA_MMA_K)

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        return UInt32(
            q_depth
            * (
                (
                    Self.kv_head_idx()
                    * UInt(Self.group) if Self.token_gen else Self.q_head_idx()
                )
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            )
        )

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        return Self.get_q_offset[output_depth]()


__extension AttentionRDNA:
    @always_inline
    fn mha_prefill_rdna(
        mut self,
    ):
        """RDNA-optimized prefill (context encoding) kernel.

        Uses Wave32 WMMA operations with 16x16x16 shape and RDNA-specific
        buffer management where P matrix reuses K's shared memory.
        """
        comptime assert Self.BK == 32, "BK must be 32 for RDNA"

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(
                UInt32(tile_size), end - kv_tile_start_row
            )

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    kv_tile_start_row,
                    UInt32(Self.kv_head_idx()),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    kv_tile_start_row,
                    UInt32(Self.kv_head_idx()),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            var num_b_rows = Int(kv_tile_num_rows)

            var k_buffer = KBufferRDNA[
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

            var v_buffer = VBufferRDNA[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                BN = Int(Self.BN),
                BK = Int(Self.BK),
                depth = Int(Self.depth),
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
                num_warps_n = Int(Self.num_warps_n),
            ](
                v_tile,
                self.smem_manager.get_v_ptr[v_tile.dtype](),
                total_rows=num_b_rows,
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

            barrier()
            self.online_softmax()
            barrier()

            self.mma_pv(v_buffer)

        for i in range(UInt32(0), UInt32(self.num_keys), UInt32(Self.BN)):
            var end = min(i + UInt32(Self.BN), UInt32(self.num_keys))
            loop_over_kvcache[Int(Self.BN)](
                i, end, end != UInt32(self.num_keys)
            )

        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)

        self.store_output()

    @always_inline
    fn mha_decoding_rdna(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """RDNA-optimized decoding (token generation) kernel.

        Uses Wave32 WMMA operations with 16x16x16 shape.
        """
        comptime assert Self.BK == 32, "BK must be 32 for RDNA"

        @always_inline
        @parameter
        fn loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                UInt32(kv_tile_start_row),
            ):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = self.gmem_manager.get_kv_tensor(
                self.k.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    UInt32(kv_tile_start_row),
                    UInt32(self.kv_head_idx()),
                    0,
                ),
                UInt32(kv_tile_num_rows),
            )

            var v_tile = self.gmem_manager.get_kv_tensor(
                self.v.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    UInt32(kv_tile_start_row),
                    UInt32(self.kv_head_idx()),
                    0,
                ),
                UInt32(kv_tile_num_rows),
            )

            self.zero_p_buffer()

            var num_b_rows = OptionalReg[Int](
                kv_tile_num_rows
            ) if not not_last_iter else None

            var k_buffer = KBufferRDNA[
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

            var v_tile_slice = v_tile.slice[:, : Int(Self.output_depth)]()
            var v_buffer = VBufferRDNA[
                tensor_core_mma = Self.get_tensor_core_mma_pv(),
                BN = Int(Self.BN),
                BK = Int(Self.BK),
                depth = Self.output_depth,
                num_threads = Int(Self.num_threads),
                num_stages = Self.num_stages,
                num_warps_n = Int(Self.num_warps_n),
            ](
                v_tile_slice,
                self.smem_manager.get_v_ptr[v_tile.dtype](),
                total_rows=kv_tile_num_rows,
            )

            @parameter
            @always_inline
            fn prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.mask_apply(
                UInt32(kv_tile_start_row),
                UInt32(kv_tile_num_rows),
                not_last_iter,
            )

            barrier()
            self.online_softmax()
            barrier()

            self.mma_pv(v_buffer)
            barrier()

        start, end = get_start_and_end_for_partitions[Int(Self.BN)](
            self.num_keys, num_partitions, Int(block_idx.x)
        )

        for i in range(start, end, Self.BN):
            var end_ = min(i + Int(Self.BN), end)
            loop_over_kvcache[Int(Self.BN)](i, end_, end_ != end)

        self.out_reg_buffer.apply_softmax_denominator(self.rowsum)
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
