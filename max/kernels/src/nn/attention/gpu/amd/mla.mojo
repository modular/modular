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
from memory import LegacyUnsafePointer as UnsafePointer
from sys import simd_width_of

from gpu import barrier, block_idx, lane_id
from layout import IntTuple, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from nn.mha_operand import MHAOperand
from nn.mha_utils import MHAConfig, get_start_and_end_for_partitions

from utils import IndexList
from utils.numerics import get_accum_type

from .attention import AttentionConfig
from .buffers import KBuffer, VBufferTransposeLoads
from .mha_gfx942 import Attention, MHAAttentionConfig


@fieldwise_init
struct MLAAttentionConfig[token_gen: Bool, config: MHAConfig](AttentionConfig):
    # share shared memory for k and v
    alias shared_kv = True
    # shared memory for the full tile vs BK blocks
    alias full_kv = False
    # pad the depth for v smem
    alias depth_padded = True
    # double buffer
    alias double_buffer = False

    @staticmethod
    @always_inline
    fn q_head_idx() -> UInt:
        return block_idx.y if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_head_idx()

    @staticmethod
    @always_inline
    fn q_tile_idx() -> UInt:
        return Self.q_head_idx() if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_tile_idx()

    @staticmethod
    @always_inline
    fn kv_head_idx() -> UInt:
        return 0 if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].kv_head_idx()

    @staticmethod
    @always_inline
    fn get_mma_shape() -> IndexList[3]:
        return MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].get_mma_shape()

    @staticmethod
    @always_inline
    fn get_q_offset[q_depth: UInt]() -> UInt32:
        return (
            q_depth
            * (
                block_idx.y
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            ) if not Self.token_gen else q_depth
            * Self.q_tile_idx()
            * Self.config.block_m()
        )

    @staticmethod
    @always_inline
    fn get_output_offset[output_depth: UInt]() -> UInt32:
        return (
            output_depth
            * (
                block_idx.y
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            ) if not Self.token_gen else output_depth
            * Self.q_tile_idx()
            * Self.config.block_m()
        )


__extension Attention:
    @always_inline
    fn mla_prefill[
        k_rope_t: MHAOperand, //,
        # cache_num_heads: Int,
        # cache_depth: Int,
    ](mut self, k_rope: k_rope_t):
        alias cache_num_heads = 1
        alias cache_depth = 576
        constrained[Self.BN == Self.depth, "BN must be equal to depth"]()
        alias simd_width = simd_width_of[Self.q_type]()

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

            alias k_rope_gmem_layout = Layout(
                IntTuple(Int(Self.BN), Int(cache_depth)),
                IntTuple(Int(cache_num_heads * cache_depth), 1),
            )

            var k_rope_runtime_layout = RuntimeLayout[k_rope_gmem_layout](
                {Int(kv_tile_num_rows), Int(cache_depth)},
                {Int(cache_num_heads * cache_depth), 1},
            )

            alias cache_group = self.num_heads // UInt(cache_num_heads)
            alias rope_depth = q_depth - Int(Self.depth)

            var k_rope_tile = LayoutTensor[
                k_rope_t.dtype,
                k_rope_gmem_layout,
                MutAnyOrigin,
                masked=True,
            ](
                k_rope.block_paged_ptr[Int(Self.BN)](
                    self.get_batch_idx(),
                    kv_tile_start_row + self.cache_start_pos,
                    Int(Self.kv_head_idx() // UInt(cache_group)),
                    cache_depth - rope_depth,
                ),
                k_rope_runtime_layout,
            )

            var k_rope_buffer = KBuffer[
                tensor_core_mma = Self.get_tensor_core_mma_qk(),
                swizzle=None,
                BN = Int(Self.BN),
                WN = Int(Self.WN),
                BK = Int(Self.BK),
                depth = Int(Self.depth),
                num_threads = Int(Self.num_threads),
                num_stages=2,
            ](
                k_rope_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_rope_tile.dtype](),
            )

            @parameter
            @always_inline
            fn prefetch_function1():
                k_rope_buffer.load_from_dram()

            self.mma_qk[
                prefetch_function=prefetch_function1,
                beg_iter=0,
                num_iters = Int(Self.depth // Self.BK),
            ](k_buffer)

            @parameter
            @always_inline
            fn prefetch_function2():
                v_buffer.load_from_dram()

            self.mma_qk[
                prefetch_function=prefetch_function2,
                beg_iter = Int(Self.depth // Self.BK),
                num_iters = rope_depth // Int(Self.BK),
                prefetched_b_tile=True,
            ](k_rope_buffer)

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
    fn mla_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        qk_max_ptr: UnsafePointer[Scalar[get_accum_type[Self.q_type]()]],
        num_partitions: Int,
    ):
        self.mha_decoding(exp_sum_ptr, qk_max_ptr, num_partitions)
