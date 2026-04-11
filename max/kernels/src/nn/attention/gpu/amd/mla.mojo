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

from std.sys import simd_width_of
from std.math.uutils import ufloordiv

from std.gpu import barrier, block_idx
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    MixedLayout,
    RuntimeInt,
    TileTensor,
)
from std.builtin.variadics import Variadic
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig

from std.utils import IndexList
from std.utils.numerics import get_accum_type

from .attention import AttentionConfig
from .buffers import KBuffer, VBufferTransposeLoads
from .mha_gfx942 import Attention, MHAAttentionConfig


@fieldwise_init
struct MLAAttentionConfig[token_gen: Bool, config: MHAConfig](AttentionConfig):
    # share shared memory for k and v
    comptime shared_kv = True
    # shared memory for the full tile vs BK blocks
    comptime full_kv = False
    # pad the depth for v smem
    comptime depth_padded = True
    # double buffer
    comptime double_buffer = False

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        return block_idx.y if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_head_idx()

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return Self.q_head_idx() if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].q_tile_idx()

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        return 0 if Self.token_gen else MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].kv_head_idx()

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        return MHAAttentionConfig[
            Self.token_gen, Self.config, 1
        ].get_mma_shape()

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: UInt]() -> UInt32:
        return UInt32(
            Int(q_depth)
            * (
                block_idx.x
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            ) if not Self.token_gen else Int(q_depth)
            * Self.q_tile_idx()
            * Self.config.block_m()
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: UInt]() -> UInt32:
        return Self.get_q_offset[output_depth]()


__extension Attention:
    @always_inline
    def mla_prefill[
        k_rope_t: MHAOperand,
        //,
        # cache_num_heads: Int,
        # cache_depth: Int,
    ](mut self, k_rope: k_rope_t):
        comptime cache_num_heads = 1
        comptime cache_depth = 576
        comptime assert Self.BN == Self.depth, "BN must be equal to depth"
        comptime simd_width = simd_width_of[Self.q_type]()

        comptime assert Self.BK == 32, "BK must be 32"

        @always_inline
        @parameter
        def loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: UInt32, end: UInt32, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                kv_tile_start_row,
            ):
                return

            var kv_tile_num_rows = min(
                UInt32(tile_size), end - kv_tile_start_row
            )

            var k_tile = self.gmem_manager.get_kv_tile(
                self.k.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    kv_tile_start_row,
                    UInt32(Self.kv_head_idx()),
                    0,
                ),
                kv_tile_num_rows,
            )

            var v_tile = self.gmem_manager.get_kv_tile(
                self.v.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    kv_tile_start_row,
                    UInt32(Self.kv_head_idx()),
                    0,
                ),
                kv_tile_num_rows,
            )

            self.zero_p_buffer()

            comptime kv_layout = Self.GlobalMemoryManagerType.KvTileLayout

            var k_buffer = KBuffer[
                kv_tile_layout=kv_layout,
                tensor_core_mma=Self.get_tensor_core_mma_qk(),
                swizzle=None,
                BN=Self.BN,
                WN=Self.WN,
                BK=Self.BK,
                depth=Self.depth,
                num_threads=Self.num_threads,
                num_stages=Self.num_stages,
            ](
                k_tile,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )

            var v_buffer = VBufferTransposeLoads[
                kv_tile_layout=kv_layout,
                tensor_core_mma=Self.get_tensor_core_mma_pv(),
                BN=Self.BN,
                BK=Self.BK,
                depth=Self.depth,
                num_threads=Self.num_threads,
                num_stages=Self.num_stages,
            ](v_tile, self.smem_manager.get_v_ptr[v_tile.dtype]())

            comptime cache_group = self.num_heads // Int(cache_num_heads)
            comptime rope_depth = q_depth - Int(Self.depth)

            # Build k_rope TileTensor with RuntimeInt valid_rows.
            comptime _k_rope_stride0 = Int(cache_num_heads * cache_depth)
            comptime KRopeTileLayout = MixedLayout[
                Variadic.types[
                    T=CoordLike,
                    RuntimeInt[DType.int64],
                    ComptimeInt[Int(Self.depth)],
                ],
                Variadic.types[
                    T=CoordLike,
                    ComptimeInt[_k_rope_stride0],
                    ComptimeInt[1],
                ],
            ]

            var k_rope_tile = TileTensor[
                k_rope_t.dtype, KRopeTileLayout, ImmutAnyOrigin
            ](
                ptr=k_rope.block_paged_ptr[Int(Self.BN)](
                    UInt32(self.get_batch_idx()),
                    kv_tile_start_row + UInt32(self.cache_start_pos),
                    UInt32(ufloordiv(Self.kv_head_idx(), cache_group)),
                    UInt32(cache_depth - rope_depth),
                ),
                layout=KRopeTileLayout(
                    Coord(
                        RuntimeInt[DType.int64](Int64(kv_tile_num_rows)),
                        Idx[Int(Self.depth)](),
                    ),
                    Coord(Idx[_k_rope_stride0](), Idx[1]()),
                ),
            )

            var k_rope_buffer = KBuffer[
                kv_tile_layout=KRopeTileLayout,
                tensor_core_mma=Self.get_tensor_core_mma_qk(),
                swizzle=None,
                BN=Self.BN,
                WN=Self.WN,
                BK=Self.BK,
                depth=Self.depth,
                num_threads=Self.num_threads,
                num_stages=2,
            ](
                k_rope_tile,
                self.smem_manager.get_k_ptr[k_rope_tile.dtype](),
            )

            # WORKAROUND: The k_rope prefetch during k_buffer MMA combined
            # with prefetched_b_tile=True triggers a miscompilation at -O2
            # (the optimization level used by bazel for nn.mojopkg), producing
            # wrong results for seq_len > 128 (multiple Q tiles). The same
            # code is correct at -O3. The LayoutTensor version of this code
            # does not exhibit the issue — only the TileTensor path.
            # Restore the prefetch once the bug is fixed:

            # @parameter
            # @always_inline
            # def prefetch_function1():
            #     k_rope_buffer.load_from_dram()

            # self.mma_qk[
            #     prefetch_function=prefetch_function1,
            #     beg_iter=0,
            #     num_iters=Int(Self.depth // Self.BK),
            # ](k_buffer)

            self.mma_qk[
                beg_iter=0,
                num_iters=Self.depth // Self.BK,
            ](k_buffer)

            @parameter
            @always_inline
            def prefetch_function2():
                v_buffer.load_from_dram()

            self.mma_qk[
                prefetch_function=prefetch_function2,
                beg_iter=Int(Self.depth // Self.BK),
                num_iters=rope_depth // Int(Self.BK),
                # prefetched_b_tile=True,
            ](k_rope_buffer)

            self.scale_p_reg()

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
            var end = min(i + UInt32(Self.BN), UInt32(self.num_keys))
            loop_over_kvcache[Self.BN](i, end, end != UInt32(self.num_keys))

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )

        self.store_output()

    @always_inline
    def mla_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        self.mha_decoding(exp_sum_ptr, qk_max_ptr, num_partitions)
