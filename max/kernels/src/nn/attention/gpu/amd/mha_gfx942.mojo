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

from std.sys.info import _cdna_4_or_newer

from std.gpu import (
    barrier,
    block_idx_uint as block_idx,
    lane_id_uint as lane_id,
)
from layout.swizzle import Swizzle
from nn.attention.mha_utils import MHAConfig, get_start_and_end_for_partitions

from std.utils import IndexList
from std.utils.numerics import get_accum_type

from .attention import Attention, AttentionConfig
from .buffers import (
    KBuffer,
    VBuffer,
    VBufferTransposeLoads,
)


@fieldwise_init
struct MHAAttentionConfig[token_gen: Bool, config: MHAConfig, group: Int](
    AttentionConfig
):
    comptime use_gfx950_mha_kernel = not Self.token_gen and (
        Self.config.depth == 64
        or Self.config.depth == 128
        or Self.config.depth == 256
    )

    # share shared memory for k and v
    comptime shared_kv = False if Self.use_gfx950_mha_kernel else True
    # shared memory for the full tile vs BK blocks
    comptime full_kv = True if Self.use_gfx950_mha_kernel else False
    # pad the depth for v smem
    comptime depth_padded = False if Self.use_gfx950_mha_kernel else True
    # double shared memory for k and v
    comptime double_buffer = True if Self.use_gfx950_mha_kernel else False

    @staticmethod
    @always_inline
    def q_head_idx() -> UInt:
        comptime if Self.token_gen:
            comptime mma_shape = Self.get_mma_shape()
            var group_idx = lane_id() % UInt(mma_shape[0])
            return block_idx.y * UInt(Self.group) + group_idx
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    def q_tile_idx() -> UInt:
        return block_idx.y if not Self.token_gen else 0

    @staticmethod
    @always_inline
    def kv_head_idx() -> UInt:
        # decode and prefill have different launch configs
        return block_idx.y if Self.token_gen else Self.q_head_idx() // UInt(
            Self.group
        )

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        comptime wider_mfma_supported = (
            _cdna_4_or_newer() and Self.config.depth != 64
        )
        var mma_shape = (
            IndexList[3](32, 32, 16) if (
                wider_mfma_supported
                # will deal with 64 later
                or Self.use_gfx950_mha_kernel
            ) else IndexList[3](32, 32, 8)
        ) if not Self.token_gen else (
            IndexList[3](16, 16, 32) if wider_mfma_supported else IndexList[3](
                16, 16, 16
            )
        )
        return mma_shape

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: UInt]() -> UInt32:
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
    def get_output_offset[output_depth: UInt]() -> UInt32:
        return Self.get_q_offset[output_depth]()


__extension Attention:
    @always_inline
    def mha_prefill_gfx942(
        mut self,
    ):
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

            var k_buffer = KBuffer[
                tensor_core_mma=Self.get_tensor_core_mma_qk(),
                swizzle=Swizzle(3, 0, 4),
                BN=Int(Self.BN),
                WN=Int(Self.WN),
                BK=Int(Self.BK),
                depth=Int(Self.depth),
                num_threads=Int(Self.num_threads),
                num_stages=Self.num_stages,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )

            var v_buffer = VBufferTransposeLoads[
                tensor_core_mma=Self.get_tensor_core_mma_pv(),
                BN=Int(Self.BN),
                BK=Int(Self.BK),
                depth=Int(Self.depth),
                num_threads=Int(Self.num_threads),
                num_stages=Self.num_stages,
            ](v_tile, self.smem_manager.get_v_ptr[v_tile.dtype]())

            @parameter
            @always_inline
            def prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

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
            loop_over_kvcache[Int(Self.BN)](
                i, end, end != UInt32(self.num_keys)
            )

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )

        self.store_output()

    @always_inline
    def mha_decoding(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        comptime assert Self.BK == 32, "BK must be 32"

        @always_inline
        @parameter
        def loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
            if self.mask_skip_and_advance(
                UInt32(kv_tile_start_row),
            ):
                return

            var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

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

            comptime swizzle = Swizzle(2, 0, 2)

            var num_b_rows = Optional[Int](
                kv_tile_num_rows
            ) if not not_last_iter else Optional[Int]()

            var k_buffer = KBuffer[
                tensor_core_mma=Self.get_tensor_core_mma_qk(),
                swizzle=swizzle,
                BN=Int(Self.BN),
                WN=Int(Self.WN),
                BK=Int(Self.BK),
                depth=Int(Self.depth),
                num_threads=Int(Self.num_threads),
                num_stages=Self.num_stages,
                token_gen=Self.token_gen,
            ](
                k_tile,
                num_b_rows,
                self.smem_manager.get_k_ptr[k_tile.dtype](),
            )
            var v_tile_slice = v_tile.slice[:, : Self.output_depth]()
            var v_buffer = VBuffer[
                tensor_core_mma=Self.get_tensor_core_mma_pv(),
                swizzle=None,
                BN=Int(Self.BN),
                WN=Int(Self.WN),
                BK=Int(Self.BK),
                depth=Self.output_depth,
                num_threads=Int(Self.num_threads),
                num_stages=Self.num_stages,
                token_gen=Self.token_gen,
            ](
                v_tile_slice,
                num_b_rows,
                self.smem_manager.get_v_ptr[v_tile.dtype](),
            )

            @parameter
            @always_inline
            def prefetch_function():
                v_buffer.load_from_dram()

            self.mma_qk[prefetch_function=prefetch_function](k_buffer)

            self.scale_p_reg()

            self.mask_apply(
                UInt32(kv_tile_start_row),
                UInt32(kv_tile_num_rows),
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

        for i in range(start, end, Int(Self.BN)):
            var end_ = min(i + Int(Self.BN), end)
            loop_over_kvcache[Int(Self.BN)](i, end_, end_ != end)

        # Apply softmax denominator.
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
