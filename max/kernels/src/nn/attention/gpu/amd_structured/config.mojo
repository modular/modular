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
"""GFX950 attention config.

Supports both prefill (token_gen=False) and decode (token_gen=True).

Matches amd/mha.mojo config target:
  full_kv=True, depth_padded=False for both.
  Prefill:  double_buffer=True.
  Decode:   double_buffer=False, double_buffer_k_only when BN<=64,
            shared_kv only at depth>256 (SMEM budget).
"""

from std.math.uutils import umod, ufloordiv
from std.gpu import block_idx, lane_id
from std.utils import IndexList
from nn.attention.mha_utils import MHAConfig


@fieldwise_init
struct AMDStructuredConfig[
    config: MHAConfig,
    group: Int,
    token_gen: Bool = False,
    mla_mode: Bool = False,
](ImplicitlyCopyable):
    comptime shared_kv = Self.token_gen and Self.config.depth > 256
    comptime full_kv = True
    comptime depth_padded = False
    comptime double_buffer = not Self.token_gen
    comptime double_buffer_k_only = (
        Self.token_gen and Self.config.block_n() <= 64
    )

    @staticmethod
    @always_inline
    def heads_per_tile() -> Int:
        # MLA: tile spans `BM` heads of the single latent kv (block_idx.y is
        # the tile idx). MHA: tile spans `group` heads of one kv head
        # (block_idx.y is the kv-head idx).
        return Self.config.block_m() if Self.mla_mode else Self.group

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        comptime if Self.token_gen:
            comptime mma_shape = Self.get_mma_shape()
            var lane_in_row = umod(lane_id(), mma_shape[0])
            return Int(block_idx.y) * Self.heads_per_tile() + lane_in_row
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        comptime if Self.token_gen:
            # MLA decode tiles queries across block_idx.y; MHA decode keeps
            # all queries of the kv-head in block 0.
            return Int(block_idx.y) if Self.mla_mode else 0
        else:
            return Int(block_idx.y)

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        comptime if Self.token_gen:
            # MLA decode: single latent kv head at index 0.
            return 0 if Self.mla_mode else Int(block_idx.y)
        else:
            return ufloordiv(Self.q_head_idx(), Self.group)

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        comptime if Self.config.dtype.is_float8():
            comptime if Self.token_gen:
                # FP8 decode: 16x16x128 when depth%128==0, else 32x32x64.
                comptime if Self.config.depth % 128 == 0:
                    return IndexList[3](16, 16, 128)
                return IndexList[3](32, 32, 64)
            # FP8 prefill: 32x32x64.
            return IndexList[3](32, 32, 64)
        # BF16 decode: 16x16x32 regardless of depth.  BF16 prefill: 32x32x16.
        comptime if Self.token_gen:
            return IndexList[3](16, 16, 32)
        return IndexList[3](32, 32, 16)

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: Int]() -> UInt32:
        comptime if Self.token_gen and Self.mla_mode:
            # MLA decode: `BM` queries per tile along block_idx.y.
            return UInt32(q_depth * Self.q_tile_idx() * Self.config.block_m())
        return UInt32(
            q_depth
            * (
                (
                    Self.kv_head_idx()
                    * Self.group if Self.token_gen else Self.q_head_idx()
                )
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            )
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: Int]() -> UInt32:
        return Self.get_q_offset[output_depth]()
