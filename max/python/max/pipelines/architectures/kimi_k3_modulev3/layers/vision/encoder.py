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
"""Kimi K3 vision blocks with RMSNorm before attention and MLP."""

from __future__ import annotations

from max.nn import Module, ModuleList, as_subgraph
from max.tensor import Tensor

from ...vision_config import KimiK3VisionConfig
from .attention import Attention
from .mlp import MLP2
from .norm import VisionRMSNorm
from .rotary_embedding import Rope2DPosEmbRepeated


class EncoderBlock(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Runs a pre-normalized attention block and feed-forward block."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.norm0 = VisionRMSNorm(
            config.vt_hidden_size, eps=config.tower_norm_eps
        )
        self.norm1 = VisionRMSNorm(
            config.vt_hidden_size, eps=config.tower_norm_eps
        )
        self.attn = Attention(
            config.vt_num_attention_heads,
            config.vt_hidden_size,
            config.qkv_hidden_size,
        )
        self.mlp = MLP2(config.vt_hidden_size, config.vt_intermediate_size)

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        rope_freqs_cis: Tensor,
    ) -> Tensor:
        """Runs both residual branches over packed image patches."""
        x = x + self.attn(
            self.norm0(x), input_row_offsets, max_seq_len, rope_freqs_cis
        )
        return x + self.mlp(self.norm1(x))


class Encoder(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Runs repeated vision blocks and final RMS normalization."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.rope_2d = Rope2DPosEmbRepeated(
            dim=config.qkv_hidden_size // config.vt_num_attention_heads,
            max_height=config.rope_max_height,
            max_width=config.rope_max_width,
            theta_base=config.rope_theta,
        )
        self.blocks = ModuleList(
            [EncoderBlock(config) for _ in range(config.vt_num_hidden_layers)]
        )
        self.norm = VisionRMSNorm(
            config.vt_hidden_size, eps=config.tower_norm_eps
        )

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Encodes each image before temporal pooling and spatial merging."""
        rope = self.rope_2d(position_ids)
        for block in self.blocks:
            x = as_subgraph(block, name="kimi_k3_vision_block")(
                x, input_row_offsets, max_seq_len, rope
            )
        return self.norm(x)
