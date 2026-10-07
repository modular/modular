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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase
from pydantic import Field


class ClipConfig(MAXModelConfigBase):
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class CLIPAttention(Module[..., Tensor]):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP attention module.

        Args:
            config: CLIP configuration for attention dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.v_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        causal_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply multi-head attention.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            causal_attention_mask: Causal attention mask.

        Returns:
            Attention output.
        """
        batch_size, seq_length, embed_dim = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = F.reshape(
            query, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        query = F.transpose(query, 1, 2)

        key = F.reshape(
            key, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        key = F.transpose(key, 1, 2)

        value = F.reshape(
            value, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        value = F.transpose(value, 1, 2)

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask

        attn_weights = F.matmul(query, F.transpose(key, -1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(F.cast(attn_weights, DType.float32), axis=-1)
        attn_weights = F.cast(attn_weights, hidden_states.dtype)

        attn_output = F.matmul(attn_weights, value)
        attn_output = F.transpose(attn_output, 1, 2)
        attn_output = F.reshape(
            attn_output, (batch_size, seq_length, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output
