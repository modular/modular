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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/mtp.py at main 4df8111afba.
"""Qwen3.5's multi-token-prediction (MTP) draft head.

The checkpoint's ``mtp.*`` tensors are a NextN-shaped head: two pre-norms, a
fusion projection, one full-attention decoder layer and a final norm. It has
no linear attention and so no recurrent state; it needs an ordinary KV slot.

Two layout facts silently produce a low-acceptance draft rather than a crash
if they are reversed:

- ``fc.weight`` is ``[hidden, 2 * hidden]`` and embedding-first: columns
  ``[0:hidden]`` multiply the normed embedding.
- Every ``mtp.*`` norm scales by ``1 + weight``.

The head embeds and projects through the target's embedding table and head,
which the speculative proposer hands it.
"""

from __future__ import annotations

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.sequential import ModuleList
from max.pipelines.architectures.qwen3_5_modulev3.layers.norm import RMSNorm
from max.pipelines.architectures.qwen3_5_modulev3.layers.rotary import (
    rope_freqs_cis,
)
from max.pipelines.architectures.qwen3_5_modulev3.model_config import (
    Qwen3_5Config,
)
from max.pipelines.architectures.qwen3_5_modulev3.qwen3_5 import (
    FULL_ATTENTION,
    Qwen3_5DecoderLayer,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping

__all__ = ["Qwen3_5MTP"]


class Qwen3_5MTP(Module[..., Tensor]):
    """The one-layer NextN draft head that ships in the checkpoint.

    It returns its final normalized hidden state, not logits: the proposer
    projects through the target's head only at the rows it samples.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        """Builds the head on its target's geometry.

        Args:
            config: The target's config, finalized against the head's
                weights, over the head's one-layer KV leaf.
        """
        self.config = config
        hidden = config.hidden_size
        eps = config.rms_norm_eps or 1e-6
        norm_dtype = config.norm_dtype or DType.float32
        self.rotary_dim = int(
            config.attn_kv_params.head_dim * config.partial_rotary_factor
        )
        self.pre_fc_norm_embedding = RMSNorm(hidden, eps=eps, dtype=norm_dtype)
        self.pre_fc_norm_hidden = RMSNorm(hidden, eps=eps, dtype=norm_dtype)
        # BF16 in every checkpoint, the NVFP4 one included.
        with default_dtype(config.compute_dtype):
            self.fc = Linear(2 * hidden, hidden, bias=False)
        self.layers = ModuleList(
            [
                Qwen3_5DecoderLayer(
                    config,
                    0,
                    self.rotary_dim,
                    kind=FULL_ATTENTION,
                    quantized=False,
                )
            ]
        )
        self.norm = RMSNorm(hidden, eps=eps, dtype=norm_dtype)

    def forward(
        self,
        embeds: Tensor,
        hidden_states: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the head over a ragged batch.

        Args:
            embeds: The target's embeddings of the step's tokens, which sit
                one position past the target's, ``[total_seq_len, hidden]``.
            hidden_states: The target's hidden states, aligned with
                ``embeds``.
            kv_collection: The head's own cache.
            input_row_offsets: The ragged row offsets.

        Returns:
            The final normalized hidden states, ``[total_seq_len, hidden]``.
        """
        mapping = DeviceMapping.replicated(embeds.mesh)
        # Embedding first: `fc.weight[:, :hidden]` multiplies the embedding.
        fused = F.concat(
            [
                self.pre_fc_norm_embedding(embeds),
                self.pre_fc_norm_hidden(hidden_states),
            ],
            axis=-1,
        )
        h = self.fc(fused)
        config = self.config
        freqs_cis = rope_freqs_cis(
            config.rope_theta, self.rotary_dim, config.max_seq_len
        ).to(mapping)
        h = self.layers[0](
            h,
            kv_collection,
            F.constant(0, DType.uint32, device=CPU()),
            freqs_cis,
            input_row_offsets,
            None,
        )
        return self.norm(h)
