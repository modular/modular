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
"""Implements the Qwen2 decoder using the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice, KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from .layers.attention import Qwen2Attention
from .layers.transformer_block import Qwen2TransformerBlock
from .model_config import Qwen2Config


class Qwen2TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], DecoderOutputs]
):
    """The Qwen2 language model: embeddings, decoder blocks and LM head."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        assert isinstance(config.kv_params, KVCacheParams)
        self.mesh = config.mesh
        self.dtype = config.dtype
        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.mesh.devices[0],
            head_dim=config.head_dim,
            interleaved=config.interleaved_rope_weights,
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size
        )
        self.layers = ModuleList(
            [
                Qwen2TransformerBlock(
                    attention=Qwen2Attention(
                        rope=self.rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        head_dim=config.head_dim,
                        kv_params=config.kv_params,
                        layer_idx=i,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                    ),
                    input_layernorm=RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                    post_attention_layernorm=RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False
            )
        )
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        if self.lm_head is None:
            logits = h @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(h)
        # Both heads shard the vocabulary across the TP mesh.
        logits = F.allgather(logits, tensor_axis=-1)
        return F.cast(logits.local_shards[0], DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> DecoderOutputs:
        """Runs the decoder over the ragged batch."""
        tokens = tokens.to(self.mesh)
        input_row_offsets = input_row_offsets.to(self.mesh)
        self.rope.freqs_cis = self.rope.freqs_cis.cast(self.dtype).to(self.mesh)

        h = self.embed_tokens(tokens)
        for layer in self.layers:
            h = layer(h, kv_collection, input_row_offsets)

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))

        logits: Tensor | None = None
        offsets: Tensor | None = None
        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = F.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=CPU(),
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = F.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=CPU(),
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        return DecoderOutputs(
            next_token_logits=last_logits, logits=logits, logit_offsets=offsets
        )


class Qwen2(Module[[DecoderInputs], DecoderOutputs]):
    """The Qwen2 decoder, registered for the ``DECODER`` role."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.language_model = Qwen2TextModel(config)
        self.config = config

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Wraps the per-device KV inputs, then runs the language model."""
        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
        per_device = tree.leaves(
            self.config.kv_params.unflatten_kv_inputs(kv_inputs),
            leaf=KVCacheInputsPerDevice,
        )
        # Every device holds its own heads' cache, but tokens arrive on the
        # first device only, so the mapping comes from the mesh.
        kv_collection = PagedCacheValues.from_upstream(
            per_device, DeviceMapping.replicated(self.config.mesh)
        )
        return self.language_model(
            inputs.tokens,
            kv_collection,
            inputs.return_n_logits,
            inputs.input_row_offsets,
        )
