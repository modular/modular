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

"""Implements the Muse Glimmer text model using the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings

from .layers.attention import MuseGlimmerAttention
from .layers.rotary_embedding import NoPERotaryEmbedding
from .layers.transformer_block import MuseGlimmerTransformerBlock
from .model_config import MuseGlimmerConfig


class MuseGlimmerTextModel(Module[..., tuple[Tensor, ...]]):
    """The Muse Glimmer language model (text-only)."""

    def __init__(self, config: MuseGlimmerConfig, mesh: DeviceMesh) -> None:
        super().__init__()
        text_config = config.text_config
        self.mesh = mesh
        self.dtype = config.dtype
        device = mesh.devices[0]

        self.rope_sliding, self.rope_full = (
            rope_cls(
                dim=text_config.hidden_size,
                n_heads=text_config.num_attention_heads,
                theta=text_config.rope_theta,
                max_seq_len=text_config.max_position_embeddings,
                device=device,
                head_dim=text_config.head_dim,
                interleaved=False,
            )
            for rope_cls in (RotaryEmbedding, NoPERotaryEmbedding)
        )

        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size, dim=text_config.hidden_size
        )
        self.embed_norm = Gemma4RMSNorm(
            text_config.hidden_size,
            eps=text_config.rms_norm_eps,
            with_weight=False,
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.lm_head = ColumnParallelLinear(
            in_dim=text_config.hidden_size,
            out_dim=text_config.vocab_size,
            bias=False,
        )
        self.output_multiplier = text_config.output_multiplier
        self.logit_softcapping = text_config.final_logit_softcapping

        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type] = kv_params_leaf

        layer_type_counts = dict.fromkeys(kv_params_by_type, 0)
        layers = []
        for i, layer_type in enumerate(text_config.layer_types):
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                MuseGlimmerTransformerBlock(
                    attention=MuseGlimmerAttention(
                        rope=(
                            self.rope_sliding
                            if text_config.use_rope[i]
                            else self.rope_full
                        ),
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx_in_cache=layer_idx_in_cache,
                        is_sliding=layer_type == "sliding_attention",
                        qk_norm_eps=text_config.rms_norm_eps,
                        qk_scale_factor=text_config.qk_scale_factor,
                        local_window_size=text_config.sliding_window,
                    ),
                    mlp=MLP(
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        activation_function="silu",
                    ),
                    hidden_size=text_config.hidden_size,
                    rms_norm_eps=text_config.rms_norm_eps,
                    post_norm_eps=text_config.post_norm_eps,
                )
            )
        self.layers = ModuleList(layers)
        self._layer_kv_key = list(text_config.layer_types)
        self.return_logits = text_config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        outputs = self.lm_head(h).cast(DType.float32) * self.output_multiplier
        cap = self.logit_softcapping
        return F.tanh(outputs / cap) * cap

    def prepare_freq_cis(self, mesh: DeviceMesh) -> None:
        for rope in (self.rope_sliding, self.rope_full):
            rope.freqs_cis = rope.freqs_cis.cast(self.dtype).to(mesh)

    def forward(
        self,
        tokens: Tensor,
        kv_by_type: dict[str, PagedCacheValues],
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        tokens = tokens.to(self.mesh)
        input_row_offsets = input_row_offsets.to(self.mesh)
        image_embeddings = image_embeddings.to(self.mesh)
        image_token_indices = image_token_indices.to(self.mesh)
        h = self.embed_norm(self.embed_tokens(tokens))
        self.prepare_freq_cis(self.mesh)

        # Image embeddings arrive already normed, so they are scattered in
        # after the embedding norm. Out-of-bounds indices are skipped, so an
        # empty (text-only or decode) batch is a no-op.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings.cast(h.dtype), image_token_indices
        )

        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

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
                input_row_offsets[1:].unsqueeze(-1) - return_n_logits_range
            )
            last_indices = offsets.reshape((-1,))
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

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)
        return ret_val


class MuseGlimmer(Module[..., tuple[Tensor, ...]]):
    """Top-level wrapper: unflattens the two-leaf KV tree, delegates."""

    def __init__(self, config: MuseGlimmerConfig, mesh: DeviceMesh) -> None:
        super().__init__()
        self.language_model = MuseGlimmerTextModel(config, mesh)
        self.kv_params = config.kv_params
        self.mesh = mesh

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        # Each device owns a whole KV cache, so the shards are replicated
        # over the TP mesh rather than sharded over it.
        kv_mapping = DeviceMapping.replicated(self.mesh)
        kv_by_type = {
            layer_type: PagedCacheValues.from_upstream(inputs, kv_mapping)
            for layer_type, inputs in zip(
                self.kv_params.children,
                self.kv_params.unflatten_basic_kv_tree(kv_inputs),
                strict=True,
            )
        }
        return self.language_model(
            tokens,
            kv_by_type,
            return_n_logits,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
        )
