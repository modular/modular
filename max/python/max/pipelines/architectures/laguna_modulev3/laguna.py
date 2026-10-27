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
"""Implements the Laguna model on ModuleV3.

Attention type, query-head count, MLP type and MLP precision all vary per
layer. Layers that agree on all four share one traced subgraph; each group
gets its own subgraph symbol, and everything else that differs between the
layers of a group -- weights, the cache index, the RoPE table -- is passed to
the subgraph as an operand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from max import tree
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module, as_subgraph
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.rotary_embedding import YarnScalingParams
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.lib import ModuleV3Outputs

from .layers.norm import LagunaRMSNorm
from .layers.rotary_embedding import (
    LagunaRotaryEmbedding,
    LagunaYarnPartialRotaryEmbedding,
)
from .layers.transformer_block import LagunaTransformerBlock, layer_shapes

_ATTENTION_TYPES = ("sliding_attention", "full_attention")


def rotary_dim(rope_params: Mapping[str, Any], head_dim: int) -> int:
    """Returns how many of a head's lanes a RoPE table rotates."""
    return int(head_dim * float(rope_params.get("partial_rotary_factor", 1.0)))


def build_rope(
    config: LagunaConfig,
    rope_params: Mapping[str, Any],
    head_dim: int,
    device: Device,
) -> RotaryEmbedding:
    """Builds one attention type's RoPE table from its HF entry.

    Args:
        config: The Laguna config.
        rope_params: One entry of the HF config's ``rope_parameters``.
        head_dim: Lanes per head, shared by both cache leaves.
        device: The device the table is built on.

    Returns:
        The rotary embedding for the layers of that attention type.
    """
    theta = float(rope_params.get("rope_theta", 10000.0))
    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
    if rope_params.get("rope_type", "default") == "yarn":
        return LagunaYarnPartialRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=theta,
            max_seq_len=config.max_seq_len,
            device=device,
            head_dim=head_dim,
            scaling_params=YarnScalingParams(
                factor=float(rope_params["factor"]),
                beta_fast=float(rope_params.get("beta_fast", 32.0)),
                beta_slow=float(rope_params.get("beta_slow", 1.0)),
                original_max_position_embeddings=int(
                    rope_params["original_max_position_embeddings"]
                ),
                truncate=True,
            ),
            partial_rotary_factor=partial_rotary_factor,
        )
    return LagunaRotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=theta,
        max_seq_len=config.max_seq_len,
        device=device,
        head_dim=head_dim,
        interleaved=False,
        scaling_params=ProportionalScalingParams(
            partial_rotary_factor=partial_rotary_factor
        ),
    )


def kv_params_by_type(
    kv_params: KVCacheParamInterface,
) -> dict[str, KVCacheParams]:
    """Maps each attention type to the cache leaf its layers address."""
    if isinstance(kv_params, MultiKVCacheParams):
        leaves: dict[str, KVCacheParams] = {}
        for key, leaf in kv_params.children.items():
            assert isinstance(leaf, KVCacheParams)
            leaves[key] = leaf
        return leaves
    assert isinstance(kv_params, KVCacheParams)
    return {"full_attention": kv_params}


class LagunaTextModel(Module[..., ModuleV3Outputs]):
    """The Laguna decoder stack, embeddings and LM head.

    Args:
        config: The finalized Laguna config.
        norm_dtype: Storage dtype of the checkpoint's RMSNorm gammas.
    """

    def __init__(self, config: LagunaConfig, norm_dtype: DType) -> None:
        super().__init__()
        schedule = config.layer_schedule
        if schedule is None:
            raise ValueError(
                "LagunaConfig.layer_schedule is unset; it is populated by "
                "LagunaConfig.initialize_from_config."
            )
        device = config.devices[0].to_device()
        cache_leaves = kv_params_by_type(config.kv_params)
        head_dim = cache_leaves["full_attention"].head_dim

        # One RoPE table per attention type. They differ in more than theta:
        # the full-attention table is YaRN-scaled and partial-rotary.
        rope_params: dict[str, Mapping[str, Any]] = {
            "full_attention": schedule.full_rope
        }
        if schedule.sliding_rope is not None:
            rope_params["sliding_attention"] = schedule.sliding_rope
        self._ropes = {
            layer_type: build_rope(config, params, head_dim, device)
            for layer_type, params in rope_params.items()
        }

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = LagunaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps or 1e-6,
            dtype=norm_dtype,
        )
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head: Linear | None = (
            None
            if config.tie_word_embeddings
            else Linear(config.hidden_size, config.vocab_size, bias=False)
        )

        self._shapes = layer_shapes(
            schedule,
            config.quant_config.mlp_quantized_layers
            if config.quant_config is not None
            else (),
        )
        self._layer_types = [shape.attention_type for shape in self._shapes]
        # Each layer addresses its cache leaf by its index within its own
        # attention type, not by its position in the model.
        counts: dict[str, int] = {}
        self._layer_idx_in_cache: list[int] = []
        for layer_type in self._layer_types:
            self._layer_idx_in_cache.append(counts.get(layer_type, 0))
            counts[layer_type] = counts.get(layer_type, 0) + 1

        self.layers = ModuleList(
            [
                LagunaTransformerBlock(
                    config,
                    shape,
                    kv_params=cache_leaves[shape.attention_type],
                    rotary_dim=rotary_dim(
                        rope_params[shape.attention_type], head_dim
                    ),
                    norm_dtype=norm_dtype,
                )
                for shape in self._shapes
            ]
        )
        self.use_subgraphs = config.use_subgraphs

    def forward(
        self,
        tokens: Tensor,
        kv_by_type: dict[str, PagedCacheValues],
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the model and returns the last-token logits.

        Args:
            tokens: Ragged input token IDs.
            kv_by_type: The cache leaf of each attention type present.
            return_n_logits: Number of logits to return (only the last token
                is supported).
            input_row_offsets: Ragged row offsets.

        Returns:
            A one-element tuple of float32 last-token logits.
        """
        del return_n_logits
        h = self.embed_tokens(tokens)
        freqs_cis = {
            layer_type: rope.freqs_cis.cast(h.dtype).to(h.device)
            for layer_type, rope in self._ropes.items()
        }

        for idx, layer in enumerate(self.layers):
            shape = self._shapes[idx]
            call: Callable[..., Tensor] = layer
            if self.use_subgraphs:
                call = as_subgraph(layer, name=shape.subgraph_name)
            h = call(
                F.constant(
                    self._layer_idx_in_cache[idx], DType.uint32, device=CPU()
                ),
                h,
                kv_by_type[shape.attention_type],
                freqs_cis[shape.attention_type],
                input_row_offsets,
            )

        last_token_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        normed = self.norm(last_token_h)
        if self.lm_head is None:
            logits = normed @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(normed)
        return ModuleV3Outputs(next_token_logits=logits.cast(DType.float32))


class Laguna(Module[..., ModuleV3Outputs]):
    """Top-level Laguna wrapper that unflattens the variadic KV cache inputs.

    Args:
        config: The finalized Laguna config.
        kv_params: The KV cache parameters: one flat cache, or the
            sliding/full tree when the model has sliding layers.
        norm_dtype: Storage dtype of the checkpoint's RMSNorm gammas.
    """

    def __init__(
        self,
        config: LagunaConfig,
        kv_params: KVCacheParamInterface,
        norm_dtype: DType,
    ) -> None:
        super().__init__()
        self.language_model = LagunaTextModel(config, norm_dtype)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_by_type: dict[str, PagedCacheValues] = {}
        if isinstance(self.kv_params, MultiKVCacheParams):
            # The tree's declaration order is the unflattening order, and a
            # swap would run every layer against the wrong cache without
            # failing anything.
            children = list(self.kv_params.children)
            if children != list(_ATTENTION_TYPES):
                raise ValueError(f"Unexpected Laguna KV tree order: {children}")
            for layer_type, leaf_inputs in zip(
                _ATTENTION_TYPES,
                self.kv_params.unflatten_basic_kv_tree(kv_inputs),
                strict=True,
            ):
                kv_by_type[layer_type] = PagedCacheValues.from_upstream(
                    leaf_inputs, tokens.mapping
                )
        else:
            per_device = tree.leaves(
                self.kv_params.unflatten_kv_inputs(kv_inputs),
                leaf=KVCacheInputsPerDevice,
            )
            kv_by_type["full_attention"] = PagedCacheValues.from_upstream(
                per_device, tokens.mapping
            )
        return self.language_model(
            tokens, kv_by_type, return_n_logits, input_row_offsets
        )
