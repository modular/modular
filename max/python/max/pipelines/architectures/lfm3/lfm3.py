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
"""The LFM3 hybrid decoder.

The module tree mirrors the checkpoint's (``model.layers.{i}``), so weights
load under their own names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, as_subgraph
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.transformer import ReturnLogits
from max.nn.norm import RMSNorm
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from .layers import Lfm3Attention, Lfm3ShortConv, PartialRotaryEmbedding
from .model_config import CONV, FULL_ATTENTION, STATE_CACHE_KEY, Lfm3Config


class Lfm3MLP(Module[[Tensor], Tensor]):
    """``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the SwiGLU MLP."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3DecoderLayer(Module[..., Tensor]):
    """A pre-norm residual block around one operator and the MLP.

    Each branch's output has an optional norm.
    """

    def __init__(
        self,
        operator: Module[..., Tensor],
        layer_type: str,
        *,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float,
        use_sandwich_norm: bool = False,
    ) -> None:
        if layer_type == CONV:
            self.conv = operator
        else:
            self.self_attn = operator
        self.feed_forward = Lfm3MLP(hidden_size, intermediate_size)
        self.operator_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.use_sandwich_norm = use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.post_ffn_norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self, h: Tensor, *operator_args: Tensor | PagedCacheValues
    ) -> Tensor:
        """Runs the operator and the MLP, each as a residual branch."""
        operator = getattr(self, "conv", None) or self.self_attn
        out = operator(self.operator_norm(h), *operator_args)
        if self.use_sandwich_norm:
            out = self.post_operator_norm(out)
        h = h + out
        out = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            out = self.post_ffn_norm(out)
        return h + out


class Lfm3TextModel(Module[..., Tensor]):
    """The embedding, the layer stack and the final norm."""

    def __init__(self, config: Lfm3Config) -> None:
        self.layer_types = config.layer_types
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        ropes = {
            layer_type: PartialRotaryEmbedding(
                head_dim=config.head_dim,
                theta=theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                partial_rotary_factor=config.partial_rotary_factor[layer_type],
            )
            for layer_type, theta in config.rope_theta.items()
        }
        layers = []
        for layer_type in self.layer_types:
            operator: Lfm3Attention | Lfm3ShortConv
            if layer_type == CONV:
                operator = Lfm3ShortConv(
                    config.hidden_size, config.conv_kernel_size
                )
            else:
                operator = Lfm3Attention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    kv_params=config.kv_params.child(
                        layer_type, MHAKVCacheParams
                    ),
                    rope=ropes[layer_type],
                    norm_eps=config.norm_eps,
                    sliding_window=(
                        None
                        if layer_type == FULL_ATTENTION
                        else config.sliding_window
                    ),
                )
            layers.append(
                Lfm3DecoderLayer(
                    operator,
                    layer_type,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    norm_eps=config.norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                )
            )
        self.layers = ModuleList(layers)
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        h: Tensor,
        kv: dict[str, PagedCacheValues],
        conv_pool: Tensor,
        conv_rows: Tensor,
        input_row_offsets: Tensor,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        """Returns the pre-norm hidden states of the embedded tokens ``h``.

        ``segment_ids`` sets the full attention layers' mask (see
        :class:`Lfm3Attention`); the other layers stay causal.
        """
        query_start_loc = F.cast(input_row_offsets, DType.int32)
        # Jenga zeroes a request's state rows before its first forward, so
        # every request resumes from and updates its rows.
        has_initial_state = F.full(
            [conv_rows.shape[1]], True, dtype=DType.bool, device=h.device
        )
        # The blocks of each layer type share one subgraph, and each passes
        # its own index into its type's cache or state table.
        for i, (layer_type, layer) in enumerate(
            zip(self.layer_types, self.layers, strict=True)
        ):
            index = self.layer_types[:i].count(layer_type)
            call: Callable[..., Tensor] = as_subgraph(layer, name=layer_type)
            if layer_type == CONV:
                h = call(
                    h,
                    conv_pool,
                    conv_rows,
                    F.constant(index, DType.int64, device=CPU()),
                    query_start_loc,
                    has_initial_state,
                )
                continue
            segments = (
                [segment_ids]
                if segment_ids is not None and layer_type == FULL_ATTENTION
                else []
            )
            h = call(
                h,
                kv[layer_type],
                F.constant(index, DType.uint32, device=CPU()),
                input_row_offsets,
                *segments,
            )
        return h


class Lfm3(Module[..., ModuleV3Outputs]):
    """LFM3 for causal language modeling, with tied embeddings."""

    returns_hidden_states: ClassVar[bool] = False
    """Whether :meth:`decode` also returns each sequence's normed last hidden
    state."""

    def __init__(self, config: Lfm3Config) -> None:
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.model = Lfm3TextModel(config)

    def _logits(self, h: Tensor) -> Tensor:
        h = self.model.embedding_norm(h)
        return F.cast(h @ self.model.embed_tokens.weight.T, DType.float32)

    def unpack_kv(
        self, kv_inputs: tuple[Tensor, ...], mapping: DeviceMapping
    ) -> tuple[dict[str, PagedCacheValues], Tensor, Tensor]:
        """Splits the flat cache inputs.

        Returns the attention caches and the conv state pool and rows.
        """
        kv_tree = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in kv_inputs)
        )
        kv = {
            key: PagedCacheValues.from_upstream(
                tree.leaves(kv_tree[key], leaf=KVCacheInputsPerDevice),
                mapping,
            )
            for key in self.kv_params.children
            if key != STATE_CACHE_KEY
        }
        (state,) = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        (conv,) = state.leaves
        return (
            kv,
            Tensor.from_shard_values([conv.pool]),
            Tensor.from_shard_values([conv.live_row_ids]),
        )

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the decoder over the tokens and returns the logits."""
        del return_n_logits
        return self.decode(
            self.model.embed_tokens(tokens), input_row_offsets, kv_inputs
        )

    def decode(
        self,
        embeds: Tensor,
        input_row_offsets: Tensor,
        kv_inputs: tuple[Tensor, ...],
    ) -> ModuleV3Outputs:
        """Runs the decoder over token embeddings and returns the logits."""
        kv, conv_pool, conv_rows = self.unpack_kv(kv_inputs, embeds.mapping)
        h = self.model(embeds, kv, conv_pool, conv_rows, input_row_offsets)
        last = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        hidden_states = (
            self.model.embedding_norm(last)
            if self.returns_hidden_states
            else None
        )
        if self.return_logits == ReturnLogits.ALL:
            logits = self._logits(h)
            return ModuleV3Outputs(
                next_token_logits=F.gather(
                    logits, input_row_offsets[1:] - 1, axis=0
                ),
                logits=logits,
                logit_offsets=input_row_offsets,
                hidden_states=hidden_states,
            )
        return ModuleV3Outputs(
            next_token_logits=self._logits(last), hidden_states=hidden_states
        )
