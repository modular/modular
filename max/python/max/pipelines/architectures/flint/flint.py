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

"""Implements the Flint model on the ModuleV3 API."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import ops
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import ModuleV3Outputs

from .layers.attention import FlintAttention
from .layers.hybrid_kv_cache import (
    FULL,
    SLIDING,
    hybrid_layer_plan,
    unflatten_hybrid_kv,
)
from .layers.moe import FlintMoE, FlintRouter
from .layers.nope_rope import NopeRotaryEmbedding
from .layers.transformer_block import FlintTransformerBlock
from .model_config import FlintConfig


def is_dense_mlp(config: FlintConfig, layer_idx: int) -> bool:
    """Returns whether a layer uses a dense MLP rather than the MoE.

    ``decoder_sparse_step`` counts from one, so a step of two puts the MoE on
    odd layers.
    """
    return (layer_idx in config.mlp_only_layers) or not (
        config.num_experts > 0
        and (layer_idx + 1) % config.decoder_sparse_step == 0
    )


def _make_mlp(config: FlintConfig, layer_idx: int) -> MLP | FlintMoE:
    is_dense = is_dense_mlp(config, layer_idx)
    if is_dense:
        return MLP(
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            bias=False,
            activation_function=config.hidden_act,
        )
    gate_cls = functools.partial(
        FlintRouter,
        norm_topk_prob=config.norm_topk_prob,
    )
    return FlintMoE(
        hidden_dim=config.hidden_size,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_tok,
        moe_dim=config.moe_intermediate_size,
        gate_cls=gate_cls,
        has_shared_experts=True,
        shared_experts_dim=config.shared_expert_intermediate_size,
    )


class FlintTextModel(
    Module[
        [Tensor, PagedCacheValues, PagedCacheValues, Tensor, Tensor],
        ModuleV3Outputs,
    ],
):
    """The Flint language model.

    Sliding-window and NoPE full-attention layers over a dense-then-MoE
    feed-forward stack, with one KV cache child per attention type.
    """

    def __init__(self, config: FlintConfig) -> None:
        super().__init__()
        self.devices = config.devices
        self.config = config

        # Sliding layers use rope; full attention layers use the
        # identity rotation so both share the fused store path.
        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,  # HF uses rotate_half, not interleaved
        )
        self.nope_rope = NopeRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps or 1e-5)
        self.tie_word_embeddings = config.tie_word_embeddings
        self.vocab_size = config.vocab_size
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        # One paged cache child per attention type, each stacking its
        # own layers.
        plan = hybrid_layer_plan(config.kv_params, config.layer_types)

        create_norm = functools.partial(
            RMSNorm, config.hidden_size, eps=config.rms_norm_eps or 1e-5
        )
        layers = []
        for i, slot in enumerate(plan):
            is_sliding = slot.layer_type == SLIDING
            layers.append(
                FlintTransformerBlock(
                    attention=FlintAttention(
                        rope=(self.rope if is_sliding else self.nope_rope),
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=slot.kv_params,
                        layer_idx=slot.layer_idx_in_cache,
                        is_sliding=is_sliding,
                        sliding_window=config.sliding_window,
                        rms_norm_eps=config.rms_norm_eps or 1e-5,
                        has_bias=config.attention_bias,
                    ),
                    mlp=_make_mlp(config, i),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                )
            )
        self.layers = ModuleList(layers)
        self._layer_kv_key = list(config.layer_types)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        # TRANSPOSED logits GEMM: vocab is the matmul row count, not
        # the gemv N dimension. At fp32 with vocab >= 65536 the gemv
        # path hits a launch ceiling and crashes (gemv.mojo:2339);
        # the transposed form selects the GEMM path.
        w = (
            self.embed_tokens.weight
            if self.tie_word_embeddings
            else self.lm_head.weight
        )
        logits_t = w @ F.transpose(h, 1, 0)
        return F.cast(F.transpose(logits_t, 1, 0), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        sliding_kv: PagedCacheValues,
        full_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> ModuleV3Outputs:
        kv_by_type = {SLIDING: sliding_kv, FULL: full_kv}
        h = self.embed_tokens(tokens)

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None
        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets
        return ModuleV3Outputs(
            next_token_logits=last_logits,
            logits=logits,
            logit_offsets=offsets,
            hidden_states=None,
        )


class Flint(Module[..., ModuleV3Outputs]):
    """Unflattens the KV cache inputs and runs the text model."""

    def __init__(
        self,
        config: FlintConfig,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        # The pipeline hands models its cache tree as the generic interface;
        # Flint's `construct_kv_params` always builds the hybrid tree, so
        # narrow it once here rather than in every user of the tree.
        assert isinstance(kv_params, MultiKVCacheParams), (
            "Flint needs the hybrid (Multi) KV cache tree; got "
            f"{type(kv_params).__name__}"
        )
        self.language_model = FlintTextModel(config)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        # The variadic KV arguments are the tree's leaves, sliding child
        # first.
        sliding_kv, full_kv = unflatten_hybrid_kv(
            self.kv_params, variadic_args, tokens
        )
        return self.language_model(
            tokens, sliding_kv, full_kv, return_n_logits, input_row_offsets
        )
