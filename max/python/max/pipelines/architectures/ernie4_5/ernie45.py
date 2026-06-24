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

"""Implements the ERNIE-4.5 model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits

from .layers.attention import ERNIE45Attention
from .layers.rope import build_ernie45_freqs_cis
from .layers.transformer_block import ERNIE45TransformerBlock
from .model_config import ERNIE45Config


class Ernie45Rope:
    """Lightweight RoPE shim that builds precomputed freqs_cis.

    This object mirrors the small interface expected by the attention
    layers while deferring construction of the full freqs_cis tensor to
    the helper builder function.
    """

    def __init__(
        self, head_dim: int, max_seq_len: int, rope_theta: float
    ) -> None:
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._rope_theta = rope_theta
        self.interleaved = True

    @property
    def freqs_cis(self) -> TensorValue:
        """Return the precomputed ``freqs_cis`` tensor used for RoPE.

        The returned layout matches ERNIE-4.5's interleaved GPT-J style:
        ``[cos0,sin0,cos1,sin1,...]`` per position.
        """
        return build_ernie45_freqs_cis(
            head_dim=self._head_dim,
            max_seq_len=self._max_seq_len,
            rope_theta=self._rope_theta,
        )


class ERNIE45TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """ERNIE-4.5 decoder-only transformer.

    0.3B: 18 layers, hidden=1024, GQA 16Q/2KV, head_dim=128, RMSNorm, SwiGLU.

    Key differences from Llama:
      - GPT-J style interleaved RoPE (swap adjacent pairs, not halves)
      - freqs_cis layout: [cos0,sin0,cos1,sin1,...] (interleaved pairs)
      - rope_dim = head_dim * n_heads (not hidden_size) since head_dim != hidden_size//n_heads
    """

    def __init__(self, config: ERNIE45Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rms_norm_eps is None:
            raise ValueError("rms_norm_eps cannot be None for ERNIE-4.5.")

        create_norm: Callable[..., Module[[Tensor], Tensor]] = (
            functools.partial(
                RMSNorm, config.hidden_size, eps=config.rms_norm_eps
            )
        )

        head_dim = ERNIE45Config.get_head_dim_from_config(config)  # 128

        # Build ERNIE-4.5's custom freqs_cis (GPT-J interleaved layout)
        rope = Ernie45Rope(
            head_dim=head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size, out_dim=config.vocab_size, bias=False
            )

        layers = []
        for i in range(config.num_hidden_layers):
            attention = ERNIE45Attention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=i,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                clip_qkv=config.clip_qkv,
            )
            mlp = MLP(
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
            )
            layers.append(
                ERNIE45TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits cast to float32.

        Uses tied embeddings when ``tie_word_embeddings`` is enabled,
        otherwise applies the separate ``lm_head`` projection.

        Args:
            h: Hidden-state tensor of shape ``(seq_len, hidden_size)``.

        Returns:
            Float32 logit tensor of shape ``(seq_len, vocab_size)``.
        """

        if self.tie_word_embeddings:
            logits = F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        else:
            assert self.lm_head is not None
            logits = F.cast(self.lm_head(h), DType.float32)
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling
        return logits

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        """Run the ERNIE-4.5 decoder stack and return logits/hidden states.

        Args:
            tokens: Flat token IDs, shape ``(total_seq_len,)``.
            kv_collection: Paged KV cache for all layers.
            return_n_logits: Controls variable-logit return mode.
            input_row_offsets: Ragged-batch row delimiters,
                shape ``(batch_size + 1,)``.

        Returns:
            Tuple whose layout depends on ``return_logits`` /
            ``return_hidden_states``:
                ``(last_logits,)``
                ``(last_logits, logits, offsets)``
                ``(last_logits, hidden_states)``
                ``(last_logits, logits, offsets, hidden_states)``
        """

        h = self.embed_tokens(tokens)
        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
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
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)
        if self.return_hidden_states == ReturnHiddenStates.LAST:
            ret_val += (last_h,)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            ret_val += (self.norm(h),)
        return ret_val


class ERNIE45(Module[..., tuple[Tensor, ...]]):
    """Top-level ERNIE-4.5 model."""

    def __init__(
        self, config: ERNIE45Config, kv_params: KVCacheParamInterface
    ) -> None:
        super().__init__()
        self.language_model = ERNIE45TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        """Unpack KV cache tensors and delegate to ``ERNIE45TextModel``.

        Args:
            tokens: Flat int64 token IDs.
            return_n_logits: Single-element int64 logit-count control tensor.
            input_row_offsets: uint32 ragged-batch row delimiters.
            *variadic_args: Flattened paged KV cache tensors (all layers,
                all devices) produced by ``KVCacheParamInterface.get_symbolic_inputs``.

        Returns:
            Output tuple forwarded from ``ERNIE45TextModel.forward``.
        """

        kv_inputs = iter(x._graph_value for x in variadic_args)
        unflattened = self.kv_params.get_symbolic_inputs().unflatten(kv_inputs)
        assert isinstance(unflattened, KVCacheInputs), (
            f"Expected KVCacheInputs, got {type(unflattened)}"
        )
        kv_collections = unflattened.inputs
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
