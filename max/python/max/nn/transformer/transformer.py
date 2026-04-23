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

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import TypeVar

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops

from ..embedding import Embedding
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Layer, LayerList, Module
from ..linear import Linear
from ..rotary_embedding import RotaryEmbedding


def forward_sharded_layers(
    layers: Sequence[Callable[[TensorValue], TensorValue]],
    xs: Sequence[TensorValue],
) -> list[TensorValue]:
    """Forward pass through sharded layers.

    Args:
        layers: Sequence of callable layers that return TensorValue
        xs: Input tensors, one per layer

    Returns:
        List of output tensors from each layer

    Raises:
        AssertionError: If the number of layers and input tensors don't match
    """
    assert len(xs) == len(layers), (
        f"Number of layers ({len(layers)}) must match number of inputs ({len(xs)})"
    )
    return [layer(x) for layer, x in zip(layers, xs, strict=True)]


def extract_hs(
    return_hidden_states: ReturnHiddenStates,
    last_token_hs_distributed: Sequence[TensorValue],
    all_hs_distributed: Sequence[TensorValue],
    normalizer: Sequence[Callable[[TensorValue], TensorValue]],
    eagle3_captured_hs: list[list[TensorValue]] | None = None,
) -> tuple[TensorValue, ...]:
    """Extract hidden states from the model.

    Args:
        return_hidden_states: Which hidden states to return.
        last_token_hs_distributed: Hidden states from the last token
            (identical across devices; the first entry is returned).
        all_hs_distributed: Per-device hidden states from all tokens — one
            entry per device (distinct batch shards in DP, identical
            replicas in TP/single-device).
        normalizer: Per-device normalization functions.
        eagle3_captured_hs: For ``EAGLE3`` mode, a list of captured hidden
            states at specific layer indices.  Each entry is a per-device list
            of tensors. The entries are concatenated along the feature dimension
            to produce a single fused hidden-state tensor per device.

    Returns:
        Empty tuple for ``NONE``; ``(TensorValue,)`` for ``LAST``; for
        ``ALL`` / ``ALL_NORMALIZED`` / ``EAGLE3`` the per-device tensors are
        returned as positional tuple elements (``N`` entries, one per
        device).
    """
    if return_hidden_states == ReturnHiddenStates.LAST:
        # Each entry in last_token_hs_distributed is identical.
        # Just return the first one.
        return (last_token_hs_distributed[0],)
    elif return_hidden_states == ReturnHiddenStates.LAST_PER_DEVICE:
        return tuple(last_token_hs_distributed)
    elif return_hidden_states == ReturnHiddenStates.ALL:
        return tuple(all_hs_distributed)
    elif return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
        norm_hs = forward_sharded_layers(normalizer, all_hs_distributed)
        return tuple(norm_hs)
    elif return_hidden_states == ReturnHiddenStates.EAGLE3:
        assert eagle3_captured_hs is not None and len(eagle3_captured_hs) > 0
        num_devices = len(eagle3_captured_hs[0])
        fused_per_dev = [
            ops.concat(
                [layer_hs[i] for layer_hs in eagle3_captured_hs],
                axis=-1,
            )
            for i in range(num_devices)
        ]
        return tuple(fused_per_dev)
    else:
        return tuple()


class TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: Module,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.residual_multiplier = residual_multiplier

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        residual_multiplier = ops.constant(
            self.residual_multiplier, x.dtype, device=x.device
        )
        attn_out = self.self_attn(
            layer_idx,
            self.input_layernorm(x),
            kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp


class ReturnLogits(str, Enum):
    LAST_TOKEN = "last_token"
    VARIABLE = "variable"
    ALL = "all"


class ReturnHiddenStates(str, Enum):
    NONE = "none"
    LAST = "last"
    # Like LAST but returns the post-allgather per-device tensors rather
    # than collapsing to a single copy. Lets callers consume the data that
    # was already placed on every device by the allgather, avoiding a
    # redundant broadcast.
    LAST_PER_DEVICE = "last_per_device"
    ALL_NORMALIZED = "all_normalized"
    ALL = "all"
    EAGLE3 = "eagle3"


def logits_postprocess(
    h: TensorValue,
    input_row_offsets: TensorValue,
    return_n_logits: TensorValue,
    norm: Callable[[TensorValue], TensorValue],
    lm_head: Callable[[TensorValue], TensorValue],
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    logits_scaling: float = 1.0,
) -> tuple[TensorValue, ...]:
    """Performs common logits postprocessing for single-device models.

    Handles last-token gathering, logits computation, optional logits
    scaling, and hidden states return.

    Args:
        h: The hidden states from the final transformer layer.
        input_row_offsets: The row offsets for ragged batching.
        return_n_logits: The number of logits to return per sequence.
        norm: The normalization function applied before the language model head.
        lm_head: The language model head projection.
        return_logits: Which logits to return.
        return_hidden_states: Which hidden states to return. Defaults to
            ``ReturnHiddenStates.NONE``.
        logits_scaling: A divisor applied to logits after projection. Logits
            are divided by this value before returning. Defaults to ``1.0``
            (no scaling).

    Returns:
        A tuple of ``(last_logits,)``, optionally extended with
        ``(logits, offsets)`` when returning multiple logits, and further
        extended with hidden states when requested. Hidden-state entries
        for ``ALL`` / ``ALL_NORMALIZED`` / ``EAGLE3`` modes are returned as
        a ``list[TensorValue]`` (one entry per device); other entries are
        single ``TensorValue``.
    """
    last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
    last_logits = ops.cast(lm_head(norm(last_h)), DType.float32)
    logits = None
    offsets = None

    if return_logits == ReturnLogits.VARIABLE:
        return_n_logits_range = ops.range(
            return_n_logits[0],
            0,
            -1,
            out_dim="return_n_logits_range",
            device=h.device,
            dtype=DType.int64,
        )
        offsets = (
            ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
        )
        last_indices = ops.reshape(offsets, shape=(-1,))
        last_tokens = ops.gather(h, last_indices, axis=0)
        logits = ops.cast(lm_head(norm(last_tokens)), DType.float32)
        offsets = ops.range(
            0,
            TensorValue(last_indices.shape[0]) + return_n_logits[0],
            return_n_logits[0],
            out_dim="logit_offsets",
            device=h.device,
            dtype=DType.int64,
        )
    elif return_logits == ReturnLogits.ALL:
        logits = ops.cast(lm_head(norm(h)), DType.float32)
        offsets = input_row_offsets

    if logits_scaling != 1.0:
        last_logits = last_logits / logits_scaling
        if logits is not None:
            logits = logits / logits_scaling

    ret_val: tuple[TensorValue, ...] = (last_logits,)
    if offsets is not None:
        assert logits is not None
        ret_val += (logits, offsets)

    ret_val += extract_hs(
        return_hidden_states=return_hidden_states,
        last_token_hs_distributed=[last_h],
        all_hs_distributed=[h],
        normalizer=[norm],
    )

    return ret_val


class LogitsPostprocessMixin:
    """Mixin providing logits postprocessing for single-device models.

    Subclasses must define :attr:`norm`, :attr:`lm_head`, and
    :attr:`return_logits`. The :attr:`return_hidden_states` and
    :attr:`logits_scaling` attributes are optional and have defaults.
    """

    norm: Callable[[TensorValue], TensorValue]
    """The normalization function applied before the language model head."""

    lm_head: Callable[[TensorValue], TensorValue]
    """The language model head projection."""

    return_logits: ReturnLogits
    """Which logits to return."""

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """Which hidden states to return. Defaults to :obj:`ReturnHiddenStates.NONE`."""

    logits_scaling: float = 1.0
    """A divisor applied to logits after projection. Logits are divided by
    this value before returning. Defaults to ``1.0`` (no scaling)."""

    def _postprocess_logits(
        self,
        h: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
    ) -> tuple[TensorValue, ...]:
        return logits_postprocess(
            h,
            input_row_offsets,
            return_n_logits,
            norm=self.norm,
            lm_head=self.lm_head,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.logits_scaling,
        )


Block = TypeVar("Block", bound=Module, covariant=True)


class Transformer(LogitsPostprocessMixin, Module):
    """A transformer model consisting of :class:`TransformerBlock` layers.

    Args:
        dim: The model dimension.
        n_heads: The number of attention heads.
        layers: The list of transformer blocks.
        norm: The normalization layer applied before the language model head.
        output: The language model head projection.
        embedding: The token embedding layer.
        kv_params: The key-value cache parameters.
        rope: The rotary position embedding.
        return_logits: Which logits to return. Defaults to
            ``ReturnLogits.LAST_TOKEN``.
        return_hidden_states: Which hidden states to return. Defaults to
            ``ReturnHiddenStates.NONE``.
        embedding_multiplier: A scalar applied to token embeddings after
            lookup. Defaults to ``1.0`` (no scaling).
        logits_scaling: A divisor applied to logits after projection. Logits
            are divided by this value before returning. Defaults to ``1.0``
            (no scaling).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[Block],
        norm: Layer,
        output: Linear,
        embedding: Embedding,
        kv_params: KVCacheParams,
        rope: RotaryEmbedding,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        embedding_multiplier: float = 1.0,
        logits_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.embedding_multiplier = embedding_multiplier
        self.rope = rope
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
        self.logits_scaling = logits_scaling

    def _process_hidden_states(
        self,
        h: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        return self._postprocess_logits(h, input_row_offsets, return_n_logits)

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        hidden_states: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        return self._process_hidden_states(
            h, kv_collection, return_n_logits, input_row_offsets
        )
