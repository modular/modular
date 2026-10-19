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
"""Layers of the DSpark drafter, which proposes speculative draft tokens.

The drafter reads hidden states captured from selected layers of the target
model, combines them into one context stream, and runs a few small
attention-plus-FFN stages over a block of draft positions. Small heads add a
per-position bias from the previous draft token and estimate how likely each
draft token is to be accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.embedding import Embedding
from max.nn.kernels import (
    KEY_CACHE_INDEX,
    kv_cache_gather_rows_ragged,
    store_k_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from ...deepseekV4.layers.quantization import fp8_qat_quantize
from ...deepseekV4.layers.ragged import arange_to
from ...deepseekV4.layers.rope import apply_rope_tail
from ...deepseekV4.layers.sparse_attention import (
    sparse_attention as gathered_sparse_attention,
)
from ..layer_resolver import LayerInfo
from ..model_config import DeepseekV41Config
from .query import DeepseekV41Query
from .ragged import V41RaggedRows as RaggedRows
from .rope import rope_for_layer

# One E8M0 scale per 32 elements (MXFP8).
_MXFP8_BLOCK_SIZE = 32


def dspark_main_proj_quant_config() -> QuantConfig:
    """The MXFP8 ``QuantConfig`` for ``main_proj`` and the stages' ``wo_b``.

    ``block_size=(1, 32)`` declares ``weight_scale`` as ``[out_dim, in_dim /
    32]`` (one scale per row per 32 columns), not the checkpoint's 32x32 block
    shape, so the scale must be expanded to this shape before loading.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        # Unconditionally quantized; no per-layer sets to enumerate.
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP8,
    )


_LayerState = TypeVar("_LayerState")


def capture_target_taps(
    layer_ids: Iterable[int],
    target_layer_ids: Sequence[int],
    entry: _LayerState,
    run_layer: Callable[[int, _LayerState], _LayerState],
) -> tuple[_LayerState, dict[int, _LayerState]]:
    """Runs ``run_layer`` over ``layer_ids``, capturing selected layer inputs.

    The tap for layer ``idx`` is the state *entering* that layer (the previous
    layer's output): for ``target_layer_ids = [5, 6]`` the taps are the
    outputs of layers 4 and 5. Generic over the state type so the indexing
    can be unit tested without tensors.

    Args:
        layer_ids: The layer ids to run, in order.
        target_layer_ids: The layers whose incoming state is captured.
        entry: The state entering ``layer_ids[0]``.
        run_layer: Runs one layer given its id and incoming state; returns the
            outgoing state.

    Returns:
        The state after the last layer, and a dict from each id of
        ``target_layer_ids`` within ``layer_ids`` to the state entering it.
    """
    targets = set(target_layer_ids)
    aux_hidden_by_layer: dict[int, _LayerState] = {}
    x = entry
    for layer_id in layer_ids:
        if layer_id in targets:
            aux_hidden_by_layer[layer_id] = x
        x = run_layer(layer_id, x)
    return x, aux_hidden_by_layer


def combine_hidden_states(
    aux_hidden_states: Sequence[TensorValue],
    main_proj: Linear,
    main_norm: RMSNorm,
) -> TensorValue:
    """Computes ``main_norm(main_proj(concat(aux_hidden_states)))``.

    Args:
        aux_hidden_states: The captured layer inputs, each ``[T, hidden]``.
        main_proj: Projection from ``hidden * n_taps`` to ``hidden``.
        main_norm: RMS norm over ``hidden``.

    Returns:
        ``[T, hidden]`` combined hidden states.
    """
    concatenated = ops.concat(list(aux_hidden_states), axis=-1)
    return main_norm(main_proj(concatenated))


class DSparkMarkovHead(Module):
    """The sequential bias correction DSpark adds between draft positions.

    Embeds the previously drafted token and projects it to a bias over the
    draft vocabulary, which the caller adds to the next position's logits.
    Both weights stay replicated under tensor parallelism, avoiding
    collectives in the serial draft loop.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        device: DeviceRef,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Declares the embedding table and the projection back to logits.

        Args:
            vocab_size: Rows of the embedding table.
            draft_vocab_size: Width of the projected bias.
            markov_rank: Width of the embedding.
            device: Device the weights live on.
            dtype: Dtype of both weights.
        """
        super().__init__()
        self.embed = Embedding(
            vocab_size, markov_rank, dtype=dtype, device=device
        )
        self.head = Linear(
            in_dim=markov_rank,
            out_dim=draft_vocab_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def __call__(
        self, prev_tokens: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Embeds ``prev_tokens`` and projects the embedding to a bias.

        Args:
            prev_tokens: int32 token ids, any leading shape (``[b, 1]`` for
                one serial step).

        Returns:
            ``(markov_embed, bias)``: the embedding (``prev_tokens.shape +
            [markov_rank]``) and the projected bias (``prev_tokens.shape +
            [draft_vocab_size]``), both in this module's compute dtype.

        Raises:
            ValueError: If ``prev_tokens`` is not int32.
        """
        if prev_tokens.dtype != DType.int32:
            raise ValueError(
                "DSparkMarkovHead expects int32 token ids, got "
                f"{prev_tokens.dtype}"
            )
        markov_embed = self.embed(prev_tokens)
        bias = self.head(markov_embed)
        return markov_embed, bias

    def bias_float32(self, prev_tokens: TensorValue) -> TensorValue:
        """Computes ``head(embed(prev_tokens))`` in float32.

        The bf16 weights are widened in the graph.

        Args:
            prev_tokens: ``[b]`` int32 token ids.

        Returns:
            ``[b, draft_vocab_size]`` float32.

        Raises:
            ValueError: If ``prev_tokens`` is not int32.
        """
        if prev_tokens.dtype != DType.int32:
            raise ValueError(
                "DSparkMarkovHead expects int32 token ids, got "
                f"{prev_tokens.dtype}"
            )
        return ops.matmul(
            ops.cast(self.embed(prev_tokens), DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )


class DSparkConfidenceHead(Module):
    """Predicts one draft position's acceptance probability.

    A single float32 linear over the concatenation of a hidden state and
    that position's Markov embedding.
    """

    def __init__(
        self, *, hidden_size: int, markov_rank: int, device: DeviceRef
    ) -> None:
        """Declares the float32 projection.

        Args:
            hidden_size: Width of the hidden state.
            markov_rank: Width of the Markov embedding.
            device: Device the weight lives on.
        """
        super().__init__()
        self.proj = Linear(
            in_dim=hidden_size + markov_rank,
            out_dim=1,
            dtype=DType.float32,
            device=device,
            has_bias=False,
        )

    def __call__(
        self, hidden: TensorValue, markov_embed: TensorValue
    ) -> TensorValue:
        """Scores one position's acceptance probability.

        Args:
            hidden: ``[..., hidden_size]`` bfloat16 hidden state.
            markov_embed: ``[..., markov_rank]`` bfloat16 Markov embedding,
                :meth:`DSparkMarkovHead.__call__`'s first return value.

        Returns:
            ``[...]`` float32 confidence score.

        Raises:
            ValueError: If either input is not bfloat16.
        """
        if hidden.dtype != DType.bfloat16:
            raise ValueError(
                f"DSparkConfidenceHead expects bfloat16 hidden, got {hidden.dtype}"
            )
        if markov_embed.dtype != DType.bfloat16:
            raise ValueError(
                "DSparkConfidenceHead expects bfloat16 markov_embed, got "
                f"{markov_embed.dtype}"
            )
        combined = ops.cast(
            ops.concat([hidden, markov_embed], axis=-1), DType.float32
        )
        return ops.squeeze(self.proj(combined), axis=-1)


# One power-of-two scale per 32 elements across the whole post-RoPE KV vector.
DSPARK_KV_QUANT_BLOCK = 32


def dspark_kv_quantize(kv: TensorValue) -> TensorValue:
    """Rounds KV rows onto the MXFP8 grid, keeping the input dtype.

    Each block of 32 elements shares one power-of-two scale. GPU-only, like
    every ``float8_e4m3fn`` cast.
    """
    return fp8_qat_quantize(kv, DSPARK_KV_QUANT_BLOCK)


@dataclass(frozen=True)
class DSparkWindow:
    """One stage's context window, per request, as the block attention reads it.

    Request ``r``'s block starts at position ``L_r`` and sees context
    positions ``L_r - window .. L_r - 1``, those below zero masked
    (:func:`dspark_window_positions`). Row order does not matter.
    """

    kv: TensorValue
    """``[b, window, head_dim]`` context KV rows per request, as stored."""
    valid: TensorValue
    """``[b, window]`` bool: which rows hold a real position."""


def dspark_window_positions(
    starts: TensorValue, window: int
) -> tuple[TensorValue, TensorValue]:
    """The context positions each request's block reads, and which are real.

    Args:
        starts: ``[b]`` int32, each request's first block position ``L_r``
            (one past its last committed token, whose context KV is already
            stored).
        window: The sliding window length.

    Returns:
        ``([b, window]`` int32 positions clamped to ``>= 0``, ``[b, window]``
        bool validity). Column ``w`` is position ``L_r - window + w``.
    """
    device = starts.device
    back = ops.range(-window, 0, dtype=DType.int32, device=device)
    positions = ops.unsqueeze(
        ops.cast(starts, DType.int32), -1
    ) + ops.unsqueeze(back, 0)
    valid = positions >= 0
    zero = ops.constant(0, DType.int32, device)
    return ops.max(positions, zero), valid


def dspark_block_indices(valid: TensorValue, block_size: int) -> TensorValue:
    """Builds the attention indices for each block query.

    Indexes ``window`` context rows followed by ``block_size`` block rows,
    returning ``[b, block_size, window + block_size]`` int32. Every block
    query sees every valid context row and every block row (the block is
    non-causal). ``-1`` marks an invalid context row.
    """
    device = valid.device
    b, window = valid.shape[0], int(valid.shape[1])
    context = ops.where(
        valid,
        ops.unsqueeze(
            ops.range(0, window, dtype=DType.int32, device=device), 0
        ),
        ops.constant(-1, DType.int32, device),
    )
    block = ops.range(
        window, window + block_size, dtype=DType.int32, device=device
    )
    return ops.concat(
        [
            ops.broadcast_to(
                ops.unsqueeze(context, 1), [b, block_size, window]
            ),
            ops.broadcast_to(
                ops.reshape(block, [1, 1, block_size]),
                [b, block_size, block_size],
            ),
        ],
        axis=-1,
    )


def grouped_output_projection(
    out: TensorValue,
    freqs_cis: TensorValue,
    wo_a: Linear,
    wo_b: Linear,
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
) -> TensorValue:
    """Projects ``[t, num_heads, head_dim]`` attention output to ``hidden``.

    Applies inverse RoPE at the query positions, then ``wo_a`` block-diagonal
    over ``o_groups``, then ``wo_b``.
    """
    t = out.shape[0]
    out = ops.reshape(
        apply_rope_tail(
            ops.reshape(out, [1, t, num_heads, head_dim]),
            freqs_cis,
            int(freqs_cis.shape[-2]) * 2,
            inverse=True,
        ),
        [t, num_heads, head_dim],
    )
    group_width = num_heads * head_dim // o_groups
    grouped = ops.transpose(ops.reshape(out, [t, o_groups, group_width]), 0, 1)
    wo_a_weight = ops.reshape(wo_a.weight, [o_groups, o_lora_rank, group_width])
    projected = ops.matmul(grouped, ops.transpose(wo_a_weight, -1, -2))
    flattened = ops.reshape(
        ops.transpose(projected, 0, 1), [t, o_groups * o_lora_rank]
    )
    return wo_b(flattened)


class DeepseekV41DSparkAttention(Module):
    """One DSpark stage's attention.

    A block query attends over the stage's context KV (projected from the
    combined hidden states ``main_x`` and stored per verified token, see
    :meth:`context_kv`) plus the block's own KV, projected the same way and
    never stored. Stages are uncompressed sliding-window layers.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_info: LayerInfo,
        device: DeviceRef,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if layer_info.side != "drafter" or layer_info.compress_ratio != 0:
            raise ValueError(
                f"layer {layer_info.layer_id} is not a ratio-0 DSpark stage "
                f"(side={layer_info.side!r}, "
                f"compress_ratio={layer_info.compress_ratio})"
            )
        self.layer_info = layer_info
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.block_size = config.dspark_block_size
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5
        self.rope = rope_for_layer(
            config, layer_info.layer_id, max_seq_len, device
        )
        # Host-dequantized bf16; no per-head query renorm.
        self.query_proj = DeepseekV41Query(
            config.hidden_size,
            config.q_lora_rank,
            self.head_dim,
            self.num_heads,
            config.rms_norm_eps,
            DType.bfloat16,
            device,
        )
        self.kv_norm = RMSNorm(
            self.head_dim, DType.float32, config.rms_norm_eps
        )
        self.attn_sink = Weight(
            name="attn_sink",
            dtype=DType.float32,
            shape=(self.num_heads,),
            device=device,
        )
        # Host-dequantized bf16: used as a raw weight in an einsum.
        self.wo_a = Linear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            DType.bfloat16,
            device,
        )
        self.wo_b = Linear(
            in_dim=self.o_groups * self.o_lora_rank,
            out_dim=config.hidden_size,
            dtype=DType.float8_e4m3fn,
            device=device,
            quant_config=dspark_main_proj_quant_config(),
        )

    def _freqs_cis(self, positions: TensorValue) -> TensorValue:
        return ops.gather(self.rope.freqs_cis_base(), positions, axis=0)

    def _kv(self, kv_raw: TensorValue, freqs_cis: TensorValue) -> TensorValue:
        """``kv_norm``, then RoPE on the trailing dims, then the quantize."""
        return dspark_kv_quantize(
            apply_rope_tail(
                self.kv_norm(kv_raw), freqs_cis, int(freqs_cis.shape[-2]) * 2
            )
        )

    def _kv_raw(self, x: TensorValue) -> TensorValue:
        """Computes ``wkv(x)`` from the KV rows of the fused ``wq_a | wkv``."""
        q_lora_rank = self.query_proj.q_lora_rank
        weight = self.query_proj.fused_wqa_wkv.weight
        kv_weight = ops.slice_tensor(
            weight.to(x.device),
            [slice(q_lora_rank, q_lora_rank + self.head_dim), slice(None)],
        )
        return ops.matmul(x, ops.transpose(kv_weight, 0, 1))

    def context_kv(
        self, main_x: TensorValue, positions: TensorValue
    ) -> TensorValue:
        """This stage's context KV rows for ``main_x``'s tokens.

        ``kv_norm(wkv(main_x))``, RoPE at each token's position, then the
        32-wide quantize.

        Args:
            main_x: ``[1, T, hidden_size]``, the drafter's combined taps.
            positions: ``[T]`` int32, each token's absolute position.

        Returns:
            ``[T, head_dim]``.
        """
        freqs_cis = self._freqs_cis(positions)
        kv = self._kv(self._kv_raw(main_x), freqs_cis)
        return ops.reshape(kv, [-1, self.head_dim])

    def _swa_layer(self) -> TensorValue:
        """This stage's layer index in the sliding-window cache."""
        return ops.constant(
            self.layer_info.layer_id, DType.uint32, DeviceRef.CPU()
        )

    def store_context(
        self,
        main_x: TensorValue,
        rows: RaggedRows,
        swa_collection: PagedCacheValues,
    ) -> None:
        """Stores :meth:`context_kv` for every token of a forward pass.

        Args:
            main_x: ``[1, T, hidden]`` combined hidden states of the forward
                pass's tokens.
            rows: That forward's bookkeeping; ``swa_collection``'s
                ``cache_lengths`` must be its ``starts`` (the pre-forward
                lengths), so token ``i`` of request ``r`` lands at position
                ``starts[r] + i``.
            swa_collection: The sliding-window KV cache.
        """
        kv = self.context_kv(main_x, rows.positions)
        store_k_cache_ragged(
            swa_collection,
            ops.reshape(
                ops.cast(kv, swa_collection.kv_blocks.dtype),
                [-1, 1, self.head_dim],
            ),
            rows.offsets,
            self._swa_layer(),
        )

    def read_window(
        self, starts: TensorValue, swa_collection: PagedCacheValues
    ) -> DSparkWindow:
        """Reads each request's :class:`DSparkWindow` from the cache.

        Reads positions ``starts[r] - window .. starts[r] - 1``. Rows stored
        for tokens not yet accepted sit at or past ``starts`` and are
        overwritten by the next store before any window reads them.

        Args:
            starts: ``[b]`` int32, each request's first block position.
            swa_collection: The sliding-window KV cache.

        Returns:
            The window: ``[b, window, head_dim]`` KV rows and their validity.
        """
        positions, valid = dspark_window_positions(starts, self.window)
        device = starts.device
        b = starts.shape[0]
        per_request = ops.cast(arange_to(b + 1, b + 1, device), DType.uint32)
        kv = kv_cache_gather_rows_ragged(
            swa_collection,
            positions,
            per_request,
            self._swa_layer(),
            key_or_value=KEY_CACHE_INDEX,
        )
        return DSparkWindow(kv=kv, valid=valid)

    def shard_heads(self, num_devices: int) -> None:
        """Declares how the attention heads split across devices.

        ``wq_b`` and ``attn_sink`` split per head, ``wo_a`` by group and
        ``wo_b`` by input, so each device's output is a partial sum. The
        latent side (``fused_wqa_wkv``, norms) stays whole, so every device
        computes the whole context KV.

        Raises:
            ValueError: If heads or groups do not divide across devices.
        """
        if self.num_heads % num_devices or self.o_groups % num_devices:
            raise ValueError(
                f"{self.num_heads} heads in {self.o_groups} groups do not "
                f"split across {num_devices} devices"
            )
        self.query_proj.wq_b.sharding_strategy = ShardingStrategy.rowwise(
            num_devices
        )
        self.attn_sink.sharding_strategy = ShardingStrategy.axiswise(
            0, num_devices
        )
        self.wo_a.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.wo_b.sharding_strategy = ShardingStrategy.columnwise(num_devices)

    def keep_local_heads(self, num_devices: int) -> None:
        """Makes this replica compute only its share of :meth:`shard_heads`."""
        self.num_heads //= num_devices
        self.o_groups //= num_devices
        self.query_proj.local_num_heads //= num_devices

    def __call__(
        self, h: TensorValue, rows: RaggedRows, cache: object | None
    ) -> TensorValue:
        """Attends each draft block over its context window and itself.

        Maps ``[1, b * block_size, hidden]`` normed block tokens to the same
        shape.

        Args:
            h: The ``attn_norm``'d collapsed stream of ``b`` blocks, request
                major, ``block_size`` rows each.
            rows: The blocks' bookkeeping; only ``rows.positions`` is read.
            cache: This stage's :class:`DSparkWindow`.

        Raises:
            TypeError: If ``cache`` is not a :class:`DSparkWindow`.
        """
        if not isinstance(cache, DSparkWindow):
            raise TypeError(
                "DeepseekV41DSparkAttention requires a DSparkWindow, got "
                f"{type(cache)!r}"
            )
        k, heads, d = self.block_size, self.num_heads, self.head_dim
        t, hidden = h.shape[1], h.shape[2]
        b = cache.kv.shape[0]
        freqs_cis = self._freqs_cis(rows.positions)
        _, q, kv_raw = self.query_proj(h, freqs_cis)
        block_kv = self._kv(kv_raw, freqs_cis)

        # `t` rows are `b` blocks of `k`: rebind `t` to `b * k` to split it.
        q = ops.reshape(
            ops.rebind(ops.reshape(q, [t, heads, d]), [b * k, heads, d]),
            [b, k, heads, d],
        )
        block_kv = ops.reshape(
            ops.rebind(ops.reshape(block_kv, [t, d]), [b * k, d]), [b, k, d]
        )
        table = ops.concat(
            [ops.cast(cache.kv, block_kv.dtype), block_kv], axis=1
        )
        out = gathered_sparse_attention(
            q,
            table,
            self.attn_sink,
            dspark_block_indices(cache.valid, k),
            self.scale,
        )
        out = ops.rebind(ops.reshape(out, [b * k, heads, d]), [t, heads, d])
        projected = grouped_output_projection(
            out,
            freqs_cis,
            self.wo_a,
            self.wo_b,
            num_heads=heads,
            head_dim=d,
            o_groups=self.o_groups,
            o_lora_rank=self.o_lora_rank,
        )
        return ops.reshape(projected, [1, t, hidden])
