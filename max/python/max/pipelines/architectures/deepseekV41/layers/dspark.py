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
from max.nn.quant_ops import _matmul_float8_mxfp8

from ...deepseekV4.layers.ragged import arange_to
from ..layer_resolver import LayerInfo
from ..model_config import (
    DeepseekV41Config,
    q_mxfp8_enabled,
    woa_mxfp8_enabled,
)
from .attention import (
    bf16_grouped_projection,
    mxfp8_grouped_projection,
    undo_rope_tail,
    wo_a_linear,
)
from .csa2 import (
    INDEX_WIDTH,
    csa2_attention,
    leaf_slots,
    store_window_kv,
)
from .indexer import rope_cos_sin_table
from .mxfp8_matmul import mxfp8_linear
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


@dataclass(frozen=True)
class DSparkWindow:
    """One stage's context window, per request, as the block attention names it.

    Request ``r``'s block starts at position ``L_r`` and sees context
    positions ``L_r - window .. L_r - 1`` (:func:`dspark_window_slots`).
    """

    slots: TensorValue
    """``[b, window]`` int32 absolute window-cache slots of those positions
    in the stage's layer, ``-1`` below position zero."""
    cache: PagedCacheValues
    """The window cache leaf the slots name."""


def dspark_window_slots(
    cache: PagedCacheValues, starts: TensorValue, window: int, layer: int
) -> TensorValue:
    """Returns the window-cache slots of the context each block reads.

    Args:
        cache: The window cache leaf.
        starts: ``[b]`` int32, each request's first block position ``L_r``
            (one past its last committed token, whose context record is
            already stored).
        window: The sliding window length.
        layer: The stage's layer among the leaf's layers.

    Returns:
        ``[b, window]`` int32 slots; column ``w`` holds position
        ``L_r - window + w``, ``-1`` where that is negative.
    """
    device = starts.device
    b = starts.shape[0]
    back = ops.range(-window, 0, dtype=DType.int32, device=device)
    positions = ops.unsqueeze(
        ops.cast(starts, DType.int32), -1
    ) + ops.unsqueeze(back, 0)
    return leaf_slots(
        cache, ops.unsqueeze(arange_to(b, b, device), -1), positions, layer
    )


def dspark_block_index_lists(
    window_slots: TensorValue, block_slots: TensorValue, width: int
) -> TensorValue:
    """Builds each block query's attention list of window-cache slots.

    Every block query names its request's context slots, then every slot of
    its own block (the block is non-causal), then ``-1`` up to ``width``.

    Args:
        window_slots: ``[b, window]`` int32 context slots, ``-1`` for none.
        block_slots: ``[b, k]`` int32 slots of the block's rows.
        width: The list width, at least ``window + k``.

    Returns:
        ``[b * k, width]`` int32, request major.
    """
    b, window = window_slots.shape[0], int(window_slots.shape[1])
    k = int(block_slots.shape[1])
    lists = ops.concat(
        [
            ops.broadcast_to(ops.unsqueeze(window_slots, 1), [b, k, window]),
            ops.broadcast_to(ops.unsqueeze(block_slots, 1), [b, k, k]),
            ops.broadcast_to(
                ops.constant(-1, DType.int32, window_slots.device),
                [b, k, width - window - k],
            ),
        ],
        axis=-1,
    )
    return ops.reshape(lists, [b * k, width])


def grouped_output_projection(
    out: TensorValue,
    freqs_cis: TensorValue,
    wo_a: Linear,
    wo_b: Linear,
    *,
    num_heads: int,
    head_dim: int,
    o_groups: int,
) -> TensorValue:
    """Projects ``[t, num_heads, head_dim]`` attention output to ``hidden``.

    Applies inverse RoPE at the query positions, then ``wo_a`` block-diagonal
    over ``o_groups`` (MXFP8 when it carries a ``weight_scale``, else bf16),
    then ``wo_b``.
    """
    out = ops.reshape(out, [out.shape[0], num_heads * head_dim])
    if wo_a.weight_scale is not None:
        return mxfp8_linear(
            wo_b,
            mxfp8_grouped_projection(
                out,
                wo_a.weight,
                wo_a.weight_scale,
                o_groups,
                rope=(freqs_cis, head_dim),
            ),
        )
    return mxfp8_linear(
        wo_b,
        bf16_grouped_projection(
            undo_rope_tail(out, freqs_cis, head_dim), wo_a.weight, o_groups
        ),
    )


class DeepseekV41DSparkAttention(Module):
    """One DSpark stage's attention.

    A block query attends over the stage's context records (projected from
    the combined hidden states ``main_x`` and stored per verified token, see
    :meth:`store_context`) plus the block's own records, projected the same
    way and stored at the block's positions, where the next forward's
    context store overwrites them. Stages are uncompressed sliding-window
    layers.
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
        # Each block query names its window and its block, in whole tiles
        # of 64 list entries.
        self.list_entries = -(-(self.window + self.block_size) // 64) * 64
        if self.list_entries > INDEX_WIDTH:
            raise ValueError(
                f"a window of {self.window} and a block of "
                f"{self.block_size} do not fit a CSA2 list"
            )
        self.rope = rope_for_layer(
            config, layer_info.layer_id, max_seq_len, device
        )
        # MXFP8, or bf16 host-dequantized weights with `MAX_DSV41_Q_MXFP8=0`.
        # No per-head query renorm.
        self.query_proj = DeepseekV41Query(
            config.hidden_size,
            config.q_lora_rank,
            self.head_dim,
            self.num_heads,
            config.rms_norm_eps,
            DType.bfloat16,
            device,
            quant_config=(
                dspark_main_proj_quant_config() if q_mxfp8_enabled() else None
            ),
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
        self.wo_a = wo_a_linear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            device,
            mxfp8=woa_mxfp8_enabled(),
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

    def _kv_raw(self, x: TensorValue) -> TensorValue:
        """Computes ``wkv(x)`` from the KV rows of the fused ``wq_a | wkv``."""
        q_lora_rank = self.query_proj.q_lora_rank
        fused = self.query_proj.fused_wqa_wkv
        kv_rows = [slice(q_lora_rank, q_lora_rank + self.head_dim), slice(None)]
        kv_weight = ops.slice_tensor(fused.weight.to(x.device), kv_rows)
        if fused.weight_scale is None:
            return ops.matmul(x, ops.transpose(kv_weight, 0, 1))
        kv_scale = ops.slice_tensor(fused.weight_scale.to(x.device), kv_rows)
        out = _matmul_float8_mxfp8(
            ops.reshape(x, [-1, x.shape[-1]]), kv_weight, kv_scale
        )
        return ops.reshape(out, [*x.shape[:-1], self.head_dim])

    def store_context(
        self,
        main_x: TensorValue,
        rows: RaggedRows,
        swa_collection: PagedCacheValues,
    ) -> None:
        """Stores every stage context record of a forward pass's tokens.

        Args:
            main_x: ``[1, T, hidden]`` combined hidden states of the forward
                pass's tokens.
            rows: That forward's bookkeeping; ``swa_collection``'s
                ``cache_lengths`` must be its ``starts`` (the pre-forward
                lengths), so token ``i`` of request ``r`` lands at position
                ``starts[r] + i``.
            swa_collection: The sliding-window KV cache.

        A record is ``kv_norm(wkv(main_x))``, rotated at the token's
        position and quantized to MXFP8 by the store.
        """
        self._store(self._kv_raw(main_x), rows, swa_collection)

    def _store(
        self, kv_raw: TensorValue, rows: RaggedRows, cache: PagedCacheValues
    ) -> TensorValue:
        """Stores ``kv_norm(kv_raw)``'s records at ``rows``' positions.

        Returns:
            ``[T]`` int32, the slot each row was stored at.
        """
        slots = leaf_slots(
            cache, rows.bid, rows.positions, self.layer_info.layer_id
        )
        latent = self.kv_norm(kv_raw)
        store_window_kv(
            cache.kv_blocks,
            ops.reshape(ops.cast(latent, DType.bfloat16), [-1, self.head_dim]),
            rows.positions,
            rope_cos_sin_table(self.rope.freqs_cis_base()),
            slots,
            block_stride=cache.page_stride,
        )
        return slots

    def read_window(
        self, starts: TensorValue, swa_collection: PagedCacheValues
    ) -> DSparkWindow:
        """Names each request's :class:`DSparkWindow` in the cache.

        Positions ``starts[r] - window .. starts[r] - 1``. Records stored for
        tokens not yet accepted sit at or past ``starts`` and are
        overwritten by the next store before any window names them.

        Args:
            starts: ``[b]`` int32, each request's first block position.
            swa_collection: The sliding-window KV cache.
        """
        return DSparkWindow(
            slots=dspark_window_slots(
                swa_collection, starts, self.window, self.layer_info.layer_id
            ),
            cache=swa_collection,
        )

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
        shape. The block's records are stored at its positions in the
        stage's window cache, then each block query attends over the slots
        of its window and of its whole block.

        Args:
            h: The ``attn_norm``'d collapsed stream of ``b`` blocks, request
                major, ``block_size`` rows each.
            rows: The blocks' bookkeeping: each row's request and position.
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
        b = cache.slots.shape[0]
        freqs_cis = self._freqs_cis(rows.positions)
        _, q, kv_raw = self.query_proj(h, freqs_cis)
        block_slots = self._store(kv_raw, rows, cache.cache)

        # `t` rows are `b` blocks of `k`: rebind `t` to `b * k` to split it.
        block_slots = ops.reshape(
            ops.rebind(ops.reshape(block_slots, [t]), [b * k]), [b, k]
        )
        lists = ops.rebind(
            dspark_block_index_lists(
                cache.slots, block_slots, self.list_entries
            ),
            [t, self.list_entries],
        )
        idx = ops.concat(
            [
                lists,
                ops.broadcast_to(
                    ops.constant(-1, DType.int32, lists.device),
                    [t, INDEX_WIDTH - self.list_entries],
                ),
            ],
            axis=-1,
        )
        q = ops.reshape(q, [t, heads, d])
        out = csa2_attention(
            q,
            idx,
            self.attn_sink,
            cache.cache.kv_blocks,
            None,
            scale=self.scale,
            window_stride=cache.cache.page_stride,
            window_entries=self.list_entries,
        )
        out = ops.cast(out, q.dtype)
        projected = grouped_output_projection(
            out,
            freqs_cis,
            self.wo_a,
            self.wo_b,
            num_heads=heads,
            head_dim=d,
            o_groups=self.o_groups,
        )
        return ops.reshape(projected, [1, t, hidden])
