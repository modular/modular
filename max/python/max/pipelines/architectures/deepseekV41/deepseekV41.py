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
"""The DeepSeek-V4.1 model: decoder blocks, the trunk and the full model.

Each decoder block wraps attention and a feed-forward network in
hyper-connections, which keep several copies of the residual stream and learn
how to mix them. Some blocks first add an Engram value: an embedding looked up
from hashed n-grams of the input tokens. Image span positions take their
embeddings from the vision model and take no part in Engram.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    DimLike,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import Embedding
from max.nn.hyper_connection import HyperConnection
from max.nn.kernels import mhc_post, mhc_post_supports
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.transformer import DecoderOutputs, logits_postprocess
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from typing_extensions import Self

from ..deepseekV4.layers import (
    DeepseekV4Expert,
    DeepseekV4Gate,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    apply_rope_tail,
    expand_copies,
    hc_post,
    rope_for_layer,
)
from ..deepseekV4.layers.quantization import (
    fp4_qat_quantize,
    fp8_qat_quantize,
)
from .layer_resolver import LayerInfo
from .layer_resolver import resolve as resolve_layers
from .layers import final_collapse, fused_hc_step
from .layers.attention import (
    CSA2Attention,
    CSA2Cache,
    make_csa2_attn_factory,
)
from .layers.dspark_moe import DeepseekV41DSparkMoE
from .layers.engram import Engram
from .layers.engram_hash import EngramLayout
from .layers.engram_lookup import (
    EngramLookup,
    EngramOperand,
    EngramPending,
    EngramTokens,
)
from .layers.moe import (
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV41MoE,
    nvfp4_moe_ffn_factory,
)
from .layers.ragged import ReplicaTokens
from .layers.ragged import V41RaggedRows as RaggedRows
from .model_config import (
    DeepseekV41Config,
    head_bf16_enabled,
    position_capacity,
)

# Re-exports the V4 layers this model uses unchanged.
__all__ = [
    "DeepseekV4Expert",
    "DeepseekV4Gate",
    "DeepseekV4MoE",
    "DeepseekV4RotaryEmbedding",
    "DeepseekV41",
    "DeepseekV41Block",
    "DeepseekV41Trunk",
    "RaggedRows",
    "VisionMerge",
    "apply_rope_tail",
    "fp4_qat_quantize",
    "fp8_qat_quantize",
    "rope_for_layer",
]


def _adopt_shards(
    source: Module, replicas: Sequence[Module], devices: list[DeviceRef]
) -> None:
    """Points each replica's weights at the matching shard of ``source``'s.

    ``replicas`` are structurally identical to ``source`` and walked in
    step. A weight with no strategy is replicated; a ``Linear``'s goes
    through its setter so a block-scaled ``weight_scale`` is sharded with the
    weight.
    """
    n = len(devices)
    if isinstance(source, Linear) and source.sharding_strategy is None:
        source.sharding_strategy = ShardingStrategy.replicate(n)
    for name, value in list(vars(source).items()):
        if not isinstance(value, Weight):
            continue
        if value.sharding_strategy is None:
            value.sharding_strategy = ShardingStrategy.replicate(n)
        for replica, shard in zip(replicas, value.shard(devices), strict=True):
            setattr(replica, name, shard)
    for name, child in source.sublayers.items():
        _adopt_shards(
            child, [replica.sublayers[name] for replica in replicas], devices
        )


@dataclass(frozen=True)
class VisionMerge:
    """Image span embeddings and the token rows they replace."""

    embeddings: TensorValue
    """``[n, d]`` rows from the vision model."""

    indices: TensorValue
    """``[n]`` int32 token row of each embedding; ``int32`` min is skipped.

    The scatter wraps any other negative index to a row from the end.
    """

    def for_rows(
        self, offsets: TensorValue, first: TensorValue, past: TensorValue
    ) -> VisionMerge:
        """Keeps the embeddings for the token rows of requests ``[first, past)``.

        Their rows count from the first of those requests' rows; every other
        row is skipped. ``offsets`` are the batch's ``[B + 1]`` row offsets on
        the embeddings' device; ``first`` and ``past`` are int64 request
        indices on CPU, so no host value is copied to the device.
        """
        device = self.indices.device
        off = ops.cast(offsets, DType.int32)
        start, stop = (
            ops.rebind(
                ops.slice_tensor(
                    off, [(slice(i, i + 1), f"vision_row_bound_{name}")]
                ),
                [1],
            )
            for i, name in ((first, "start"), (past, "stop"))
        )
        inside = ops.logical_and(self.indices >= start, self.indices < stop)
        return VisionMerge(
            self.embeddings,
            ops.where(
                inside,
                self.indices - start,
                ops.constant(np.iinfo(np.int32).min, DType.int32, device),
            ),
        )


def merge_vision(x: TensorValue, vision: VisionMerge | None) -> TensorValue:
    """Scatters the image span rows into the ``[1, s, d]`` token embeddings."""
    if vision is None:
        return x
    s, d = x.shape[1], x.shape[2]
    merged = merge_multimodal_embeddings(
        ops.reshape(x, [s, d]), vision.embeddings, vision.indices
    )
    return ops.reshape(merged, [1, s, d])


def is_text_token(tokens: TensorValue) -> TensorValue:
    """``[s]`` bool, false at image span positions."""
    return ops.not_equal(ops.reshape(tokens, [-1]), IMAGE_SENTINEL_BASE_ID)


def engram_tokens(
    tokens: TensorValue,
    rows: RaggedRows,
    lookback_token_ids: TensorValue | None,
    depth: int,
) -> EngramTokens:
    """Returns one forward's Engram hash inputs, on ``tokens``' device.

    ``lookback_token_ids`` is ``[b, depth]``: column ``j`` holds the token at
    ``start - 1 - j``, ``-1`` before the sequence start. ``None`` means every
    request starts at position 0. Image span tokens, in the chunk or its
    lookback, are dead: no n-gram spans them.
    """
    device = tokens.device
    ids = ops.cast(ops.reshape(tokens, [-1]), DType.int32)
    num_reqs = rows.batch
    if lookback_token_ids is None:
        lookback = ops.broadcast_to(
            ops.constant(-1, DType.int32, device), [num_reqs, depth]
        )
    else:
        lookback = ops.rebind(
            ops.cast(lookback_token_ids, DType.int32), [num_reqs, depth]
        )
    return EngramTokens(
        ids=ids,
        positions=ops.cast(rows.positions, DType.int64),
        query_start_loc=ops.cast(rows.offsets, DType.int32),
        dead_mask=ops.equal(ids, IMAGE_SENTINEL_BASE_ID),
        lookback=lookback,
        lookback_dead=ops.equal(lookback, IMAGE_SENTINEL_BASE_ID),
    )


def _hyper_connection(
    config: DeepseekV41Config, device: DeviceRef
) -> HyperConnection:
    """Builds the hyper-connection around a block's attention or FFN."""
    return HyperConnection(
        config.hidden_size,
        config.hc_mult,
        device,
        DType.float32,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
    )


def _hc_pre(
    x: TensorValue, hc: HyperConnection, pre_mix: TensorValue | None
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """Merges the ``[1, s, hc, d]`` stream copies into one ``[1, s, d]`` input.

    The copies are merged with the ``pre`` weights computed by the previous
    sublayer, never with this sublayer's own, so each sublayer's mixing
    weights are computed once and carried forward (Single-Pass mHC).
    ``pre_mix=None`` marks the first sublayer of a stack, whose copies are
    still identical; it selects copy 0.

    Returns:
        The merged stream and this sublayer's ``post``, ``comb`` and ``pre``
        mixing weights.
    """
    b, s = x.shape[0], x.shape[1]
    if pre_mix is None:
        folded_pre_mix = ops.constant(
            np.eye(hc.hc_mult, dtype=np.float32)[0],
            DType.float32,
            device=x.device,
        )
    else:
        folded_pre_mix = ops.reshape(pre_mix, [-1, hc.hc_mult])
    post, comb, y, pre = hc(
        ops.reshape(x, [-1, hc.hc_mult, hc.hidden_size]), folded_pre_mix
    )
    return (
        ops.reshape(y, [b, s, hc.hidden_size]),
        ops.reshape(post, [b, s, hc.hc_mult]),
        ops.reshape(comb, [b, s, hc.hc_mult, hc.hc_mult]),
        ops.reshape(pre, [b, s, hc.hc_mult]),
    )


def _hc_mean(x: TensorValue) -> TensorValue:
    """Averages ``[1, s, hc, d]`` over the copies, accumulating in float32.

    A sum of per-copy slices is one elementwise pass; a reduce over the cast
    copies materializes a float32 copy of all of them first.
    """
    hc_mult = int(x.shape[2])
    total = ops.cast(x[:, :, 0, :], DType.float32)
    for i in range(1, hc_mult):
        total = total + ops.cast(x[:, :, i, :], DType.float32)
    return ops.cast(total * (1.0 / hc_mult), x.dtype)


@dataclass(frozen=True)
class HCStreams:
    """The ``[1, s, hc, d]`` stream copies between two sublayers.

    With ``out`` set, the copies are ``residual`` plus the earlier
    sublayer's ``[1, s, d]`` output, still to be written back through
    ``post`` and ``comb``; the next sublayer fuses that write-back into its
    own mixing step. With ``out`` unset, ``residual`` holds the copies.
    ``pre`` is the collapse weight carried into the next sublayer, ``None``
    before the first sublayer of a stack.
    """

    residual: TensorValue
    pre: TensorValue | None = None
    out: TensorValue | None = None
    post: TensorValue | None = None
    comb: TensorValue | None = None

    def materialize(self) -> TensorValue:
        """Returns the ``[1, s, hc, d]`` copies with ``out`` written back."""
        if self.out is None:
            return self.residual
        assert self.post is not None and self.comb is not None
        b, s, hc_mult, d = self.residual.shape
        if not self.residual.device.is_gpu() or not mhc_post_supports(
            int(hc_mult), int(d), self.residual.dtype
        ):
            return hc_post(self.out, self.residual, self.post, self.comb)
        written = mhc_post(
            ops.reshape(self.residual, [-1, hc_mult, d]),
            ops.reshape(ops.cast(self.out, self.residual.dtype), [-1, d]),
            ops.reshape(self.post, [-1, hc_mult]),
            ops.reshape(self.comb, [-1, hc_mult, hc_mult]),
        )
        return ops.reshape(written, [b, s, hc_mult, d])

    def collapse(self) -> TensorValue:
        """Merges the copies with ``pre`` into the final ``[1, s, d]`` state."""
        assert self.pre is not None  # at least one sublayer always runs
        return final_collapse(self.materialize(), self.pre)


def _per_token(value: np.ndarray, like: TensorValue) -> TensorValue:
    """Broadcasts the float32 ``value`` to ``[b, s, *value.shape]`` of ``like``."""
    return ops.broadcast_to(
        ops.constant(value, DType.float32, device=like.device),
        [like.shape[0], like.shape[1], *value.shape],
    )


def _identity_write_back(
    copies: TensorValue, pre: TensorValue, out: TensorValue
) -> HCStreams:
    """Holds ``copies`` as ``out`` still to be written back unchanged.

    With ``post = 0`` and ``comb = I`` the write-back leaves the copies as
    they are, so the next sublayer takes the fused step instead of the
    unfused gates and collapse. ``out`` is any finite ``[1, s, d]`` tensor.
    """
    hc_mult = int(copies.shape[2])
    return HCStreams(
        copies,
        pre=pre,
        out=out,
        post=_per_token(np.zeros(hc_mult, dtype=np.float32), copies),
        comb=_per_token(np.eye(hc_mult, dtype=np.float32), copies),
    )


def first_streams(x: TensorValue, hc_mult: int) -> HCStreams:
    """Returns the copies of the ``[1, s, d]`` embedding the first block reads.

    The copies are identical, so ``pre`` selects copy 0.
    """
    pre = _per_token(np.eye(hc_mult, dtype=np.float32)[0], x)
    return _identity_write_back(expand_copies(x, hc_mult), pre, x)


def _replace_copies(streams: HCStreams, copies: TensorValue) -> HCStreams:
    """Returns ``copies`` in place of the copies of ``streams``, with its ``pre``."""
    if streams.out is None or streams.pre is None:
        return HCStreams(copies, pre=streams.pre)
    return _identity_write_back(copies, streams.pre, streams.out)


def _hc_enter(
    streams: HCStreams, hc: HyperConnection, norm: RMSNorm
) -> tuple[TensorValue, HCStreams]:
    """Mixes the copies into a sublayer's normalized ``[1, s, d]`` input.

    Returns the input and the copies the sublayer reads, with its own
    ``post``, ``comb`` and ``pre``; set ``out`` on them once it runs.
    """
    if streams.out is None:
        h, post, comb, pre = _hc_pre(streams.residual, hc, streams.pre)
        return norm(h), HCStreams(
            streams.residual, pre=pre, post=post, comb=comb
        )
    assert streams.post is not None and streams.comb is not None
    assert streams.pre is not None
    h, residual, post, comb, pre = fused_hc_step(
        streams.out,
        streams.residual,
        streams.post,
        streams.comb,
        streams.pre,
        hc,
        norm,
    )
    return h, HCStreams(residual, pre=pre, post=post, comb=comb)


def _with_out(streams: HCStreams, out: TensorValue) -> HCStreams:
    """Records the sublayer output ``out`` on the copies it read."""
    return HCStreams(
        streams.residual,
        pre=streams.pre,
        out=out,
        post=streams.post,
        comb=streams.comb,
    )


def _apply_engram(
    x: TensorValue,
    engram_fn: Engram,
    kv: TensorValue,
    token_mask: TensorValue | None,
) -> TensorValue:
    """Gates the Engram value ``kv`` into the ``[1, s, hc, d]`` stream.

    Reshapes to ``[s, hc, d]`` for :meth:`Engram.gate` and back.
    """
    b, s, hc_mult, hidden_size = (
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
    )
    y = engram_fn.gate(
        ops.reshape(x, [-1, hc_mult, hidden_size]), kv, token_mask
    )
    return ops.reshape(y, [b, s, hc_mult, hidden_size])


@dataclass
class ReplicaBatch:
    """A replica-major batch cut into each data-parallel device's requests."""

    tokens: list[TensorValue]
    """Per device, its own ``[1, t_i]`` token ids."""
    batch_tokens: list[TensorValue]
    """Per device, the whole batch's ``[1, T]`` token ids."""
    rows: list[RaggedRows]
    """Per device, its own requests' bookkeeping."""
    replica_tokens: ReplicaTokens
    """Where each device's rows sit in the batch."""
    lookbacks: list[TensorValue | None]
    """Per device, its requests' Engram lookback rows, or ``None``."""


def gather_rows(
    xs: Sequence[TensorValue],
    total: DimLike,
    signal_buffers: Sequence[BufferValue],
) -> TensorValue:
    """Gathers each device's ``[1, t_i, d]`` rows into ``[T, d]`` on device 0.

    Devices hold consecutive runs of the batch's rows, in device order.
    """
    hidden = int(xs[0].shape[-1])
    gathered = ops.allgather(
        [ops.reshape(x, [-1, hidden]) for x in xs], signal_buffers, axis=0
    )
    return ops.rebind(gathered[0], [total, hidden])


class DeepseekV41Block(Module):
    """A decoder block: attention then FFN, each inside a hyper-connection.

    The block takes its ``attn`` and ``ffn`` modules and owns the
    hyper-connection mixing and the optional Engram step. Each call takes the
    previous sublayer's ``pre`` mixing weights and returns this block's FFN
    ``pre`` for the next block.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_info: LayerInfo,
        device: DeviceRef,
        *,
        attn: Module,
        ffn: Module,
        engram_fn: Engram | None = None,
    ) -> None:
        super().__init__()
        if layer_info.has_engram and engram_fn is None:
            raise ValueError(
                f"layer {layer_info.layer_id} is an Engram layer "
                "(has_engram=True) but no engram_fn was given"
            )
        if not layer_info.has_engram and engram_fn is not None:
            raise ValueError(
                f"layer {layer_info.layer_id} is not an Engram layer "
                "(has_engram=False) but an engram_fn was given"
            )
        self.layer_info = layer_info
        self.attn = attn
        self.ffn = ffn
        self.engram_fn = engram_fn
        self.attn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.hc_attn = _hyper_connection(config, device)
        self.hc_ffn = _hyper_connection(config, device)

    @staticmethod
    def tensor_parallel(
        blocks: Sequence[DeepseekV41Block],
        streams: Sequence[HCStreams],
        rows: Sequence[RaggedRows],
        token_ids: Sequence[TensorValue],
        caches: Sequence[object],
        signal_buffers: Sequence[BufferValue],
        *,
        engram_operands: Sequence[EngramOperand] | None = None,
        engram_token_masks: Sequence[TensorValue | None] | None = None,
    ) -> tuple[list[HCStreams], list[TensorValue]]:
        """Runs :meth:`__call__` on every device, one replica per device.

        Attention is split by heads, and its output-projection partial sums
        are all-reduced before the hyper-connection merge; for
        :class:`CSA2Attention`, indexer scores are also all-reduced before the
        top-k. The FFN all-reduces its experts' sum. The Engram projection is
        split by columns and all-gathered before the gate. The gate and both
        hyper-connections run in full on every device.

        Returns:
            Each device's output copies and the copies its attention read.
        """
        moes: list[DeepseekV41MoE | DeepseekV41DSparkMoE] = []
        for block in blocks:
            assert isinstance(block.ffn, (DeepseekV41MoE, DeepseekV41DSparkMoE))
            moes.append(block.ffn)

        if blocks[0].engram_fn is not None:
            if engram_operands is None:
                raise ValueError(
                    f"layer {blocks[0].layer_info.layer_id} is an Engram "
                    "layer but no engram_operands was given"
                )
            engrams: list[Engram] = []
            for block in blocks:
                assert block.engram_fn is not None
                engrams.append(block.engram_fn)
            kvs = [
                engram.project(operand)
                for engram, operand in zip(
                    engrams, engram_operands, strict=True
                )
            ]
            if len(kvs) > 1:
                kvs = ops.allgather(kvs, signal_buffers, axis=-1)
            masks = engram_token_masks or [None] * len(blocks)
            streams = [
                _replace_copies(
                    stream,
                    _apply_engram(stream.materialize(), engram, kv, mask),
                )
                for stream, engram, kv, mask in zip(
                    streams, engrams, kvs, masks, strict=True
                )
            ]

        attn_entered = [
            _hc_enter(stream, block.hc_attn, block.attn_norm)
            for block, stream in zip(blocks, streams, strict=True)
        ]
        if not isinstance(blocks[0].attn, CSA2Attention):
            partials = [
                block.attn(h, row, cache)
                for block, (h, _), row, cache in zip(
                    blocks, attn_entered, rows, caches, strict=True
                )
            ]
        else:
            partials = DeepseekV41Block._csa2_partials(
                blocks, [h for h, _ in attn_entered], rows, caches
            )
        attended = ops.allreduce.sum(partials, signal_buffers)
        streams = [
            _with_out(read, h)
            for h, (_, read) in zip(attended, attn_entered, strict=True)
        ]
        return DeepseekV41Block._ffn_parallel(
            blocks,
            streams,
            lambda normed: DeepseekV41MoE.tensor_parallel(
                moes, normed, token_ids, signal_buffers
            ),
        ), [read.residual for _, read in attn_entered]

    @staticmethod
    def data_parallel(
        blocks: Sequence[DeepseekV41Block],
        streams: Sequence[HCStreams],
        rows: Sequence[RaggedRows],
        batch_token_ids: Sequence[TensorValue],
        caches: Sequence[object],
        signal_buffers: Sequence[BufferValue],
        replica_tokens: ReplicaTokens,
        *,
        engram_operands: Sequence[EngramOperand] | None = None,
        engram_token_masks: Sequence[TensorValue | None] | None = None,
    ) -> tuple[list[HCStreams], list[TensorValue]]:
        """Runs :meth:`__call__` with each device on its own requests.

        Engram, attention (with its indexer) and both hyper-connections run
        whole on each device, over its own rows, with no collective. The FFN
        is expert-parallel over the whole batch
        (:meth:`DeepseekV41MoE.data_parallel`).

        Args:
            batch_token_ids: Per device, the whole batch's ``[1, T]`` token
                ids, for the MoE gate.
            replica_tokens: Where each device's rows sit in the batch.

        Every other per-device argument covers only that device's rows.

        Returns:
            Each device's output copies and the copies its attention read.
        """
        moes: list[DeepseekV41MoE | DeepseekV41DSparkMoE] = []
        for block in blocks:
            assert isinstance(block.ffn, (DeepseekV41MoE, DeepseekV41DSparkMoE))
            moes.append(block.ffn)

        if blocks[0].engram_fn is not None:
            if engram_operands is None:
                raise ValueError(
                    f"layer {blocks[0].layer_info.layer_id} is an Engram "
                    "layer but no engram_operands was given"
                )
            engrams: list[Engram] = []
            for block in blocks:
                assert block.engram_fn is not None
                engrams.append(block.engram_fn)
            masks = engram_token_masks or [None] * len(blocks)
            streams = [
                _replace_copies(
                    stream,
                    _apply_engram(
                        stream.materialize(),
                        engram,
                        engram.project(operand),
                        mask,
                    ),
                )
                for stream, engram, operand, mask in zip(
                    streams, engrams, engram_operands, masks, strict=True
                )
            ]

        attn_entered = [
            _hc_enter(stream, block.hc_attn, block.attn_norm)
            for block, stream in zip(blocks, streams, strict=True)
        ]
        streams = [
            _with_out(read, block.attn(h, row, cache))
            for block, (h, read), row, cache in zip(
                blocks, attn_entered, rows, caches, strict=True
            )
        ]
        return DeepseekV41Block._ffn_parallel(
            blocks,
            streams,
            lambda normed: DeepseekV41MoE.data_parallel(
                moes, normed, batch_token_ids, replica_tokens, signal_buffers
            ),
        ), [read.residual for _, read in attn_entered]

    @staticmethod
    def _csa2_partials(
        blocks: Sequence[DeepseekV41Block],
        normed: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[object],
    ) -> list[TensorValue]:
        """Returns each device's CSA2 output-projection partial sum."""
        attns: list[CSA2Attention] = []
        for block in blocks:
            assert isinstance(block.attn, CSA2Attention)
            attns.append(block.attn)
        states = [
            attn.begin(h, row, cache)
            for attn, h, row, cache in zip(
                attns, normed, rows, caches, strict=True
            )
        ]
        return [
            attn.end(state, row)
            for attn, state, row in zip(attns, states, rows, strict=True)
        ]

    @staticmethod
    def _ffn_parallel(
        blocks: Sequence[DeepseekV41Block],
        streams: Sequence[HCStreams],
        moe: Callable[[list[TensorValue]], list[TensorValue]],
    ) -> list[HCStreams]:
        """Runs the FFN half of a multi-device block.

        The attention sublayer's ``pre`` weights merge the FFN's input;
        ``moe`` maps every device's normed input to its FFN output.
        """
        ffn_entered = [
            _hc_enter(stream, block.hc_ffn, block.ffn_norm)
            for block, stream in zip(blocks, streams, strict=True)
        ]
        moved = moe([h for h, _ in ffn_entered])
        return [
            _with_out(read, h)
            for h, (_, read) in zip(moved, ffn_entered, strict=True)
        ]

    def __call__(
        self,
        streams: HCStreams,
        rows: RaggedRows,
        token_ids: TensorValue,
        cache: object | None = None,
        *,
        engram_operand: EngramOperand | None = None,
        engram_token_mask: TensorValue | None = None,
    ) -> tuple[HCStreams, TensorValue]:
        """Runs the block on the stream copies entering it.

        Args:
            streams: The copies entering this block, carrying the previous
                block's FFN ``pre`` and possibly its pending FFN output.
            rows: Ragged-batch bookkeeping, forwarded to ``attn``.
            token_ids: Forwarded to ``ffn``.
            cache: Forwarded to ``attn``; opaque to this block.
            engram_operand: This layer's Engram table rows, gathered for
                these tokens. Required iff ``self.engram_fn is not None``.
            engram_token_mask: ``[s]`` bool; ``False`` turns Engram off at
                that position. ``None`` keeps it on everywhere.

        Returns:
            The block's output copies, carried into the next block or into
            the final merge after the last block, and the ``[1, s, hc, d]``
            copies its attention read.

        Raises:
            ValueError: If this is an Engram layer and ``engram_operand`` is
                missing.
        """
        if self.engram_fn is not None:
            if engram_operand is None:
                raise ValueError(
                    f"layer {self.layer_info.layer_id} is an Engram layer "
                    "but no engram_operand was given"
                )
            streams = _replace_copies(
                streams,
                _apply_engram(
                    streams.materialize(),
                    self.engram_fn,
                    self.engram_fn.project(engram_operand),
                    engram_token_mask,
                ),
            )

        h, attn_read = _hc_enter(streams, self.hc_attn, self.attn_norm)
        streams = _with_out(attn_read, self.attn(h, rows, cache))

        h, ffn_read = _hc_enter(streams, self.hc_ffn, self.ffn_norm)
        return _with_out(ffn_read, self.ffn(h, token_ids)), attn_read.residual


class DeepseekV41Trunk(Module):
    """Embeds tokens, runs the decoder blocks and merges the stream copies.

    Attention, FFN and Engram are built per layer by ``attn_factory``,
    ``ffn_factory`` and ``engram_factory`` (the last only for ``has_engram``
    layers). Holds the target model's layers only, not the drafter's.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef,
        max_seq_len: int,
        *,
        attn_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef, int], Module
        ],
        ffn_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef], Module
        ],
        engram_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef], Engram
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # bf16, not `config.dtype` (the packed quantized dtype): the
        # embedding is stored unquantized.
        self.embed = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=DType.bfloat16,
            device=device,
        )
        self.layer_infos = [
            row for row in resolve_layers(config) if row.side != "drafter"
        ]
        self.layers = LayerList(
            [
                DeepseekV41Block(
                    config,
                    row,
                    device,
                    attn=attn_factory(config, row, device, max_seq_len),
                    ffn=ffn_factory(config, row, device),
                    engram_fn=(
                        engram_factory(config, row, device)
                        if row.has_engram and engram_factory is not None
                        else None
                    ),
                )
                for row in self.layer_infos
            ]
        )

    def __call__(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        *,
        engram_operands: Mapping[int, EngramOperand] | None = None,
        vision: VisionMerge | None = None,
    ) -> TensorValue:
        """Runs the trunk and returns the pre-norm ``[1, s, d]`` stream.

        Args:
            tokens: ``[1, s]`` token ids.
            rows: Ragged-batch bookkeeping, forwarded to every ``attn``.
            cache: Forwarded to every ``attn`` unchanged; opaque here.
            engram_operands: Each Engram layer's table rows gathered for
                these tokens, keyed by ``layer_id``; required for every
                ``has_engram`` layer.
            vision: Image span rows merged into the token embeddings;
                ``None`` for a batch without images.

        Returns:
            ``[1, s, d]`` pre-norm hidden states.
        """
        x, _ = self.forward_with_taps(
            tokens,
            rows,
            cache,
            tap_layer_ids=(),
            engram_operands=engram_operands,
            vision=vision,
        )
        return x

    def forward_with_taps(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        *,
        tap_layer_ids: Sequence[int],
        engram_operands: Mapping[int, EngramOperand] | None = None,
        vision: VisionMerge | None = None,
    ) -> tuple[TensorValue, TensorValue | None]:
        """Runs :meth:`__call__`, also returning hidden states at some layers.

        A tap is the ``[1, s, hc, d]`` stream entering a layer in
        ``tap_layer_ids``, averaged over its copies. ``tap_layer_ids=()``
        adds nothing to the graph. Engram is off at image span positions.

        Returns:
            The pre-norm ``[1, s, d]`` stream, and the taps concatenated in
            layer order, ``[1, s, d * len(tap_layer_ids)]`` (``None`` for no
            taps).

        Raises:
            ValueError: If a tap id is not a trunk layer, repeats, or is an
                Engram layer (its tap would have to follow the Engram step,
                which happens inside the block).
        """
        tap_ids = set(tap_layer_ids)
        if len(tap_ids) != len(tap_layer_ids):
            raise ValueError(f"repeated tap layer ids: {list(tap_layer_ids)}")
        trunk_ids = {row.layer_id for row in self.layer_infos}
        if not tap_ids <= trunk_ids:
            raise ValueError(
                f"tap layer ids {sorted(tap_ids - trunk_ids)} are not trunk "
                "layers"
            )
        taps: list[TensorValue] = []
        streams = first_streams(
            merge_vision(self.embed(tokens), vision), self.config.hc_mult
        )
        engram_token_mask = is_text_token(tokens)
        for layer_info, layer in zip(
            self.layer_infos, self.layers, strict=True
        ):
            tapped = layer_info.layer_id in tap_ids
            if tapped and layer_info.has_engram:
                raise ValueError(
                    f"tap layer {layer_info.layer_id} is an Engram layer"
                )
            operand = None
            if layer_info.has_engram:
                if (
                    engram_operands is None
                    or layer_info.layer_id not in engram_operands
                ):
                    raise ValueError(
                        f"layer {layer_info.layer_id} is an Engram layer "
                        "but engram_operands has no entry for it"
                    )
                operand = engram_operands[layer_info.layer_id]
            streams, entry = layer(
                streams,
                rows,
                tokens,
                cache,
                engram_operand=operand,
                engram_token_mask=engram_token_mask,
            )
            if tapped:
                taps.append(_hc_mean(entry))
        return streams.collapse(), (ops.concat(taps, axis=-1) if taps else None)

    @staticmethod
    def tensor_parallel(
        trunks: Sequence[DeepseekV41Trunk],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        *,
        engram_operands: Sequence[Mapping[int, EngramOperand]],
        visions: Sequence[VisionMerge] | None = None,
    ) -> TensorValue:
        """Runs :meth:`__call__` over one trunk replica per device.

        Every argument but ``signal_buffers`` is per device.

        Returns:
            Device 0's ``[1, s, d]`` stream; every device holds the same one.
        """
        x, _ = DeepseekV41Trunk.tensor_parallel_with_taps(
            trunks,
            tokens,
            rows,
            caches,
            signal_buffers,
            tap_layer_ids=(),
            engram_operands=engram_operands,
            visions=visions,
        )
        return x

    @staticmethod
    def tensor_parallel_with_taps(
        trunks: Sequence[DeepseekV41Trunk],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        *,
        tap_layer_ids: Sequence[int],
        engram_operands: Sequence[Mapping[int, EngramOperand]],
        visions: Sequence[VisionMerge] | None = None,
    ) -> tuple[TensorValue, list[TensorValue] | None]:
        """Runs :meth:`forward_with_taps` over one trunk replica per device.

        Returns:
            Device 0's ``[1, s, d]`` stream and, per device, the identical
            ``[1, s, d * len(tap_layer_ids)]`` taps (``None`` for no taps).
        """
        lead = trunks[0]
        tap_ids = set(tap_layer_ids)
        if len(tap_ids) != len(tap_layer_ids):
            raise ValueError(f"repeated tap layer ids: {list(tap_layer_ids)}")
        trunk_ids = {row.layer_id for row in lead.layer_infos}
        if not tap_ids <= trunk_ids:
            raise ValueError(
                f"tap layer ids {sorted(tap_ids - trunk_ids)} are not trunk "
                "layers"
            )
        taps: list[list[TensorValue]] = [[] for _ in trunks]
        per_device: Sequence[VisionMerge | None] = visions or [None] * len(
            trunks
        )
        streams = [
            first_streams(
                merge_vision(trunk.embed(tok), vision), lead.config.hc_mult
            )
            for trunk, tok, vision in zip(
                trunks, tokens, per_device, strict=True
            )
        ]
        engram_token_masks = [is_text_token(tok) for tok in tokens]
        for i, layer_info in enumerate(lead.layer_infos):
            tapped = layer_info.layer_id in tap_ids
            if tapped and layer_info.has_engram:
                raise ValueError(
                    f"tap layer {layer_info.layer_id} is an Engram layer"
                )
            blocks = []
            for trunk in trunks:
                block = trunk.layers[i]
                assert isinstance(block, DeepseekV41Block)
                blocks.append(block)
            streams, entries = DeepseekV41Block.tensor_parallel(
                blocks,
                streams,
                rows,
                tokens,
                caches,
                signal_buffers,
                engram_operands=(
                    [
                        operands[layer_info.layer_id]
                        for operands in engram_operands
                    ]
                    if layer_info.has_engram
                    else None
                ),
                engram_token_masks=engram_token_masks,
            )
            if tapped:
                for per_dev, entry in zip(taps, entries, strict=True):
                    per_dev.append(_hc_mean(entry))
        return streams[0].collapse(), (
            [ops.concat(per_dev, axis=-1) for per_dev in taps]
            if tap_layer_ids
            else None
        )

    @staticmethod
    def data_parallel_with_taps(
        trunks: Sequence[DeepseekV41Trunk],
        tokens: Sequence[TensorValue],
        batch_tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        replica_tokens: ReplicaTokens,
        *,
        tap_layer_ids: Sequence[int],
        engram_operands: Sequence[Mapping[int, EngramOperand]]
        | Sequence[EngramPending],
        visions: Sequence[VisionMerge] | None = None,
    ) -> tuple[list[TensorValue], list[TensorValue] | None]:
        """Runs :meth:`forward_with_taps` with each device on its own requests.

        ``tokens``, ``rows``, ``caches``, ``engram_operands`` and ``visions``
        (:meth:`VisionMerge.for_rows`) cover each device's own rows;
        ``batch_tokens`` is the whole batch's ``[1, T]`` token ids on every
        device (:meth:`DeepseekV41Block.data_parallel`). Pending operands
        (:meth:`DeepseekV41._engram_operands_data_parallel`) are joined on
        the stream entering the first Engram layer.

        Returns:
            Per device, its own ``[1, t_i, d]`` stream and its
            ``[1, t_i, d * len(tap_layer_ids)]`` taps (``None`` for no taps).
        """
        lead = trunks[0]
        tap_ids = set(tap_layer_ids)
        if len(tap_ids) != len(tap_layer_ids):
            raise ValueError(f"repeated tap layer ids: {list(tap_layer_ids)}")
        trunk_ids = {row.layer_id for row in lead.layer_infos}
        if not tap_ids <= trunk_ids:
            raise ValueError(
                f"tap layer ids {sorted(tap_ids - trunk_ids)} are not trunk "
                "layers"
            )
        taps: list[list[TensorValue]] = [[] for _ in trunks]
        per_device: Sequence[VisionMerge | None] = visions or [None] * len(
            trunks
        )
        streams = [
            first_streams(
                merge_vision(trunk.embed(tok), vision), lead.config.hc_mult
            )
            for trunk, tok, vision in zip(
                trunks, tokens, per_device, strict=True
            )
        ]
        engram_token_masks = [is_text_token(tok) for tok in tokens]
        pending = [p for p in engram_operands if isinstance(p, EngramPending)]
        joined: list[Mapping[int, EngramOperand]] = [
            p for p in engram_operands if not isinstance(p, EngramPending)
        ]
        for i, layer_info in enumerate(lead.layer_infos):
            tapped = layer_info.layer_id in tap_ids
            if tapped and layer_info.has_engram:
                raise ValueError(
                    f"tap layer {layer_info.layer_id} is an Engram layer"
                )
            if layer_info.has_engram and pending:
                # the latest tensor of each device's stream, so the main
                # stream waits for the lookup only after the layers before
                joined = [
                    p.join(
                        stream.out
                        if stream.out is not None
                        else stream.residual
                    )
                    for p, stream in zip(pending, streams, strict=True)
                ]
                pending = []
            blocks = []
            for trunk in trunks:
                block = trunk.layers[i]
                assert isinstance(block, DeepseekV41Block)
                blocks.append(block)
            streams, entries = DeepseekV41Block.data_parallel(
                blocks,
                streams,
                rows,
                batch_tokens,
                caches,
                signal_buffers,
                replica_tokens,
                engram_operands=(
                    [operands[layer_info.layer_id] for operands in joined]
                    if layer_info.has_engram
                    else None
                ),
                engram_token_masks=engram_token_masks,
            )
            if tapped:
                for per_dev, entry in zip(taps, entries, strict=True):
                    per_dev.append(_hc_mean(entry))
        return [stream.collapse() for stream in streams], (
            [ops.concat(per_dev, axis=-1) for per_dev in taps]
            if tap_layer_ids
            else None
        )


def _engram_factory(
    config: DeepseekV41Config, layer_info: LayerInfo, device: DeviceRef
) -> Engram:
    """Builds one Engram layer.

    The Engram tables are not module weights: they stay in pinned host memory
    and are gathered once per forward, outside the layer.
    """
    del layer_info
    n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    # bf16 compute, not the packed `config.dtype`.
    return Engram(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        n_hash_cols=n_hash_cols,
        head_dim=config.engram_head_dim,
        rms_norm_eps=config.rms_norm_eps,
        device=device,
        dtype=DType.bfloat16,
    )


def _csa2_num_comp(config: DeepseekV41Config) -> int:
    """Returns the static ``comp_indices`` width of every CSA2 layer.

    Equals ``config.index_topk``; rows with fewer visible entries are
    ``-1``-padded by the top-k. Defaults to 512 when the field is unset.
    """
    return config.index_topk if config.index_topk is not None else 512


class DeepseekV41(Module):
    """The DeepSeek-V4.1 model: trunk, final norm and LM head.

    ``__call__`` takes ragged-batch bookkeeping (``rows``) because the Engram
    lookup needs each request's positions; :meth:`serve` builds ``rows`` from
    the serving inputs.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef | None = None,
        *,
        engram_host_tables: Mapping[int, tuple[TensorValue, TensorValue]],
        engram_token_map: np.ndarray | None = None,
    ) -> None:
        """Builds the model tree.

        Args:
            config: The resolved model config.
            device: The device this instance's weights/ops live on; defaults
                to ``config.devices[0]``.
            engram_host_tables: This device's pinned, host-resident
                ``(codes, scales)`` table pair per Engram layer id: the full
                tables on one device, else this rank's rows. Pass ``{}`` for
                a config with no Engram layers.
            engram_token_map: The compressed token map derived from the
                tokenizer. Required when ``config`` declares Engram layers;
                ``None`` otherwise.
        """
        super().__init__()
        self.config = config
        if device is None:
            device = config.devices[0]
        self.device = device
        self.ep_batch_manager: EPBatchManager | None = None
        """Shared by the data-parallel replicas' routed experts; its graph
        inputs must be fetched before the forward
        (:meth:`data_parallel_replicas`)."""

        attn_factory, _ = make_csa2_attn_factory(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            window=config.sliding_window,
            num_comp=_csa2_num_comp(config),
        )
        self.trunk = DeepseekV41Trunk(
            config,
            device,
            position_capacity(config),
            attn_factory=attn_factory,
            ffn_factory=nvfp4_moe_ffn_factory,
            engram_factory=_engram_factory,
        )
        self.norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        # bfloat16 by default; ``MAX_DSV41_HEAD_BF16=0`` keeps the head in
        # float32 (upcast on load).
        self.head_bf16 = head_bf16_enabled()
        self.head = Linear(
            config.hidden_size,
            config.vocab_size,
            DType.bfloat16 if self.head_bf16 else DType.float32,
            device,
        )

        self._engram_layout = EngramLayout.from_config(config)
        self._engram_token_map = (
            engram_token_map if self._engram_layout is not None else None
        )
        self._engram_host_tables = engram_host_tables
        self._engram_lookup_cache: EngramLookup | None = None

    def _lm_head(self, x: TensorValue) -> TensorValue:
        """Returns the head's float32 ``[t, vocab]`` logits.

        With a bfloat16 head (the default) the input is rounded to bfloat16
        and the bfloat16 logits are widened to float32. Otherwise the matmul
        runs in float32.
        """
        if self.head_bf16:
            return ops.cast(
                ops.matmul(
                    ops.cast(x, DType.bfloat16),
                    ops.transpose(self.head.weight, 0, 1),
                ),
                DType.float32,
            )
        return ops.matmul(
            ops.cast(x, DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )

    def _engram_lookup(self) -> EngramLookup | None:
        """Returns the Engram ops' builder, or ``None`` without Engram."""
        if self._engram_layout is None:
            return None
        if self._engram_lookup_cache is None:
            assert self._engram_token_map is not None
            self._engram_lookup_cache = EngramLookup(
                self._engram_layout, self._engram_token_map
            )
        return self._engram_lookup_cache

    def _engram_tables(self, lookup: EngramLookup) -> list[TensorValue]:
        """Returns this device's table inputs, layer-major."""
        return [
            table
            for layer_id in lookup.layer_ids
            for table in self._engram_host_tables[layer_id]
        ]

    def _engram_operands(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        lookback_token_ids: TensorValue | None = None,
    ) -> Mapping[int, EngramOperand]:
        """Returns each Engram layer's gathered rows from the full tables.

        ``lookback_token_ids``: see :func:`engram_tokens`.
        """
        lookup = self._engram_lookup()
        if lookup is None:
            return {}
        return lookup.full(
            engram_tokens(
                tokens, rows, lookback_token_ids, lookup.lookback_depth
            ),
            self._engram_tables(lookup),
            self.device,
        )

    @staticmethod
    def _engram_operands_data_parallel(
        replicas: Sequence[DeepseekV41],
        split: ReplicaBatch,
    ) -> list[Mapping[int, EngramOperand]] | list[EngramPending]:
        """Each data-parallel replica's Engram operands, for its own rows.

        With the host lookup (``EngramLookup.host_gather``), the lookups run
        on a side stream and come back pending, for
        :meth:`DeepseekV41Trunk.data_parallel_with_taps` to join on the
        stream entering the first Engram layer; the layers before it
        overlap them.
        """
        inputs = zip(
            replicas, split.tokens, split.rows, split.lookbacks, strict=True
        )
        lookup = replicas[0]._engram_lookup()
        if (
            lookup is None
            or not lookup.host_gather
            or replicas[0].trunk.layer_infos[0].has_engram
        ):
            return [
                replica._engram_operands(tok, row, lookback)
                for replica, tok, row, lookback in inputs
            ]
        return [
            lookup.full_on_side_stream(
                engram_tokens(tok, row, lookback, lookup.lookback_depth),
                replica._engram_tables(lookup),
                replica.device,
            )
            for replica, tok, row, lookback in inputs
        ]

    @staticmethod
    def _engram_operands_tensor_parallel(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        lookback_token_ids: Sequence[TensorValue] | None,
        signal_buffers: Sequence[BufferValue],
    ) -> list[Mapping[int, EngramOperand]]:
        """Runs :meth:`_engram_operands` over replicas holding table shards.

        Each rank gathers its hash columns from its own rows, then the ranks
        all-gather and assemble the full operands on every device.
        """
        lookup = replicas[0]._engram_lookup()
        if lookup is None:
            return [{} for _ in replicas]
        lookbacks: Sequence[TensorValue | None] = (
            lookback_token_ids
            if lookback_token_ids is not None
            else [None] * len(replicas)
        )
        return list(
            lookup.sharded(
                [
                    engram_tokens(tok, row, lookback, lookup.lookback_depth)
                    for tok, row, lookback in zip(
                        tokens, rows, lookbacks, strict=True
                    )
                ],
                [replica._engram_tables(lookup) for replica in replicas],
                [replica.device for replica in replicas],
                signal_buffers,
            )
        )

    def __call__(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        engram_lookback_ids: TensorValue | None = None,
        vision: VisionMerge | None = None,
    ) -> TensorValue:
        """Runs the trunk and returns the pre-norm ``[1, s, d]`` stream.

        ``engram_lookback_ids``: see :func:`engram_tokens`.
        """
        operands = self._engram_operands(tokens, rows, engram_lookback_ids)
        return self.trunk(
            tokens, rows, cache, engram_operands=operands, vision=vision
        )

    def serve(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        cache: object,
        engram_lookback_ids: TensorValue | None = None,
        vision: VisionMerge | None = None,
    ) -> DecoderOutputs:
        """Builds the serving graph body: ragged tokens in, logits out.

        ``engram_lookback_ids``: see :func:`engram_tokens`.

        Raises:
            TypeError: If ``cache`` is not a :class:`CSA2Cache`.
        """
        if not isinstance(cache, CSA2Cache):
            raise TypeError(
                f"DeepseekV41.serve requires a CSA2Cache, got {type(cache)!r}"
            )
        t = tokens.shape[0]
        starts = ops.cast(cache.swa_collection.cache_lengths, DType.int32)
        rows = RaggedRows.from_offsets(input_row_offsets, t, starts)
        x = self(
            ops.reshape(tokens, [1, t]),
            rows,
            cache,
            engram_lookback_ids=engram_lookback_ids,
            vision=vision,
        )
        return logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            input_row_offsets,
            return_n_logits,
            norm=self.norm,
            lm_head=self._lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
        )

    def tensor_parallel_replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
    ) -> list[Self]:
        """Builds one copy of the model per device, for the multi-device graph.

        Each copy's weights are shards of this model's, which stays the
        source the weights are loaded into. Attention (with its indexer)
        splits heads and the MoE splits routed experts. Engram's projection
        splits columns, and each replica reads only its rank's rows of the
        host tables, so table traffic splits across ranks. All else is
        replicated. Call once, after :meth:`load_state_dict`.

        Args:
            devices: One device per replica.
            engram_host_tables: One ``engram_host_tables`` mapping (as in
                the constructor) per device, in ``devices`` order: rank
                ``i``'s rows of each table.

        Returns:
            The per-device replicas.
        """
        return self._replicas(devices, engram_host_tables, split_attention=True)

    def data_parallel_replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
    ) -> list[Self]:
        """Builds one copy of the model per device for data parallelism.

        Like :meth:`tensor_parallel_replicas`, but only the MoE splits its
        routed experts. Attention, its indexer and Engram's projection stay
        whole, since each device runs them on its own requests, and every
        replica reads the full Engram tables. With ``config.ep_config``,
        the replicas' routed experts share one :class:`EPBatchManager`
        (:attr:`ep_batch_manager`). Call once, after :meth:`load_state_dict`.

        Args:
            devices: One device per replica.
            engram_host_tables: Per device, in ``devices`` order, the full
                ``engram_host_tables`` mapping (as in the constructor).

        Returns:
            The per-device replicas.
        """
        return self._replicas(
            devices, engram_host_tables, split_attention=False
        )

    def _replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
        *,
        split_attention: bool,
    ) -> list[Self]:
        """Builds the per-device copies, splitting the routed experts.

        ``split_attention`` also splits attention heads (with the indexer's)
        and Engram's projection columns.
        """
        n = len(devices)
        for block in self.trunk.layers:
            assert isinstance(block, DeepseekV41Block)
            assert isinstance(block.attn, CSA2Attention)
            assert isinstance(block.ffn, DeepseekV41MoE)
            block.ffn.shard_experts(n)
            if split_attention:
                block.attn.shard_heads(n)
                if block.engram_fn is not None:
                    block.engram_fn.shard_columns(n)
        # `type(self)` so subclasses replicate their extra submodules too.
        replicas = [
            type(self)(
                self.config,
                device,
                engram_host_tables=tables,
                engram_token_map=self._engram_token_map,
            )
            for device, tables in zip(devices, engram_host_tables, strict=True)
        ]
        _adopt_shards(self, replicas, list(devices))
        manager = (
            EPBatchManager(self.config.ep_config)
            if not split_attention and self.config.ep_config is not None
            else None
        )
        for rank, replica in enumerate(replicas):
            replica.ep_batch_manager = manager
            for block in replica.trunk.layers:
                assert isinstance(block, DeepseekV41Block)
                assert isinstance(block.attn, CSA2Attention)
                assert isinstance(block.ffn, DeepseekV41MoE)
                if split_attention:
                    block.attn.keep_local_heads(n)
                block.ffn.keep_local_experts(n, rank)
                block.ffn.ep_batch_manager = manager
        return replicas

    @staticmethod
    def serve_tensor_parallel(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        engram_lookback_ids: Sequence[TensorValue] | None = None,
        visions: Sequence[VisionMerge] | None = None,
    ) -> DecoderOutputs:
        """Runs :meth:`serve` with one replica per device.

        ``tokens``, ``input_row_offsets``, ``caches``,
        ``engram_lookback_ids`` and ``visions`` are per device; logits come
        from device 0.
        """
        lead = replicas[0]
        t = tokens[0].shape[0]
        rows = [
            RaggedRows.from_offsets(
                offsets,
                t,
                ops.cast(cache.swa_collection.cache_lengths, DType.int32),
            )
            for offsets, cache in zip(input_row_offsets, caches, strict=True)
        ]
        tokens_2d = [ops.reshape(tok, [1, t]) for tok in tokens]
        x = DeepseekV41Trunk.tensor_parallel(
            [replica.trunk for replica in replicas],
            tokens_2d,
            rows,
            caches,
            signal_buffers,
            engram_operands=DeepseekV41._engram_operands_tensor_parallel(
                replicas, tokens_2d, rows, engram_lookback_ids, signal_buffers
            ),
            visions=visions,
        )
        return logits_postprocess(
            ops.reshape(x, [t, lead.config.hidden_size]),
            input_row_offsets[0],
            return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=lead.config.return_logits,
            return_hidden_states=lead.config.return_hidden_states,
        )

    @staticmethod
    def split_replicas(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        caches: Sequence[CSA2Cache],
        engram_lookback_ids: Sequence[TensorValue] | None = None,
        *,
        prefix: str = "",
    ) -> ReplicaBatch:
        """Cuts a replica-major batch into each device's own requests.

        ``tokens``, ``input_row_offsets`` and ``engram_lookback_ids`` hold the
        whole batch on every device; ``caches`` are per device, each covering
        its replica's requests. ``host_input_row_offsets`` (the whole
        batch's) and ``data_parallel_splits`` (``[n + 1]`` request
        boundaries of the replicas) are on CPU. ``prefix`` names the split
        dims, distinct per call within one graph.
        """
        n = len(replicas)
        total = tokens[0].shape[0]
        host_offsets = ops.cast(host_input_row_offsets, DType.int64)
        local_tokens, local_offsets = split_batch_replicated(
            [replica.device for replica in replicas],
            list(tokens),
            list(input_row_offsets),
            host_offsets,
            data_parallel_splits,
            prefix=prefix,
        )
        rows = [
            RaggedRows.from_offsets(
                offsets,
                tok.shape[0],
                ops.cast(cache.swa_collection.cache_lengths, DType.int32),
            )
            for tok, offsets, cache in zip(
                local_tokens, local_offsets, caches, strict=True
            )
        ]
        lookbacks: list[TensorValue | None] = [None] * n
        if engram_lookback_ids is not None:
            lookbacks = [
                ops.slice_tensor(
                    lookback,
                    [
                        (
                            slice(
                                data_parallel_splits[i],
                                data_parallel_splits[i + 1],
                            ),
                            f"{prefix}_engram_lookback_split_{i}",
                        ),
                        slice(None),
                    ],
                )
                for i, lookback in enumerate(engram_lookback_ids)
            ]
        return ReplicaBatch(
            tokens=[
                ops.reshape(tok, [1, tok.shape[0]]) for tok in local_tokens
            ],
            batch_tokens=[ops.reshape(tok, [1, total]) for tok in tokens],
            rows=rows,
            replica_tokens=ReplicaTokens.from_splits(
                host_offsets, data_parallel_splits, total, n
            ),
            lookbacks=lookbacks,
        )

    @staticmethod
    def serve_data_parallel(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        return_n_logits: TensorValue,
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        engram_lookback_ids: Sequence[TensorValue] | None = None,
        visions: Sequence[VisionMerge] | None = None,
    ) -> DecoderOutputs:
        """Runs :meth:`serve` with each device on its own replica's requests.

        The arguments are as in :meth:`split_replicas`; ``visions`` hold the
        whole batch's image rows on every device, and each device merges its
        own. The streams are gathered onto device 0, which computes the
        logits for the whole batch.
        """
        lead = replicas[0]
        split = DeepseekV41.split_replicas(
            replicas,
            tokens,
            input_row_offsets,
            host_input_row_offsets,
            data_parallel_splits,
            caches,
            engram_lookback_ids,
        )
        xs, _ = DeepseekV41Trunk.data_parallel_with_taps(
            [replica.trunk for replica in replicas],
            split.tokens,
            split.batch_tokens,
            split.rows,
            caches,
            signal_buffers,
            split.replica_tokens,
            tap_layer_ids=(),
            engram_operands=DeepseekV41._engram_operands_data_parallel(
                replicas, split
            ),
            visions=(
                [
                    vision.for_rows(
                        offsets,
                        data_parallel_splits[i],
                        data_parallel_splits[i + 1],
                    )
                    for i, (vision, offsets) in enumerate(
                        zip(visions, input_row_offsets, strict=True)
                    )
                ]
                if visions is not None
                else None
            ),
        )
        return logits_postprocess(
            gather_rows(xs, split.replica_tokens.total, signal_buffers),
            input_row_offsets[0],
            return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=lead.config.return_logits,
            return_hidden_states=lead.config.return_hidden_states,
        )
