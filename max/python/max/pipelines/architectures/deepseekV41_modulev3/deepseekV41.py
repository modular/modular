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
"""The DeepSeek-V4.1 text trunk, tensor or data parallel over one mesh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max import tree
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.data_parallelism import split_batch_replicated
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.architectures.deepseekV41.engram_lookback import (
    read_engram_lookback,
)
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
    Unknown,
)

if TYPE_CHECKING:
    from transformers import AutoConfig

from .interface import V41Config, extra_input_names
from .layer_resolver import resolve
from .layers.attention import RaggedRows, absolute_positions, row_ids
from .layers.block import DecoderLayer, leaf_values, slot_constants
from .layers.engram import Engram
from .layers.mhc import mhc_final_collapse, mhc_identity_writeback, mhc_post
from .layers.norm import RMSNorm
from .layers.rope import RopeParams
from .model_config import DeepseekV41Config

__all__ = ["Batch", "DeepseekV41", "validate_dspark_taps"]


def validate_dspark_taps(config: DeepseekV41Config) -> tuple[int, ...]:
    """The tap layer ids to capture, asserted legal for this trunk.

    The ``mtp.*`` weights were trained against exactly
    ``dspark_target_layer_ids``'s entering streams, so any other choice is a
    different drafter, never a setting. A tap must be a trunk layer, appear
    once, and sit outside ``engram_layer_ids`` -- an Engram layer's tap
    would have to follow the Engram step, which happens inside the block.

    Returns:
        The tap ids, in ``dspark_target_layer_ids`` order.

    Raises:
        ValueError: On a repeated, non-trunk, or Engram tap id.
    """
    taps = tuple(config.dspark_target_layer_ids)
    if len(set(taps)) != len(taps):
        raise ValueError(f"repeated tap layer ids: {list(taps)}")
    trunk = set(range(config.num_hidden_layers))
    if not set(taps) <= trunk:
        raise ValueError(
            f"tap layer ids {sorted(set(taps) - trunk)} are not trunk layers"
        )
    engram = set(config.engram_layer_ids)
    for tap in taps:
        if tap in engram:
            raise ValueError(
                f"tap layer {tap} is an Engram layer (its tap would have "
                "to follow the Engram step, which happens inside the block)"
            )
    return taps


def _hc_mean(stream: Tensor) -> Tensor:
    """Averages ``[rows, hc, hidden]`` over the copies, accumulating f32.

    The drafter's ``main_proj`` was trained on the hc-mean of the entering
    stream, never on copy 0 (the V4 precedent's trap: taking copy 0 gives
    the right shape and the wrong numbers).
    """
    return F.squeeze(F.mean(stream.cast(DType.float32), axis=1), 1).cast(
        stream.dtype
    )


def _capture_tap(
    y: Tensor, stream: Tensor, post: Tensor, comb: Tensor, hc_mult: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One tap layer's capture: ``(tap, stream, y, post, comb)``.

    The stream entering a layer is deferred here: ``y`` (the previous
    layer's feed-forward output) is still unwritten, described by ``post``
    and ``comb``. The drafter's tap is the *written-back* stream, so the
    pending boundary is materialized first -- the same ``mhc_post`` the
    Engram path performs -- the tap is the hc-mean of that, and the
    write-back is reset to identity so the layer body's own boundary does
    not write ``y`` in a second time.

    Returns:
        ``(tap, materialized_stream, y, post, comb)`` where the last three
        are the identity write-back of the materialized stream.
    """
    stream = mhc_post(
        y,
        stream,
        post,
        comb.reshape([stream.shape[0], hc_mult, hc_mult]),
        hc_mult,
    )
    tap = _hc_mean(stream)
    reset_y, reset_post, reset_comb = mhc_identity_writeback(stream)
    return tap, stream, reset_y, reset_post, reset_comb


def _split_replicated_batch(
    t: Tensor,
    input_row_offsets: Tensor,
    input_row_offsets_i64: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
) -> tuple[Tensor, Tensor]:
    """Splits a replicated ragged batch into DP shards (tokens, offsets).

    The DeepseekV3 ModuleV3 pattern: each rank cuts its requests' rows out
    of its own replicated copy, so no rank receives another's data.
    """
    devices = [DeviceRef.from_device(d) for d in mapping.mesh.devices]
    t_shards, offsets_shards = split_batch_replicated(
        devices,
        [TensorValue(shard) for shard in t.local_shards],
        [TensorValue(shard) for shard in input_row_offsets.local_shards],
        TensorValue(input_row_offsets_i64),
        TensorValue(data_parallel_splits),
    )
    return (
        Tensor.from_shard_values(t_shards, mapping),
        Tensor.from_shard_values(offsets_shards, mapping),
    )


def _slice_requests(
    t: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
    prefix: str,
) -> Tensor:
    """Slices a replicated per-request tensor into each rank's requests.

    ``t`` is ``[batch, ...]`` over the DP mesh; rank ``r`` keeps rows
    ``[splits[r], splits[r + 1])`` -- the same request ranges the token
    split uses, cut from its own replicated copy.
    """
    splits = TensorValue(data_parallel_splits)
    shards = []
    for r, shard in enumerate(t.local_shards):
        shards.append(
            ops.slice_tensor(
                TensorValue(shard),
                [
                    (
                        slice(splits[r], splits[r + 1]),
                        f"{prefix}_requests_r{r}",
                    ),
                    ...,
                ],
            )
        )
    return Tensor.from_shard_values(shards, mapping)


def _per_rank_ragged_rows(
    row_offsets: Tensor,
    cache_lengths: Tensor,
    tokens: Tensor,
    mapping: DeviceMapping,
) -> RaggedRows:
    """Ragged rows per rank, from its own rebased offsets and lengths.

    Each rank's cache-lengths shard is its own replica's record (the KV
    inputs are per-device), so the per-rank rows are the same
    single-device computation the tensor-parallel trunk runs once.
    """
    shards = [
        RaggedRows(
            offsets,
            *row_ids(offsets, shard.shape[0]),
            absolute_positions(offsets, lengths, shard.shape[0]),
            F.cast(offsets[1:] - offsets[:-1], DType.int32),
        )
        for offsets, lengths, shard in zip(
            row_offsets.local_shards,
            cache_lengths.local_shards,
            tokens.local_shards,
            strict=True,
        )
    ]
    return RaggedRows(
        offsets=Tensor.from_shard_values([s.offsets for s in shards], mapping),
        bid=Tensor.from_shard_values([s.bid for s in shards], mapping),
        local=Tensor.from_shard_values([s.local for s in shards], mapping),
        positions=Tensor.from_shard_values(
            [s.positions for s in shards], mapping
        ),
        lengths=Tensor.from_shard_values([s.lengths for s in shards], mapping),
    )


def _tie_shape(
    t: TensorValue, like: TensorValue, *, offset: int = 0
) -> TensorValue:
    """Rebinds ``t``'s shape onto ``like``'s, plus ``offset`` on axis 0.

    Each rank's sliced fields carry independently-created symbolic dims
    (per-slice labels), so the engram kernel's host-side shape checks --
    ``lookback_dead`` matching ``lookback``, the row offsets holding
    ``lookback``'s requests plus one -- see unequal dims where the values
    agree. The rebind asserts the tie per rank.
    """
    target = like.shape if offset == 0 else [like.shape[0] + offset]
    return ops.rebind(t, target)


_tie_shape_op = F.functional(_tie_shape)


def _dp_axis_name(mesh: DeviceMesh) -> str:
    """The single mesh axis's name, ``dp`` under data parallelism."""
    return mesh.axis_names[0]


def _host_total_rows(
    splits: TensorValue, row_bounds: TensorValue, *, last: int
) -> TensorValue:
    """The batch's total token rows as a host-resident 0-d int64 scalar.

    ``splits[last]`` is the global request count (the DP split pair's last
    boundary) and ``row_bounds`` holds the cumulative row offsets, so the
    entry there is the last request's end row: the total every rank's shard
    rows sum to, without a device read. Both inputs are CPU-resident, which
    is what makes the grouped-MoE tile-regime estimate host arithmetic the
    launch reads in place instead of a device-to-host materialization.
    """
    return row_bounds[splits[last]]


_host_total_rows_op = F.functional(_host_total_rows)


def _replay_decoder_start(config: DeepseekV41Config) -> int:
    """Returns the first layer that runs on the kept rows, after checking it.

    The last KV source projects the decoder's global KV for every row, so the
    layers after it are the ones that can run on kept rows only.

    Raises:
        ValueError: If there is no layer after the last KV source, an Engram
            layer or a DSpark tap sits in the decoder (before it, every row
            is still live), or the graph returns per-row outputs, which
            would silently cover the kept rows only.
    """
    if not config.kv_source_layer_ids:
        raise ValueError("decoder replay needs at least one KV source layer")
    decoder_start = max(config.kv_source_layer_ids) + 1
    if decoder_start >= config.num_hidden_layers:
        raise ValueError(
            f"decoder replay needs a layer after the last KV source "
            f"({decoder_start - 1}); the trunk has {config.num_hidden_layers}"
        )
    if any(layer >= decoder_start for layer in config.engram_layer_ids):
        raise ValueError(
            "decoder replay needs every Engram layer before layer "
            f"{decoder_start}, the first decoder layer"
        )
    if any(layer < decoder_start for layer in config.dspark_target_layer_ids):
        raise ValueError(
            "decoder replay needs every DSpark tap in the decoder (layer "
            f">= {decoder_start}); got {config.dspark_target_layer_ids}"
        )
    if (
        config.return_logits == ReturnLogits.ALL
        or config.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED
    ):
        raise ValueError(
            "decoder replay returns the kept rows only; all-row logits and "
            "hidden states need every row"
        )
    return decoder_start


def _host_plan_rows(
    splits: TensorValue, host_kept: TensorValue, *, last: int
) -> TensorValue:
    """The kept rows' global count as a host int64 scalar.

    ``host_kept`` is the plan's cumulative kept bounds and ``splits[last]`` the
    global request count, so the entry there is the total, read the way
    :func:`_host_total_rows` reads the full batch's.
    """
    return host_kept[splits[last]]


_host_plan_rows_op = F.functional(_host_plan_rows)


@dataclass(frozen=True)
class ReplayRows:
    """The rows the decoder keeps past the last KV source, host-planned.

    Each request keeps a suffix of its merged rows, so the kept rows stay
    contiguous per request. Positions and cache slots stay absolute: a
    skipped row is never recomputed, and the kept rows' leaves read through
    :attr:`starts`. Under data parallelism every field is per rank, in that
    rank's own row frame.
    """

    rows: RaggedRows
    """The kept rows' ragged bookkeeping, at the full batch's positions."""
    indices: Tensor
    """``[kept]`` int32 source row of each kept row in the full batch."""
    starts: Tensor
    """``[batch]`` int32 absolute position of each request's first kept row."""
    swa_begin: Tensor
    """``[batch]`` int32 window bound, raised to each trimmed request's start."""
    longest: Tensor
    """``[1]`` uint32 host: the exact longest kept row count on this device."""
    host_offsets: Tensor | None = None
    """Under data parallelism, the CPU int64 kept bounds the MoE row count
    reads."""


@dataclass
class Batch:
    """One forward's staged inputs, shared by every layer.

    Under data parallelism (``dp > 1``) the per-row and per-request fields
    hold each rank's own shard: its requests' token rows, rebased row
    offsets and request-sliced extras. ``comm`` carries the EP communication
    buffers the MoE dispatch/combine reads, and the ``tokens_full`` /
    ``row_offsets_full`` pair keeps the pre-split replicated batch the
    vocab-parallel embedding reads before the split.

    ``draft_width`` is the step's draft rows appended per request, ``None``
    without a spec decoder. Nothing on the trunk reads it: it rides along for
    the spec adapter's merged-window arithmetic.

    ``replay`` holds the decoder's kept rows when the export plans them;
    ``rows`` always describes the full batch the layers up to the last KV
    source run on.
    """

    tokens: Tensor
    rows: RaggedRows
    swa_begin: Tensor
    global_write_floor: Tensor
    image_token_mask: Tensor
    lookback: Tensor
    lookback_dead: Tensor
    tables: list[Tensor]
    kv: dict[str, PagedCacheValues]
    comm: EPCommBuffers | None = None
    tokens_full: Tensor | None = None
    row_offsets_full: Tensor | None = None
    data_parallel_splits: Tensor | None = None
    input_row_offsets_i64: Tensor | None = None
    draft_width: Tensor | None = None
    replay: ReplayRows | None = None

    @property
    def row_offsets(self) -> Tensor:
        """The ragged offsets of the rows the trunk's outputs hold.

        With replay that is the kept rows: the hidden states, taps and logits
        come out indexed by them, so an output gather must not read ``rows``.
        """
        return self.output_rows.offsets

    @property
    def positions(self) -> Tensor:
        """Each output row's absolute position, ``[rows]``."""
        return self.output_rows.positions

    @property
    def output_rows(self) -> RaggedRows:
        """The rows the decoder, and so every trunk output, runs on."""
        return self.rows if self.replay is None else self.replay.rows


class DeepseekV41(Module[..., ModuleV3Outputs]):
    """Runs all live rows through the text decoder, over one device mesh.

    With ``data_parallel_degree > 1`` the mesh axis is a data-parallel
    axis: every rank runs the full attention heads on its own request shard
    (the token split happens in-graph from ``data_parallel_splits``), the
    routed experts stay expert-sharded with a P2P dispatch/combine, and no
    residual-path all-reduce runs anywhere.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        kv_params: MultiKVCacheParams,
        mesh: DeviceMesh,
        *,
        text_config: AutoConfig,
        token_map: np.ndarray,
        data_parallel_degree: int = 1,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_params = kv_params
        self.mesh = mesh
        self.dp = data_parallel_degree
        self.ep_batch_manager = ep_batch_manager
        if self.dp > 1 and mesh.num_devices != self.dp:
            raise ValueError(
                f"data_parallel_degree {self.dp} must equal the mesh size "
                f"({mesh.num_devices}); DP attention runs TP=1 per rank."
            )
        if self.dp > 1 and ep_batch_manager is None:
            # The routed experts shard over the same DP mesh, so a
            # tensor-parallel MoE would reduce across ranks that hold
            # different requests. DP attention is only wired with the
            # expert-parallel MoE.
            raise ValueError(
                "data-parallel attention requires the expert-parallel MoE: "
                "pass the EP batch manager (ep_size must equal the device "
                "count)."
            )
        if self.dp > 1 and (
            config.return_logits != ReturnLogits.LAST_TOKEN
            or config.return_hidden_states != ReturnHiddenStates.NONE
        ):
            # The all-row and variable logit paths (and the hidden-state
            # paths that read the same gathers) are wired for the
            # replicated rows of tensor parallelism; under DP they would
            # silently cover one rank's shard only.
            raise ValueError(
                "data parallelism serves next-token logits only; the "
                "all-row and variable logit outputs and the hidden-state "
                "outputs stay tensor-parallel."
            )
        self.interface_config = V41Config.from_hf(
            {"text_config": vars(text_config)}
        )
        self.decoder_replay = self.interface_config.decoder_replay
        self.decoder_start = (
            _replay_decoder_start(config) if self.decoder_replay else None
        )
        rope = RopeParams.from_text_config(text_config)
        with default_device(mesh), default_dtype(DType.bfloat16):
            # The vocab-sharded embedding and head split their weight over
            # the mesh axis by name; under DP that axis is ``dp``.
            self.embed = VocabParallelEmbedding(
                config.vocab_size,
                dim=config.hidden_size,
                tp_axis=_dp_axis_name(mesh) if self.dp > 1 else None,
            )
            layers = []
            # The resolver also resolves the DSpark stages past the trunk
            # (their rows share the SWA leaf); the trunk runs its own
            # layers only.
            for info in resolve(config, include_drafter=False):
                engram = None
                if info.layer_id in config.engram_layer_ids:
                    engram = Engram(
                        layer_ids=tuple(config.engram_layer_ids),
                        num_embeddings=tuple(config.engram_num_embeddings),
                        max_ngram_size=int(text_config.engram_max_ngram_size),
                        n_heads=int(text_config.engram_n_heads),
                        head_dim=int(text_config.engram_head_dim),
                        compressed_vocab_size=int(
                            text_config.engram_compressed_vocab_size
                        ),
                        engram_vocab_size=int(text_config.engram_vocab_size),
                        hidden_size=config.hidden_size,
                        hc_mult=config.hc_mult,
                        pad_token_id=int(text_config.engram_pad_token_id),
                        rms_norm_eps=config.rms_norm_eps,
                        device=mesh,
                        token_map=token_map,
                    )
                layers.append(
                    DecoderLayer(
                        config,
                        info,
                        mesh,
                        rope=rope,
                        max_position=config.max_seq_len,
                        candidate_block_size=config.candidate_block_size,
                        engram=engram,
                        ep_batch_manager=ep_batch_manager,
                        data_parallel=self.dp > 1,
                    )
                )
            self.layers = ModuleList(layers)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                tp_axis=_dp_axis_name(mesh) if self.dp > 1 else None,
            )

    def batch_from_parts(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv: Mapping[str, PagedCacheValues],
        extras: Mapping[str, Tensor],
        *,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: EPCommBuffers | None = None,
        draft_width: Tensor | None = None,
    ) -> Batch:
        """Stages already-split inputs onto the mesh, as a :class:`Batch`.

        ``forward`` unflattens the graph's variadic tail into per-leaf
        caches and named extras and lands here; the DSpark target adapter
        (:mod:`.spec_adapters`) starts from the
        driver's per-leaf caches instead and calls this directly, so the
        two paths cannot drift apart.

        Under data parallelism (``self.dp > 1``) the DP split inputs are
        required and every per-row/per-request field comes out per rank.

        Raises:
            ValueError: If ``self.dp > 1`` and either half of the DP split
                pair is missing.
        """
        mesh = self.mesh
        tokens = F.distributed_broadcast(tokens, mesh)
        offsets = F.distributed_broadcast(input_row_offsets, mesh)
        swa_begin = F.distributed_broadcast(extras["swa_begin"], mesh)
        floor = F.distributed_broadcast(extras["global_write_floor"], mesh)
        image_token_mask = F.distributed_broadcast(
            extras["image_token_mask"], mesh
        )
        if self.interface_config.engram_lookback_history:
            # The overlap export replaces the host window with the window
            # plus its device history: the row the source map names reads
            # the window the previous step recorded in-graph, because the
            # host does not know that step's accepted drafts yet. The
            # Engram hash reads the same int32 window either way.
            history = extras["engram_lookback_history"].graph_values[0]
            assert isinstance(history, BufferValue)
            lookback = F.cast(
                read_engram_lookback(
                    extras["engram_lookback_ids"],
                    extras["engram_lookback_source"],
                    history,
                ),
                DType.int32,
            )
        else:
            lookback = extras["engram_lookback"]
        lookback = F.distributed_broadcast(lookback, mesh)
        lookback_dead = F.distributed_broadcast(
            extras["engram_lookback_dead"], mesh
        )
        tables = [
            extras[f"engram_table_{i}_{part}"]
            for i in range(len(self.config.engram_layer_ids))
            for part in ("weight", "scale")
        ]
        if self.dp > 1:
            if data_parallel_splits is None or input_row_offsets_i64 is None:
                raise ValueError(
                    "data parallelism requires the data_parallel_splits and "
                    "input_row_offsets_i64 graph inputs."
                )
            shard_mapping = DeviceMapping(mesh, (Sharded(0),))
            tokens_full, offsets_full = tokens, offsets
            tokens, offsets = _split_replicated_batch(
                tokens,
                offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                shard_mapping,
            )
            # The image mask is per step token (the router reads it per
            # row), so it splits with the tokens.
            image_token_mask, _ = _split_replicated_batch(
                image_token_mask,
                offsets_full,
                input_row_offsets_i64,
                data_parallel_splits,
                shard_mapping,
            )
            swa_begin = _slice_requests(
                swa_begin, data_parallel_splits, shard_mapping, "swa_begin"
            )
            floor = _slice_requests(
                floor, data_parallel_splits, shard_mapping, "write_floor"
            )
            lookback = _slice_requests(
                lookback, data_parallel_splits, shard_mapping, "lookback"
            )
            lookback_dead = _slice_requests(
                lookback_dead,
                data_parallel_splits,
                shard_mapping,
                "lookback_dead",
            )
            # The engram kernel's shape checks compare symbolic dims, and
            # each sliced field carries its own; tie the companion's dims
            # onto ``lookback``'s, per rank (the rule-less tie drops the
            # placement, so re-claim the row sharding).
            lookback_dead = _tie_shape_op(
                lookback_dead, lookback
            ).rebind_mapping(lookback.mapping)
            rows = _per_rank_ragged_rows(
                offsets,
                kv["swa"].cache_lengths,
                tokens,
                shard_mapping,
            )
        else:
            positions = absolute_positions(
                offsets, kv["swa"].cache_lengths, tokens.shape[0]
            )
            bid, local = row_ids(offsets, tokens.shape[0])
            rows = RaggedRows(
                offsets,
                bid,
                local,
                positions,
                F.cast(offsets[1:] - offsets[:-1], DType.int32),
            )
        replay = (
            self._replay_rows(
                rows,
                swa_begin,
                kv["swa"].cache_lengths,
                extras,
                data_parallel_splits,
            )
            if self.decoder_replay
            else None
        )
        return Batch(
            tokens=tokens,
            rows=rows,
            swa_begin=swa_begin,
            global_write_floor=floor,
            image_token_mask=image_token_mask,
            lookback=lookback,
            lookback_dead=lookback_dead,
            tables=tables,
            kv=dict(kv),
            comm=comm_buffers,
            tokens_full=(tokens_full if self.dp > 1 else None),
            row_offsets_full=(offsets_full if self.dp > 1 else None),
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
            draft_width=draft_width,
            replay=replay,
        )

    def _rebased(
        self, skip: Tensor, cache_lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Each request's first kept position and its window floor.

        Both derive from the ``cache_lengths`` the full graph positions its
        rows with, so a kept row sits where the full graph puts it even when
        the host cannot know that position (a row in flight, whose device
        position runs ahead of the host's by its accepted drafts). A trimmed
        request's floor is its own start, since no window KV below it was
        written this step; an untrimmed request's is its ordinary window
        visibility.
        """
        lengths = F.rebind(
            F.cast(cache_lengths, DType.int32),
            [skip.shape[0]],
            "replay: one cache length per request",
        )
        start = lengths + skip
        floor = F.where(
            skip > 0,
            start,
            F.elementwise_max(lengths - self.config.sliding_window, 0),
        )
        return start, floor

    def _replay_rows(
        self,
        rows: RaggedRows,
        swa_begin: Tensor,
        cache_lengths: Tensor,
        extras: Mapping[str, Tensor],
        data_parallel_splits: Tensor | None,
    ) -> ReplayRows:
        """Stages the host-planned kept rows onto the mesh.

        The host stages the kept row indices, their cumulative bounds and
        each request's front skip. Under data parallelism the kept rows are
        a ragged batch of their own, so each rank cuts its requests' share
        with the same split the token rows ride; the per-rank selection is
        host arithmetic and never compiles into the graph.
        """
        mesh = self.mesh
        kept_offsets = F.distributed_broadcast(
            extras["replay.kept_offsets"], mesh
        )
        skip = F.distributed_broadcast(extras["replay.kept_skip"], mesh)
        indices = F.distributed_broadcast(extras["replay.row_indices"], mesh)
        longest = extras["replay.rank_longest"]
        if self.dp == 1:
            start, floor = self._rebased(skip, cache_lengths)
            kept_off = F.rebind(
                kept_offsets,
                [rows.offsets.shape[0]],
                "replay: one kept bound per request",
            )
            bid, local = row_ids(kept_off, indices.shape[0])
            return ReplayRows(
                rows=RaggedRows(
                    offsets=kept_off,
                    bid=bid,
                    local=local,
                    positions=F.gather(rows.positions, indices, axis=0),
                    lengths=F.cast(kept_off[1:] - kept_off[:-1], DType.int32),
                ),
                indices=indices,
                starts=start,
                swa_begin=F.elementwise_max(swa_begin, floor),
                longest=longest,
            )
        assert data_parallel_splits is not None
        mapping = DeviceMapping(mesh, (Sharded(0),))
        skip = _slice_requests(
            skip, data_parallel_splits, mapping, "replay_skip"
        )
        # One split label per extent: the kept rows are fewer than the token
        # rows, so sharing the token split's labels would bind one symbol to
        # two different extents.
        row_shards, kept_off_shards = split_batch_replicated(
            [DeviceRef.from_device(d) for d in mesh.devices],
            [TensorValue(shard) for shard in indices.local_shards],
            [TensorValue(shard) for shard in kept_offsets.local_shards],
            TensorValue(extras["replay.host_kept_offsets"]),
            TensorValue(data_parallel_splits),
            "replay_rows",
        )
        bases = F.distributed_broadcast(extras["replay.rank_bases"], mesh)
        kept: list[RaggedRows] = []
        sources: list[Tensor] = []
        starts: list[Tensor] = []
        begins: list[Tensor] = []
        for r, (
            off,
            begin,
            lengths,
            skip_shard,
            rows_shard,
            kept_shard,
        ) in enumerate(
            zip(
                rows.offsets.local_shards,
                swa_begin.local_shards,
                cache_lengths.local_shards,
                skip.local_shards,
                row_shards,
                kept_off_shards,
                strict=True,
            )
        ):
            start, floor = self._rebased(skip_shard, lengths)
            # Two labels for one request window (the plan's split and the
            # token split's): tie them so the runtime checks the equality.
            floor = F.rebind(
                floor, [begin.shape[0]], "replay: one floor per request"
            )
            kept_off = F.rebind(
                Tensor.from_graph_value(kept_shard),
                [off.shape[0]],
                "replay: one kept bound per request",
            )
            # The staged indices address the merged rows; this rank's sit
            # at its own base there.
            source = Tensor.from_graph_value(rows_shard) - F.reshape(
                bases.local_shards[r][r], [1]
            )
            bid, local = row_ids(kept_off, source.shape[0])
            kept.append(
                RaggedRows(
                    offsets=kept_off,
                    bid=bid,
                    local=local,
                    positions=F.gather(
                        rows.positions.local_shards[r], source, axis=0
                    ),
                    lengths=F.cast(kept_off[1:] - kept_off[:-1], DType.int32),
                )
            )
            sources.append(source)
            starts.append(start)
            begins.append(F.elementwise_max(begin, floor))

        def join(parts: Sequence[Tensor], like: Tensor) -> Tensor:
            return Tensor.from_shard_values(parts, like.mapping)

        def join_rows(attr: str) -> Tensor:
            return join([getattr(k, attr) for k in kept], rows.offsets)

        return ReplayRows(
            rows=RaggedRows(
                offsets=join_rows("offsets"),
                bid=join_rows("bid"),
                local=join_rows("local"),
                positions=join_rows("positions"),
                lengths=join_rows("lengths"),
            ),
            indices=join(sources, rows.positions),
            starts=join(starts, rows.positions),
            swa_begin=join(begins, swa_begin),
            # Each rank's own element, with a constant bound: a host scalar,
            # so the kv-store host function reads a CPU address under capture.
            longest=join(
                [longest[r : r + 1] for r in range(self.dp)], rows.positions
            ),
            host_offsets=extras["replay.host_kept_offsets"],
        )

    def embed_stream(self, b: Batch) -> tuple[Tensor, Tensor]:
        """The embedding on ``hc_mult`` copies and the one-hot first ``pre``.

        Under DP the embedding runs on the full replicated batch (the
        vocab-sharded gather all-reduces while every rank still holds the
        same rows) and the stream and ``pre`` are split afterwards, so the
        per-rank trunk starts from its own rows.
        """
        hc = self.config.hc_mult
        full = b.tokens_full if b.tokens_full is not None else b.tokens
        h = self.embed(full)
        stream = F.stack([h] * hc, axis=1)
        rows = h.shape[0]
        pre = F.concat(
            [
                F.full([rows, 1], 1.0, dtype=DType.float32, device=h.mapping),
                F.full(
                    [rows, hc - 1], 0.0, dtype=DType.float32, device=h.mapping
                ),
            ],
            axis=1,
        )
        if b.tokens_full is not None:
            assert (
                b.row_offsets_full is not None
                and b.data_parallel_splits is not None
                and b.input_row_offsets_i64 is not None
                and self.dp > 1
            )
            mapping = DeviceMapping(self.mesh, (Sharded(0),))
            stream, _ = _split_replicated_batch(
                stream,
                b.row_offsets_full,
                b.input_row_offsets_i64,
                b.data_parallel_splits,
                mapping,
            )
            pre, _ = _split_replicated_batch(
                pre,
                b.row_offsets_full,
                b.input_row_offsets_i64,
                b.data_parallel_splits,
                mapping,
            )
        return stream, pre

    def _run_trunk(
        self,
        b: Batch,
        stream: Tensor,
        pre: Tensor,
        tap_set: tuple[int, ...],
    ) -> tuple[Tensor, Tensor, dict[int, Tensor] | None]:
        """Every layer over the staged batch, optionally capturing taps.

        With replay the layers from :attr:`decoder_start` on run on the kept
        rows only, so the returned stream holds those rows.

        Returns:
            ``(stream, pre, taps)``: the written-back stream with its
            carried ``pre``, and the captured taps keyed by layer id,
            ``None`` when ``tap_set`` is empty.
        """
        rows = b.rows
        offsets = rows.offsets
        positions = rows.positions
        # The grouped MoE's tile-regime estimate: the batch's total token
        # rows read off the CPU DP pair, so the launch needs no
        # device-to-host materialization. ``None`` without data parallelism.
        total_rows = (
            None
            if b.data_parallel_splits is None or b.input_row_offsets_i64 is None
            else _host_total_rows_op(
                b.data_parallel_splits, b.input_row_offsets_i64, last=self.dp
            )
        )
        y, post, comb = mhc_identity_writeback(stream)
        taps: dict[int, Tensor] | None = {} if tap_set else None
        hc = self.config.hc_mult
        selection: dict[str, Tensor] = {}
        for index, layer in enumerate(self.layers):
            if b.replay is not None and index == self.decoder_start:
                src = b.replay.indices
                if total_rows is not None:
                    assert (
                        b.data_parallel_splits is not None
                        and b.replay.host_offsets is not None
                    )
                    total_rows = _host_plan_rows_op(
                        b.data_parallel_splits,
                        TensorValue(b.replay.host_offsets),
                        last=self.dp,
                    )
                b = self._cut_to_replay(b)
                rows, offsets, positions = (
                    b.rows,
                    b.rows.offsets,
                    b.rows.positions,
                )
                stream, pre, y, post, comb = (
                    self._gather_rows(t, src)
                    for t in (stream, pre, y, post, comb)
                )
                selection = {
                    k: self._gather_rows(t, src) for k, t in selection.items()
                }
            if layer.layer_id in self.config.engram_layer_ids:
                stream = mhc_post(
                    y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
                )
                ordinal = F.constant(
                    self.config.engram_layer_ids.index(layer.layer_id),
                    DType.uint32,
                    device=self.mesh.devices[0],
                )
                ordinal = F.distributed_broadcast(ordinal, self.mesh)
                if self.dp > 1:
                    # The engram kernel wants the offsets' length tied to
                    # the batch it shares with ``lookback``. Under DP each
                    # rank's sliced fields carry their own dims, so the
                    # tie is per rank -- a global-shape rebind would
                    # misread the row-sharded offsets -- and the rule-less
                    # tie drops the placement, so re-claim the row
                    # sharding.
                    tied_offsets = _tie_shape_op(
                        offsets, b.lookback, offset=1
                    ).rebind_mapping(b.lookback.mapping)
                else:
                    tied_offsets = F.rebind(
                        offsets,
                        [b.lookback.shape[0] + 1],
                        "Engram offsets contain batch+1 entries",
                    )
                (stream,) = layer.engram(
                    [],
                    b.tables,
                    [ordinal, b.lookback, b.lookback_dead, b.image_token_mask],
                    hidden=stream,
                    tokens=b.tokens,
                    positions=positions.cast(DType.int64),
                    query_start_loc=tied_offsets,
                )
                y, post, comb = mhc_identity_writeback(stream)
            if taps is not None and layer.layer_id in tap_set:
                tap, stream, y, post, comb = _capture_tap(
                    y, stream, post, comb, hc
                )
                taps[layer.layer_id] = tap
            routed = {
                "reuse_r2": selection.get("top_k_r2"),
                "reindex_r1": selection.get("pool"),
                "reuse_r1": selection.get("top_k_r1"),
            }.get(layer.group)
            stream, pre, y, post, comb, produced = layer(
                stream,
                pre,
                y,
                post,
                comb,
                rows,
                b.swa_begin,
                b.global_write_floor,
                b.image_token_mask,
                leaf_values(layer.plan, b.kv),
                slot_constants(layer.plan),
                [] if routed is None else [routed],
                b.comm,
                total_rows,
            )
            if layer.group == "full_r2":
                selection["top_k_r2"] = produced[0]
            elif layer.group == "full_r1_cand":
                selection["top_k_r1"], selection["pool"] = produced[:2]
            elif layer.group == "reindex_r1":
                selection["top_k_r1"] = produced[0]
        stream = mhc_post(
            y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
        )
        return stream, pre, taps

    def _cut_to_replay(self, b: Batch) -> Batch:
        """The batch the decoder runs: the kept rows at their absolute slots.

        The full-batch writes ahead of the cut are untouched. Past it every
        leaf views its tail: ``cache_lengths`` becomes each request's first
        kept position, so the kept rows still write and read the same
        absolute slots the full graph would. ``max_prompt_length`` must be
        the exact longest kept count, as a host value: the select ops bound
        their score rows off it, and under device-graph capture only a staged
        host input reads through a CPU address.
        """
        replay = b.replay
        assert replay is not None
        return replace(
            b,
            rows=replay.rows,
            swa_begin=replay.swa_begin,
            image_token_mask=self._gather_rows(
                b.image_token_mask, replay.indices
            ),
            kv={
                name: replace(
                    leaf,
                    cache_lengths=F.cast(
                        replay.starts, leaf.cache_lengths.dtype
                    ),
                    max_prompt_length=replay.longest,
                )
                for name, leaf in b.kv.items()
            },
            replay=None,
        )

    def _gather_rows(self, t: Tensor, src: Tensor) -> Tensor:
        """``t``'s rows at ``src``, per rank under data parallelism.

        Each rank's ``src`` addresses its own rows and their extents differ,
        so the gather runs per shard and keeps ``t``'s mapping.
        """
        if self.dp == 1 or not t.is_distributed:
            return F.gather(t, src, axis=0)
        return Tensor.from_shard_values(
            [
                F.gather(shard, rows, axis=0)
                for shard, rows in zip(
                    t.local_shards, src.local_shards, strict=True
                )
            ],
            t.mapping,
        )

    def run_trunk_with_taps(
        self, b: Batch, stream: Tensor, pre: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`_run_trunk`, also returning the DSpark taps.

        Returns:
            ``(stream, pre, taps)`` with ``taps`` the hc-mean of the stream
            entering each layer of ``dspark_target_layer_ids``,
            concatenated in that order: ``[rows, hidden * n_taps]``.
        """
        tap_ids = validate_dspark_taps(self.config)
        stream, pre, taps = self._run_trunk(b, stream, pre, tap_ids)
        assert taps is not None
        # The trunk captures in layer order; ``main_proj`` was trained on
        # the configured order.
        return stream, pre, F.concat([taps[i] for i in tap_ids], axis=-1)

    def hidden_states(self, stream: Tensor, pre: Tensor) -> Tensor:
        """The final collapse, normalized: the rows the LM head consumes."""
        return self.norm(mhc_final_collapse(stream, pre))

    def logits_all_rows(self, hidden: Tensor) -> Tensor:
        """The LM head over whole hidden rows, gathered across the mesh."""
        return self._compute_logits(hidden)

    @staticmethod
    def _gather_last_tokens(h: Tensor, row_offsets: Tensor) -> Tensor:
        """Gathers each request's last row from the ragged batch, per shard.

        Under data parallelism ``row_offsets`` is rebased into each rank's
        local frame, so its values only address that rank's own rows; the
        gather runs on each device's own rows, never through an
        auto-sharded ``F.gather`` (the DeepseekV3 ModuleV3 pattern).
        """

        def _body(hv: TensorValue, offsets: TensorValue) -> TensorValue:
            return ops.gather(hv, offsets[1:] - 1, axis=0)

        if not h.is_distributed:
            return F.gather(h, row_offsets[1:] - 1, axis=0)
        return F.functional(_body)(h, row_offsets).rebind_mapping(h.mapping)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *rest: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the model over its graph inputs."""
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        if self.dp > 1:
            # The compile ABI puts the DP split pair right after the row
            # offsets, before the KV leaves (the batch processor inserts
            # them at the same positions).
            data_parallel_splits = rest[0]
            input_row_offsets_i64 = rest[1]
            rest = rest[2:]
        n_kv = len(self.kv_params.flattened_kv_inputs())
        upstream = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in rest[:n_kv])
        )
        # Replicas serve different requests, so their caches vary; under
        # tensor parallelism the devices all store the same latent cache.
        kv_mapping = (
            DeviceMapping(self.mesh, (Unknown(),))
            if self.dp > 1
            else DeviceMapping.replicated(self.mesh)
        )
        caches = {
            name: PagedCacheValues.from_upstream(
                tree.leaves(upstream[name], leaf=KVCacheInputsPerDevice),
                kv_mapping,
            )
            for name in self.kv_params.children
        }
        names = extra_input_names(self.interface_config, self.dp)
        consumed = n_kv + len(names)
        extras = dict(zip(names, rest[n_kv:consumed], strict=True))
        comm_buffers: EPCommBuffers | None = None
        ep_rest = list(rest[consumed:])
        if self.ep_batch_manager is not None:
            # Whatever trails the extras is the EP tail the compile input
            # list appended (the batch manager's own input types).
            comm_buffers = self.ep_batch_manager.comm_buffers(ep_rest)
        elif ep_rest:
            raise ValueError(
                f"{len(ep_rest)} inputs after the extras, but the trunk "
                "holds no EP batch manager to bind them."
            )
        b = self.batch_from_parts(
            tokens,
            input_row_offsets,
            caches,
            extras,
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
            comm_buffers=comm_buffers,
        )
        stream, pre = self.embed_stream(b)
        stream, pre, _ = self._run_trunk(b, stream, pre, ())
        hidden = mhc_final_collapse(stream, pre)
        return self._outputs(hidden, b.row_offsets, return_n_logits)

    def _outputs(
        self, hidden: Tensor, offsets: Tensor, return_n_logits: Tensor
    ) -> ModuleV3Outputs:
        if self.dp > 1:
            # One request's last row lives on its own rank (the offsets are
            # rebased per rank), so the gather runs per shard and one
            # allgather -- in rank order, which is global request order by
            # construction of the splits -- hands the head every
            # replica's requests.
            normalized = self.norm(hidden)
            last = F.allgather(
                self._gather_last_tokens(normalized, offsets), tensor_axis=0
            )
            return ModuleV3Outputs(next_token_logits=self._compute_logits(last))
        last_raw = F.gather(hidden, offsets[1:] - 1, axis=0)
        hidden = self.norm(hidden)
        last = F.gather(hidden, offsets[1:] - 1, axis=0)
        next_logits = self._compute_logits(last)
        logits = None
        logit_offsets = None
        if self.config.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(hidden)
            logit_offsets = (
                offsets.local_shards[0] if offsets.is_distributed else offsets
            )
        elif self.config.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            suffix = F.range(
                count,
                0,
                -1,
                out_dim="return_n_logits_range",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            indices = (offsets[1:].unsqueeze(-1) - suffix).reshape([-1])
            logits = self._compute_logits(F.gather(hidden, indices, axis=0))
            logit_offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            if logit_offsets.is_distributed:
                logit_offsets = logit_offsets.local_shards[0]
        hidden_states = None
        if self.config.return_hidden_states == ReturnHiddenStates.LAST:
            # Pre-final-norm rows, the same contract as the Llama3
            # ModuleV3 model's ``last_h``; the head consumes the
            # normalized ones.
            hidden_states = (
                last_raw.local_shards[0]
                if last_raw.is_distributed
                else last_raw
            )
        elif (
            self.config.return_hidden_states
            == ReturnHiddenStates.ALL_NORMALIZED
        ):
            hidden_states = (
                hidden.local_shards[0] if hidden.is_distributed else hidden
            )
        return ModuleV3Outputs(
            next_token_logits=next_logits,
            logits=logits,
            logit_offsets=logit_offsets,
            hidden_states=hidden_states,
        )

    def _compute_logits(self, hidden: Tensor) -> Tensor:
        logits = self.head(hidden)
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        return logits.cast(DType.float32)
