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
"""DeepSeek-V4.1's call, and the DSpark stages its checkpoint ships.

The stages are no second model with a cache of their own: their windows are
layers ``num_hidden_layers ..`` of the trunk's sliding-window leaf, and they
embed and project through the trunk's embedding and head. So the draft's
cache is that leaf, at the two lengths the driver gives it, and the proposer
runs the stages beside the trunk's own per-device copies, which only the
verify here makes. The graph takes the pinned Engram tables those copies
read past the canonical tail, with each request's Engram lookback window,
which each step records for the next once its drafts are accepted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from max.driver import Buffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import (
    ReturnHiddenStates,
    ReturnLogits,
    logits_postprocess,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    SampledDraftSampler,
    local_row_offsets,
)
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_input_types import draft_probs_rows_name
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from .deepseekV41 import (
    DeepseekV41,
    DeepseekV41Trunk,
    VisionMerge,
    gather_rows,
)
from .dspark_drafter import DeepseekV41DSparkDrafter
from .kv_leaves import V41KVLeafSpec
from .layers.attention import CSA2Attention, CSA2Cache
from .layers.ragged import ReplayRows, ReplicaTokens
from .layers.ragged import V41RaggedRows as RaggedRows
from .model import (
    DeepseekV41Inputs,
    ExpertStreaming,
    csa2_cache,
    engram_tables_per_device,
    make_routed_experts_device_resident,
)
from .model_config import DeepseekV41Config
from .replay import replay_min_tokens, requested_min_tokens

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

logger = logging.getLogger("max.pipelines")

__all__ = [
    "ADAPTER",
    "read_engram_lookback",
    "record_engram_lookback",
]

PRIMARY_KV = "swa"
"""The window leaf, which holds the DSpark stages' windows too."""

MERGED_ROWS_DIM = "merged_seq_len"
"""The graph dimension counting a step's prompt tokens and all its drafts.

The graph's last input has this length, which sizes the combined
prompt-and-draft token tensors inside the graph.
"""

_ENGRAM_LOOKBACK = "engram_lookback_ids"
"""Key in :attr:`BlockBatch.extra` for the ``[batch, max_ngram_size - 1]``
int64 Engram lookback window on device 0; absent without Engram layers."""

_ENGRAM_HISTORY = "engram_lookback_history"
"""Key in :attr:`BlockBatch.extra` for the buffer each step records the next
step's windows in; absent without Engram layers."""

_ENGRAM_TABLES = "engram_tables"
"""Key in :attr:`BlockBatch.extra` for each device's Engram tables."""

_REPLAY_INDICES = "kept_indices"
_REPLAY_OFFSETS = "kept_offsets"
_REPLAY_SKIP = "kept_skip"
_REPLAY_LONGEST = "kept_longest"
"""Host ``[replicas]`` longest retained rows; never broadcast to devices."""
_REPLAY_HOST_OFFSETS = "host_kept_offsets"
"""Host copy of the kept offsets; only under data parallelism."""
_REPLAY_DEVICE_INPUTS = (_REPLAY_INDICES, _REPLAY_OFFSETS, _REPLAY_SKIP)

TrunkTaps = tuple[list[TensorValue], list[RaggedRows]]
"""Per device: captured trunk hidden states and their row layout.

The states are ``[1, T, d * n_taps]``; the rows describe the verification
pass that produced them.
"""


def read_engram_lookback(
    host_lookback: TensorValue, source: TensorValue, history: BufferValue
) -> TensorValue:
    """Returns the lookback window of each row, from the host or the history.

    Args:
        host_lookback: ``[batch, depth]`` window the host built from the
            request's ``processed_length``.
        source: ``[batch]`` int64, the row of ``history`` to read, or -1 for
            the host window.
        history: ``[rows, depth]`` windows the previous step recorded.
    """
    device = host_lookback.device
    zero = ops.constant(0, DType.int64, device=device)
    recorded = ops.gather(
        ops.buffer_load(history), ops.max(source, zero), axis=0
    )
    return ops.where(ops.unsqueeze(source >= zero, -1), recorded, host_lookback)


def record_engram_lookback(
    history: BufferValue,
    lookback: TensorValue,
    tokens: TensorValue,
    input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
    num_accepted: TensorValue,
) -> None:
    """Stores each row's lookback window for the step after this one.

    The next step starts after the row's chunk and its accepted drafts, so
    its window ends with those tokens, then this step's window.

    Args:
        history: ``[rows, depth]`` buffer; row ``i`` receives batch row
            ``i``'s window.
        lookback: ``[batch, depth]`` this step's window; column ``j`` is the
            token ``j + 1`` positions before the chunk.
        tokens: ``[total_seq_len]`` ragged chunk tokens.
        input_row_offsets: ``[batch + 1]`` offsets of the chunks.
        draft_tokens: ``[batch, num_drafts]`` drafts this step verified.
        num_accepted: ``[batch]`` drafts accepted per row.
    """
    device = lookback.device
    batch, depth = lookback.shape[0], int(lookback.shape[1])
    i = ops.range(0, depth, 1, depth, dtype=DType.int64, device=device)
    starts = input_row_offsets[:-1].cast(DType.int64).rebind([batch])
    lengths = input_row_offsets[1:].cast(DType.int64).rebind([batch]) - starts
    # The last `depth` tokens up to the chunk's end, oldest first: chunk
    # position `lengths - depth + i`, or this step's window before it.
    position = ops.unsqueeze(lengths, -1) - depth + i
    zero = ops.constant(0, DType.int64, device=device)
    from_chunk = ops.gather(
        tokens, ops.unsqueeze(starts, -1) + ops.max(position, zero), axis=0
    )
    from_window = ops.gather_nd(
        lookback,
        ops.unsqueeze(ops.max(-1 - position, zero), -1),
        batch_dims=1,
    )
    ordered = ops.concat(
        [
            ops.where(position >= zero, from_chunk, from_window),
            draft_tokens.rebind([batch, draft_tokens.shape[1]]),
        ],
        axis=1,
    )
    # Newest first, ending with the last accepted draft.
    newest = ops.unsqueeze(
        num_accepted.cast(DType.int64).rebind([batch]), -1
    ) + (depth - 1 - i)
    window = ops.gather_nd(ordered, ops.unsqueeze(newest, -1), batch_dims=1)
    rows = ops.shape_to_tensor([batch]).reshape([])
    first = ops.constant(0, DType.int64, DeviceRef.CPU())
    ops.buffer_store_slice(
        history, window, [(slice(first, rows), batch), slice(None)]
    )


def replica_verify_logits(
    replicas: Sequence[DeepseekV41],
    xs: Sequence[TensorValue],
    merged_offsets: Sequence[TensorValue],
    splits: TensorValue,
    return_n_logits: TensorValue,
) -> list[TensorValue]:
    """Each replica's verify logits, computed on its own device.

    Args:
        replicas: One trunk copy per device, each with the whole LM head.
        xs: Per device, its replica's ``[1, t_i, hidden]`` merged rows.
        merged_offsets: Per device, the whole batch's merged row offsets.
        splits: ``[n + 1]`` replica request boundaries, on the host.
        return_n_logits: Logits kept per request.

    Returns:
        Per device, ``[rows_i * return_n_logits, vocab]`` float32.
    """
    return [
        logits_postprocess(
            ops.reshape(x, [-1, x.shape[-1]]),
            local_row_offsets(offsets, splits, i, f"verify_logit_offsets_{i}"),
            return_n_logits,
            norm=replica.norm,
            lm_head=replica._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        ).logits["logits"]
        for i, (replica, x, offsets) in enumerate(
            zip(replicas, xs, merged_offsets, strict=True)
        )
    ]


def _lookback_types(config: DeepseekV41Config) -> list[TensorType | BufferType]:
    """The lookback window, each row's source and the history, on device 0.

    Empty without Engram layers.
    """
    if not config.engram_layer_ids:
        return []
    depth = config.engram_max_ngram_size - 1
    device0 = config.devices[0]
    return [
        TensorType(
            DType.int64, ["engram_lookback_batch", depth], device=device0
        ),
        TensorType(DType.int64, ["engram_lookback_batch"], device=device0),
        BufferType(DType.int64, ["engram_history_rows", depth], device=device0),
    ]


def trailing_input_types(
    target_config: Any,
) -> Sequence[TensorType | BufferType]:
    """Returns the lookback inputs, the pinned Engram tables and the marker.

    The marker, last, is a host tensor whose length declares the merged
    row count (``S + B * K``) as a graph-input dim; buffers shaped by that
    dim fail to compile unless it is an input dim.
    """
    assert isinstance(target_config, DeepseekV41Config)
    return [
        *_lookback_types(target_config),
        *target_config.engram_input_types,
        TensorType(DType.int8, [MERGED_ROWS_DIM], device=DeviceRef.CPU()),
    ]


def batch_trailing_buffers(target_inputs: object) -> list[Buffer]:
    """Returns the batch processor's values of :func:`trailing_input_types`."""
    assert isinstance(target_inputs, DeepseekV41Inputs)
    lookback = [
        buffer
        for buffer in (
            target_inputs.engram_lookback_ids,
            target_inputs.engram_lookback_source,
            target_inputs.engram_lookback_history,
        )
        if buffer is not None
    ]
    assert target_inputs.merged_rows is not None
    return [
        *lookback,
        *target_inputs.engram_host_buffers,
        target_inputs.merged_rows,
    ]


def extra(target_config: Any, trailing: Sequence[Value[Any]]) -> dict[str, Any]:
    """Returns each device's Engram tables, and each row's lookback window.

    The window is read from the history where the previous step is still in
    flight (:func:`read_engram_lookback`).
    """
    assert isinstance(target_config, DeepseekV41Config)
    n_lookback = len(_lookback_types(target_config))
    tables = trailing[
        n_lookback : n_lookback + len(target_config.engram_input_types)
    ]
    out: dict[str, Any] = {
        _ENGRAM_TABLES: engram_tables_per_device(
            tables,
            target_config.engram_layer_ids,
            len(target_config.devices),
        )
    }
    if n_lookback:
        history = trailing[2].buffer
        out[_ENGRAM_LOOKBACK] = read_engram_lookback(
            trailing[0].tensor, trailing[1].tensor, history
        )
        out[_ENGRAM_HISTORY] = history
    return out


def replay_enabled(pipeline_config: PipelineConfig, target_config: Any) -> bool:
    """Returns whether the pairing compiles a replay graph beside the full one.

    The same gate the batch processor plans with, so a replay graph exists
    exactly when a batch can ask for it.
    """
    assert isinstance(target_config, DeepseekV41Config)
    if requested_min_tokens() == 0:
        return False
    threshold, disabled = replay_min_tokens(
        target_config, pipeline_config.runtime.device_graph_capture
    )
    if disabled is not None:
        logger.warning(
            "DSV41 decoder replay requested but disabled: %s", disabled
        )
    return threshold > 0


def replay_input_types(
    target_config: Any,
) -> Sequence[TensorType | BufferType]:
    """Returns the inputs only the replay graph takes, last of all."""
    assert isinstance(target_config, DeepseekV41Config)
    device0 = target_config.devices[0]
    types: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["replay_rows"], device=device0),
        TensorType(DType.uint32, ["input_row_offsets_len"], device=device0),
        TensorType(DType.int32, ["batch_size"], device=device0),
        TensorType(DType.uint32, ["replay_replicas"], device=DeviceRef.CPU()),
    ]
    if target_config.data_parallel_degree > 1:
        types.append(
            TensorType(
                DType.uint32, ["input_row_offsets_len"], device=DeviceRef.CPU()
            )
        )
    return types


def batch_replay_buffers(target_inputs: object) -> list[Buffer]:
    """Returns the batch processor's values of :func:`replay_input_types`.

    Empty for a batch that runs the full graph.
    """
    assert isinstance(target_inputs, DeepseekV41Inputs)
    return list(target_inputs.replay_buffers)


def replay_extra(
    target_config: Any, replay: Sequence[Value[Any]]
) -> dict[str, Any]:
    """Returns the replay graph's inputs, named for the verify."""
    assert isinstance(target_config, DeepseekV41Config)
    names = (
        *_REPLAY_DEVICE_INPUTS,
        _REPLAY_LONGEST,
        _REPLAY_HOST_OFFSETS,
    )[: len(replay_input_types(target_config))]
    return {
        name: value.tensor for name, value in zip(names, replay, strict=True)
    }


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the routed experts' expert-parallel inputs, if any."""
    ep_config = target.config.ep_config
    if ep_config is None:
        return ()
    return EPBatchManager(ep_config).input_types()


def batch_distributed_buffers(
    target_inputs: object,
) -> tuple[Sequence[Buffer] | None, Sequence[Buffer]]:
    """Returns each replica's context length and the expert-parallel buffers."""
    assert isinstance(target_inputs, DeepseekV41Inputs)
    return (
        target_inputs.batch_context_lengths or None,
        target_inputs.ep_buffers,
    )


def leaf_caches(
    specs: Sequence[V41KVLeafSpec],
    window: Sequence[PagedCacheValues],
    others: Mapping[str, Sequence[PagedCacheValues]],
) -> list[CSA2Cache]:
    """Rebuilds each device's :class:`CSA2Cache` from its individual caches.

    ``window`` replaces the sliding-window cache, since the driver passes
    that cache with its own lengths.
    """
    groups = [
        list(window) if spec.key == PRIMARY_KV else list(others[spec.key])
        for spec in specs
    ]
    return [csa2_cache(specs, groups, i) for i in range(len(window))]


def local_replay_rows(
    indices: Sequence[TensorValue],
    skip: Sequence[TensorValue],
    kept_offsets: TensorValue,
    splits: TensorValue,
    kept_tokens: ReplicaTokens,
    original_tokens: ReplicaTokens,
    cache_lengths: Sequence[TensorValue],
    window: int,
    longest: TensorValue,
) -> list[ReplayRows]:
    """Maps batch-global replay rows to each DP replica, including empty ones.

    ``cache_lengths`` are per device and cover only that replica's requests,
    so each replica's start and floor are rebased after the request slice.
    """
    local = []
    for i, (device_indices, device_skip) in enumerate(
        zip(indices, skip, strict=True)
    ):
        first, last = kept_tokens.ranges[i]
        request_first, request_last = splits[i], splits[i + 1]
        local_indices = device_indices[
            ((slice(first, last), f"replay_indices_{i}"),)
        ]
        original_first = original_tokens.ranges[i][0].to(local_indices.device)
        offsets = kept_offsets.to(local_indices.device)[
            ((slice(request_first, request_last + 1), f"replay_offsets_{i}"),)
        ]
        offsets = offsets - ops.cast(
            first.to(local_indices.device), DType.uint32
        )
        local_skip = device_skip[
            ((slice(request_first, request_last), f"replay_skip_{i}"),)
        ]
        local.append(
            ReplayRows.from_inputs(
                local_indices - original_first,
                offsets,
                local_skip,
                cache_lengths[i],
                window,
                longest[i : i + 1],
            )
        )
    return local


def _reset_selections(replicas: Sequence[DeepseekV41]) -> None:
    """Clears the selection the previous trace left in each attention's box.

    A trace that does not start at layer 0 would read it; the first layer
    overwrites it today, so this guards against the structure changing.
    """
    for replica in replicas:
        for block in replica.trunk.layers:
            attn = getattr(block, "attn", None)
            if isinstance(attn, CSA2Attention):
                attn.selection_box.current = None


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a DeepSeek-V4.1 trunk, returning DSpark's taps.

    Raises:
        ValueError: If the trunk wasn't built to return selected layers,
            which only block drafts read.
    """
    del draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A DeepSeek-V4.1 target verifies only for block drafts, which"
            f" read selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, DeepseekV41)
    assert isinstance(target_config, DeepseekV41Config)
    return _Verify(target, target_config)


def block_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[TrunkTaps, TensorValue | list[TensorValue]]:
    """Returns the proposer of the DSpark stages, on the verify's trunk.

    With ``draft_probs_layout="replica"`` the verify also keeps each
    replica's logits on its own device, for a verdict where the rows live.
    The trunk's LM head is replicated, so ``logits_delivery`` needs nothing
    more: no device ever receives another replica's logits.
    """
    assert isinstance(verify, _Verify)
    assert isinstance(draft, DeepseekV41DSparkDrafter)
    per_replica_probs = speculative.draft_probs_layout == "replica"
    verify.replica_logits = per_replica_probs
    return _DSparkProposer(
        verify,
        draft,
        draft_config,
        per_replica_head=speculative.draft_head_parallelism == "replica",
        per_replica_probs=per_replica_probs,
    )


class _Verify:
    """The trunk over the merged prompt and draft tokens.

    On more than one device it runs on per-device copies of the trunk:
    tensor-parallel, or data-parallel with expert-parallel MoE when the
    config's data-parallel degree is the device count.
    """

    def __init__(self, target: DeepseekV41, config: DeepseekV41Config) -> None:
        self.target = target
        self.config = config
        self.takes_distributed_inputs = config.data_parallel_degree > 1
        self.replica_logits = False
        self._replicas: list[DeepseekV41] | None = None

    def replicas(self, batch: BlockBatch) -> list[DeepseekV41]:
        """The trunk, or one copy of it per device.

        The copies are made once, in the graph, after the weights are
        loaded, as the trunk requires. They read the Engram tables the graph
        takes, and their routed experts are made device-resident there.
        """
        if self._replicas is None:
            tables = batch.extra[_ENGRAM_TABLES]
            if batch.n_devs == 1:
                self.target.bind_engram_tables(tables[0])
                replicas = [self.target]
            elif batch.data_parallel_degree > 1:
                replicas = self.target.data_parallel_replicas(
                    batch.devices, tables
                )
            else:
                replicas = self.target.tensor_parallel_replicas(
                    batch.devices, tables
                )
            streaming = self.target.expert_streaming
            if streaming is not None:
                assert isinstance(streaming, ExpertStreaming)
                # The replay graph traces the verify again, and its upload of
                # a name is the full graph's.
                known = {upload.name for upload in streaming.uploads}
                uploads = make_routed_experts_device_resident(
                    self.target,
                    replicas,
                    streaming.deferred,
                    [ref.to_device() for ref in batch.devices],
                    batch.devices,
                )
                streaming.uploads.extend(
                    upload for upload in uploads if upload.name not in known
                )
            self._replicas = replicas
        return self._replicas

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[TrunkTaps]:
        """Runs the trunk from each request's cached length.

        Writes every prompt and draft token into every KV cache and returns
        the logits at every position, plus the hidden states entering each
        DSpark target layer. Rows written past what is accepted are never
        read before the next step overwrites them: that step starts at the
        first unaccepted position, and every read is below the reader's own
        position.
        """
        assert isinstance(batch, BlockBatch)
        caches = leaf_caches(
            self.config.kv_leaf_specs,
            batch.kv_collections,
            batch.passthrough_kv,
        )
        # Images only reach a step without drafts (the pipeline verifies no
        # drafts while a request needs vision encoding), so the merged rows
        # are the prompt rows the scatter indices were computed against.
        visions = [
            VisionMerge(embeddings, indices)
            for embeddings, indices in zip(
                batch.vision_embeddings,
                batch.vision_scatter_indices,
                strict=True,
            )
        ]
        replicas = self.replicas(batch)
        if _REPLAY_INDICES in batch.extra:
            _reset_selections(replicas)
        if batch.data_parallel_degree > 1:
            logits, hidden = self._data_parallel(
                batch, replicas, caches, visions
            )
        else:
            logits, hidden = self._tensor_parallel(
                batch, replicas, caches, visions
            )
        # No whole-batch logits exist when each replica keeps its own; the
        # verdict then reads only ``replica_logits``.
        replica_logits = logits if isinstance(logits, list) else None
        return Verified(
            logits=logits[0] if isinstance(logits, list) else logits,
            hidden=hidden,
            replica_logits=replica_logits,
            roll_forward=(
                self._record_lookback
                if _ENGRAM_HISTORY in batch.extra
                else None
            ),
        )

    def _tensor_parallel(
        self,
        batch: BlockBatch,
        replicas: Sequence[DeepseekV41],
        caches: Sequence[CSA2Cache],
        visions: Sequence[VisionMerge],
    ) -> tuple[TensorValue, TrunkTaps]:
        t = batch.merged_tokens.shape[0]
        rows = [
            RaggedRows.from_offsets(
                offsets,
                t,
                ops.cast(cache.swa_collection.cache_lengths, DType.int32),
            )
            for offsets, cache in zip(
                batch.merged_offsets_per_dev, caches, strict=True
            )
        ]
        tokens = [
            ops.reshape(tok, [1, t])
            for tok in broadcast_per_device(
                batch.merged_tokens, batch.signal_buffers, batch.n_devs
            )
        ]
        replay = self._replay_per_device(batch, caches)
        rows_out, output_offsets = rows, batch.merged_offsets
        if replay is not None:
            rows_out = [item.rows for item in replay]
            t = batch.extra[_REPLAY_INDICES].shape[0]
            output_offsets = batch.extra[_REPLAY_OFFSETS]
        lookback = batch.extra.get(_ENGRAM_LOOKBACK)
        lead = replicas[0]
        tap_layer_ids = self.config.dspark_target_layer_ids
        if len(replicas) == 1:
            x, taps = lead.trunk.forward_with_taps(
                tokens[0],
                rows[0],
                caches[0],
                tap_layer_ids=tap_layer_ids,
                engram_operands=lead._engram_operands(
                    tokens[0], rows[0], lookback
                ),
                vision=visions[0],
                replay=replay[0] if replay is not None else None,
            )
            assert taps is not None
            taps_per_dev = [taps]
        else:
            x, tensor_parallel_taps = (
                DeepseekV41Trunk.tensor_parallel_with_taps(
                    [replica.trunk for replica in replicas],
                    tokens,
                    rows,
                    caches,
                    batch.signal_buffers,
                    tap_layer_ids=tap_layer_ids,
                    engram_operands=DeepseekV41._engram_operands_tensor_parallel(
                        replicas,
                        tokens,
                        rows,
                        (
                            broadcast_per_device(
                                lookback, batch.signal_buffers, batch.n_devs
                            )
                            if lookback is not None
                            else None
                        ),
                        batch.signal_buffers,
                    ),
                    visions=visions,
                    replay=replay,
                )
            )
            assert tensor_parallel_taps is not None
            taps_per_dev = tensor_parallel_taps
        outputs = logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            output_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return outputs.logits["logits"], (taps_per_dev, rows_out)

    def _replay_operands(self, batch: BlockBatch) -> list[list[TensorValue]]:
        return [
            broadcast_per_device(
                batch.extra[name], batch.signal_buffers, batch.n_devs
            )
            for name in _REPLAY_DEVICE_INPUTS
        ]

    def _replay_per_device(
        self, batch: BlockBatch, caches: Sequence[CSA2Cache]
    ) -> list[ReplayRows] | None:
        """The retained rows per device, or ``None`` for the full graph."""
        if _REPLAY_INDICES not in batch.extra:
            return None
        # One replica holds the whole batch, so its longest is the first.
        longest = batch.extra[_REPLAY_LONGEST][0:1]
        return [
            ReplayRows.from_inputs(
                indices,
                offsets,
                skip,
                cache.swa_collection.cache_lengths,
                self.config.sliding_window,
                longest,
            )
            for (indices, offsets, skip), cache in zip(
                zip(*self._replay_operands(batch), strict=True),
                caches,
                strict=True,
            )
        ]

    def _data_parallel(
        self,
        batch: BlockBatch,
        replicas: Sequence[DeepseekV41],
        caches: Sequence[CSA2Cache],
        visions: Sequence[VisionMerge],
    ) -> tuple[TensorValue | list[TensorValue], TrunkTaps]:
        """Runs the verify with each device on its own replica's requests.

        The merged batch is replica-major; each device's taps and rows cover
        its own requests, and the logits cover the whole batch on device 0,
        or with :attr:`replica_logits` each replica's own rows on its device.
        ``visions`` hold the whole batch's image rows on every device, and
        each device merges its own.
        """
        assert batch.host_merged_offsets is not None
        splits = batch.data_parallel_splits
        assert splits is not None
        manager = replicas[0].ep_batch_manager
        if manager is not None:
            assert batch.ep_inputs is not None
            manager.fetch_buffers(batch.ep_inputs)
        lookback = batch.extra.get(_ENGRAM_LOOKBACK)
        split = DeepseekV41.split_replicas(
            replicas,
            broadcast_per_device(
                batch.merged_tokens, batch.signal_buffers, batch.n_devs
            ),
            batch.merged_offsets_per_dev,
            batch.host_merged_offsets,
            splits,
            caches,
            (
                broadcast_per_device(
                    lookback, batch.signal_buffers, batch.n_devs
                )
                if lookback is not None
                else None
            ),
            prefix="verify",
        )
        replay: list[ReplayRows] | None = None
        batch_kept_indices = None
        kept_replica_tokens = None
        rows_out = split.rows
        output_offsets = batch.merged_offsets
        offsets_per_dev = batch.merged_offsets_per_dev
        total = batch.merged_tokens.shape[0]
        if _REPLAY_INDICES in batch.extra:
            replay_operands = self._replay_operands(batch)
            total = batch.extra[_REPLAY_INDICES].shape[0]
            output_offsets = batch.extra[_REPLAY_OFFSETS]
            kept_replica_tokens = ReplicaTokens.from_splits(
                batch.extra[_REPLAY_HOST_OFFSETS], splits, total, batch.n_devs
            )
            batch_kept_indices = replay_operands[0]
            offsets_per_dev = replay_operands[1]
            replay = local_replay_rows(
                replay_operands[0],
                replay_operands[2],
                output_offsets,
                splits,
                kept_replica_tokens,
                split.replica_tokens,
                [cache.swa_collection.cache_lengths for cache in caches],
                self.config.sliding_window,
                batch.extra[_REPLAY_LONGEST],
            )
            rows_out = [item.rows for item in replay]
        xs, taps = DeepseekV41Trunk.data_parallel_with_taps(
            [replica.trunk for replica in replicas],
            split.tokens,
            split.batch_tokens,
            split.rows,
            caches,
            batch.signal_buffers,
            split.replica_tokens,
            tap_layer_ids=self.config.dspark_target_layer_ids,
            batch_kept_indices=batch_kept_indices,
            kept_replica_tokens=kept_replica_tokens,
            replay=replay,
            engram_operands=DeepseekV41._engram_operands_data_parallel(
                replicas, split
            ),
            visions=[
                vision.for_rows(offsets, splits[i], splits[i + 1])
                for i, (vision, offsets) in enumerate(
                    zip(visions, batch.merged_offsets_per_dev, strict=True)
                )
            ],
        )
        assert taps is not None
        if self.replica_logits:
            return replica_verify_logits(
                replicas,
                xs,
                offsets_per_dev,
                splits,
                batch.return_n_logits,
            ), (taps, rows_out)
        lead = replicas[0]
        outputs = logits_postprocess(
            gather_rows(xs, total, batch.signal_buffers),
            output_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return outputs.logits["logits"], (taps, rows_out)

    def _record_lookback(self, batch: BlockBatch) -> None:
        """Records each row's lookback window for the next step to read.

        Only that step reads it, so it may be recorded before this step's
        draft runs.
        """
        assert batch.num_accepted is not None
        record_engram_lookback(
            batch.extra[_ENGRAM_HISTORY],
            batch.extra[_ENGRAM_LOOKBACK],
            batch.tokens,
            batch.input_row_offsets,
            batch.draft_tokens,
            batch.num_accepted,
        )


class _DSparkProposer:
    """Drafts a block of tokens with the DSpark stages.

    Every block position drafts, including the first (the last accepted
    token). The stages embed through the trunk's embedding, so
    :meth:`embed_block` passes the token ids through and
    :meth:`forward_block` embeds them.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self,
        verify: _Verify,
        draft: DeepseekV41DSparkDrafter,
        draft_config: Any,
        per_replica_head: bool = False,
        per_replica_probs: bool = False,
    ) -> None:
        self._verify = verify
        self.draft = draft
        self._per_replica_head = per_replica_head
        self._per_replica_probs = per_replica_probs
        self.config = verify.config
        self.block_size = draft_config.block_size
        self.mask_token_id = draft_config.mask_token_id
        self._stages: list[DeepseekV41DSparkDrafter] | None = None
        self._block_ids: list[TensorValue] = []

    def stages(self, batch: BlockBatch) -> list[DeepseekV41DSparkDrafter]:
        """The stages, or one copy of them per device."""
        if self._stages is None:
            self._stages = (
                [self.draft]
                if batch.n_devs == 1
                else self.draft.replicas(
                    batch.devices,
                    split_heads=batch.data_parallel_degree == 1,
                )
            )
        return self._stages

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: TrunkTaps,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Stores each stage's context KV for every verified token."""
        taps, rows = target_hidden
        for stage, tap, row, swa in zip(
            self.stages(batch), taps, rows, ctx_kv, strict=True
        ):
            stage.store_context(tap, row, swa)

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Each device's block token ids; its own blocks under DP."""
        per_dev = broadcast_per_device(
            block_ids, batch.signal_buffers, batch.n_devs
        )
        if batch.data_parallel_degree == 1:
            return [ops.reshape(ids, [1, -1]) for ids in per_dev]
        # Kept for the per-replica head, which reads each block's anchor.
        self._block_ids = []
        embeds = []
        for i, (ids, rows) in enumerate(
            zip(per_dev, self._block_ranges(batch), strict=True)
        ):
            self._block_ids.append(ids[((rows, f"dspark_block_split_{i}"),)])
            embeds.append(ops.reshape(self._block_ids[-1], [1, -1]))
        return embeds

    def _block_ranges(self, batch: BlockBatch) -> list[slice]:
        """Each device's run of block rows: its replica's requests' blocks."""
        splits = batch.data_parallel_splits
        assert splits is not None
        k = ops.constant(self.block_size, DType.int64, DeviceRef.CPU())
        return [
            slice(splits[replica] * k, splits[replica + 1] * k)
            for replica in batch.replica_of
        ]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> TensorValue | list[TensorValue]:
        """Runs the DSpark stages over each request's draft block.

        Each stage attends over its window of stored context plus the block.
        With a per-replica head, each device keeps its own blocks' states.
        """
        stages = self.stages(batch)
        trunks = self._verify.replicas(batch)
        rows = [
            RaggedRows.from_offsets(
                off, ids.shape[1], ops.cast(swa.cache_lengths, DType.int32)
            )
            for off, ids, swa in zip(offsets, embeds, block_kv, strict=True)
        ]
        windows = [
            stage.windows(row.starts, swa)
            for stage, row, swa in zip(stages, rows, block_kv, strict=True)
        ]
        if len(stages) == 1:
            return stages[0].forward_block(
                embeds[0], rows[0], windows[0], trunks[0].trunk.embed
            )
        if batch.data_parallel_degree > 1:
            total = batch.input_row_offsets.shape[0] - 1
            ranges = self._block_ranges(batch)
            hs = DeepseekV41DSparkDrafter.forward_block_data_parallel(
                stages,
                embeds,
                rows,
                windows,
                [trunk.trunk.embed for trunk in trunks],
                batch.signal_buffers,
                ReplicaTokens(
                    total=total * self.block_size,
                    ranges=[(r.start, r.stop) for r in ranges],
                ),
            )
            if self._per_replica_head:
                return hs
            return ops.reshape(
                gather_rows(hs, total * self.block_size, batch.signal_buffers),
                [1, -1, self.config.hidden_size],
            )
        return DeepseekV41DSparkDrafter.forward_block_tensor_parallel(
            stages,
            embeds,
            rows,
            windows,
            [trunk.trunk.embed for trunk in trunks],
            batch.signal_buffers,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: TensorValue | list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Turns the block's hidden states into draft tokens, one at a time."""
        if isinstance(block_hs, list):
            return self._replica_head(batch, block_hs, accepted, sampler)
        lead = self.stages(batch)[0]
        logits = lead.logits(
            block_hs, self._verify.replicas(batch)[0].head.weight
        )
        sample: Callable[[TensorValue, int], TensorValue] = (
            lambda step_logits, position: sampler.sample_next(
                step_logits, position
            )
        )
        drafts, _ = lead.draft(logits, accepted.next_tokens, sampler=sample)
        return ops.cast(drafts, accepted.next_tokens.dtype)

    def _replica_head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Drafts each replica's rows on its own device.

        The sequential head then runs over a replica's few rows on every
        device at once, rather than over the whole batch on device 0 while
        the others wait. Only the draft tokens are gathered back onto device
        0, with a sampled draft's distributions, unless they stay on each
        replica's device.
        """
        splits = batch.data_parallel_splits
        assert splits is not None
        signals = batch.signal_buffers
        rows = [f"dspark_head_rows_{r}" for r in batch.replica_of]

        def per_replica(value: TensorValue) -> list[TensorValue]:
            # ``[batch_size]`` on device 0 -> each device's replica rows.
            return [
                ops.rebind(
                    v[
                        (
                            (
                                slice(splits[r], splits[r + 1]),
                                f"dspark_head_split_{i}",
                            ),
                        )
                    ],
                    [rows[i]],
                )
                for i, (v, r) in enumerate(
                    zip(
                        broadcast_per_device(value, signals, batch.n_devs),
                        batch.replica_of,
                        strict=True,
                    )
                )
            ]

        samplers: Sequence[DraftSampler] = (
            sampler.split(per_replica, rows)
            if isinstance(sampler, SampledDraftSampler)
            else [sampler] * batch.n_devs
        )
        drafts = [
            stage.draft(
                stage.logits(hs, trunk.head.weight),
                anchors,
                sampler=local.sample_next,
            )[0]
            for stage, trunk, hs, anchors, local in zip(
                self.stages(batch),
                self._verify.replicas(batch),
                block_hs,
                self._anchors(rows),
                samplers,
                strict=True,
            )
        ]
        if isinstance(sampler, SampledDraftSampler):
            dists = [
                local.distributions()
                for local in samplers
                if isinstance(local, SampledDraftSampler)
            ]
            if self._per_replica_probs:
                sampler.adopt_replica_distributions(
                    [
                        dist.rebind([draft_probs_rows_name(r), *dist.shape[1:]])
                        for dist, r in zip(dists, batch.replica_of, strict=True)
                    ]
                )
            else:
                sampler.adopt_distributions(ops.allgather(dists, signals)[0])
        return ops.cast(
            ops.allgather(drafts, signals)[0], accepted.next_tokens.dtype
        ).rebind([accepted.next_tokens.shape[0], self.block_size])

    def _anchors(self, rows: list[str]) -> list[TensorValue]:
        """Each device's blocks' first tokens, read from its block ids.

        They are the rows of ``accepted.next_tokens`` in that device's replica.
        """
        return [
            ops.slice_tensor(ids, [(slice(None, None, self.block_size), n)])
            for ids, n in zip(self._block_ids, rows, strict=True)
        ]


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    block_proposer=block_proposer,
    ep_input_types=ep_input_types,
    batch_distributed_buffers=batch_distributed_buffers,
    # The graph takes no in-thinking input, as the fused graph didn't.
    relaxed_thinking_acceptance=False,
    prepares_every_batch=True,
    trailing_input_types=trailing_input_types,
    batch_trailing_buffers=batch_trailing_buffers,
    extra=extra,
    replay_enabled=replay_enabled,
    replay_input_types=replay_input_types,
    batch_replay_buffers=batch_replay_buffers,
    replay_extra=replay_extra,
)
