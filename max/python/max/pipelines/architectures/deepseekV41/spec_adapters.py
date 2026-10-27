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
)
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
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
from .layers.attention import CSA2Cache
from .layers.ragged import ReplicaTokens
from .layers.ragged import V41RaggedRows as RaggedRows
from .model import (
    DeepseekV41Inputs,
    ExpertStreaming,
    csa2_cache,
    engram_tables_per_device,
    make_routed_experts_device_resident,
)
from .model_config import DeepseekV41Config

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

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
) -> BlockProposer[TrunkTaps, TensorValue]:
    """Returns the proposer of the DSpark stages, on the verify's trunk."""
    del speculative
    assert isinstance(verify, _Verify)
    assert isinstance(draft, DeepseekV41DSparkDrafter)
    return _DSparkProposer(verify, draft, draft_config)


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
                streaming.uploads.extend(
                    make_routed_experts_device_resident(
                        self.target,
                        replicas,
                        streaming.deferred,
                        [ref.to_device() for ref in batch.devices],
                        batch.devices,
                    )
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
        if batch.data_parallel_degree > 1:
            logits, hidden = self._data_parallel(
                batch, replicas, caches, visions
            )
        else:
            logits, hidden = self._tensor_parallel(
                batch, replicas, caches, visions
            )
        return Verified(
            logits=logits,
            hidden=hidden,
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
                )
            )
            assert tensor_parallel_taps is not None
            taps_per_dev = tensor_parallel_taps
        outputs = logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            batch.merged_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return outputs.logits["logits"], (taps_per_dev, rows)

    def _data_parallel(
        self,
        batch: BlockBatch,
        replicas: Sequence[DeepseekV41],
        caches: Sequence[CSA2Cache],
        visions: Sequence[VisionMerge],
    ) -> tuple[TensorValue, TrunkTaps]:
        """Runs the verify with each device on its own replica's requests.

        The merged batch is replica-major; each device's taps and rows cover
        its own requests, and the logits cover the whole batch on device 0.
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
        xs, taps = DeepseekV41Trunk.data_parallel_with_taps(
            [replica.trunk for replica in replicas],
            split.tokens,
            split.batch_tokens,
            split.rows,
            caches,
            batch.signal_buffers,
            split.replica_tokens,
            tap_layer_ids=self.config.dspark_target_layer_ids,
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
        lead = replicas[0]
        outputs = logits_postprocess(
            gather_rows(xs, batch.merged_tokens.shape[0], batch.signal_buffers),
            batch.merged_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return outputs.logits["logits"], (taps, split.rows)

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
    ) -> None:
        self._verify = verify
        self.draft = draft
        self.config = verify.config
        self.block_size = draft_config.block_size
        self.mask_token_id = draft_config.mask_token_id
        self._stages: list[DeepseekV41DSparkDrafter] | None = None

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
        return [
            ops.reshape(ids[((rows, f"dspark_block_split_{i}"),)], [1, -1])
            for i, (ids, rows) in enumerate(
                zip(per_dev, self._block_ranges(batch), strict=True)
            )
        ]

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
    ) -> TensorValue:
        """Runs the DSpark stages over each request's draft block.

        Each stage attends over its window of stored context plus the block.
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
        block_hs: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Turns the block's hidden states into draft tokens, one at a time."""
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
)
