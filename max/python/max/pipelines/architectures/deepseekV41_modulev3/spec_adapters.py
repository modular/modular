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
"""DeepSeek-V4.1's call on ModuleV3, and the DSpark stages its checkpoint ships.

The drafter is not a second model with its own cache: its stages store their
context KV in the target's sliding-window leaf, at the layer slots the
resolver already reserves. The block driver is graph-value typed, so the
verify and the proposer convert at their own boundary -- ``as_tensors`` and
:meth:`PagedCacheValues.from_upstream` on the way in, ``as_graph_values`` on
the way out.

The ModuleV3 target and drafter run on one mesh rather than per-device
replicas, so each adapter method stages the single-device values the driver
hands it onto that mesh (``F.distributed_broadcast``, the trunk's own
staging rule) and collapses its driver-facing results back to rank 0.

Under data parallelism the mesh's one axis is the replica axis: the trunk
splits the merged window per rank in-graph (the driver hands it
``data_parallel_splits`` and the host mirror of the merged offsets), the
drafter's stage weights replicate and each rank drafts its own replica's
block rows, and the two heads gather every replica's rows back to the
global batch -- the trunk's own ``logits`` pattern, which keeps the accept,
the drafts and every driver-facing tensor at the global width the block
driver's phases expect. The greedy draft head never joins the vocabulary:
each rank argmaxes its own vocab shard of every row and only the per-row
winners cross the mesh.

The graph takes the trunk's own extras past the canonical tail, which the
trunk's batch processor prepares for every batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from max import functional as F
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, Dim, TensorType, TensorValue, Value, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.kv_cache import PagedCacheValues as UpstreamPagedCacheValues
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures.deepseekV41.engram_lookback import (
    read_engram_lookback,
    record_engram_lookback,
)
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    DraftSampler,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    SequentialBatch,
    gather_variable_rows,
)
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Unknown,
)

from .deepseekV41 import DeepseekV41
from .dspark_drafter import DSparkDrafter
from .inputs import DeepseekV41Inputs
from .interface import V41Config, extra_input_names
from .layers.attention import absolute_positions
from .model_config import DeepseekV41Config

__all__ = [
    "ADAPTER",
    "WINDOW_LEAF",
    "DSparkDeepseekV41Proposer",
    "DeepseekV41BlockTarget",
    "TrunkTaps",
    "block_row_positions",
    "context_store_origins",
    "window_starts",
]


WINDOW_LEAF = "swa"
"""The leaf holding the trunk's sliding window and the drafter's slots."""

TrunkTaps = tuple[Tensor, Tensor, Tensor]
"""The verify payload: the taps, and the rows and positions that produced them.

``(taps, row_offsets, positions)`` over the merged verify window. The
proposer stores each stage's context KV from exactly these, so the store
lands at the positions the verify forward ran at -- never re-derived, which
is where an off-by-one would sit silently.
"""


def _to_lead(t: Tensor) -> Tensor:
    """Rank 0's copy of a mesh tensor; single-device tensors pass through."""
    return t.local_shards[0] if t.is_distributed else t


def _extras(
    batch: BlockBatch, config: V41Config, data_parallel_degree: int
) -> dict[str, Tensor]:
    """The trunk's extras, out of the driver's ``extra`` carry-all.

    The names come off the trunk's own config view -- the one view the
    trunk's forward was built from, at its data-parallel degree -- so the
    adapter can never stage an extras set the trunk disagrees with.

    Raises:
        ValueError: If any extra is missing. A silent default here would
            feed the trunk garbage masks and Engram windows, so the miss is
            loud.
    """
    names = extra_input_names(config, data_parallel_degree)
    missing = [name for name in names if name not in batch.extra]
    if missing:
        raise ValueError(
            f"the DSpark verify batch is missing graph inputs: {missing}"
        )
    return {name: as_tensors(batch.extra[name]) for name in names}


def _leaf_kv(
    batch: BlockBatch,
    kv_params: MultiKVCacheParams,
    mesh: DeviceMesh,
    dp: int,
) -> dict[str, PagedCacheValues]:
    """The trunk's per-leaf caches: the window leaf plus the passthroughs.

    Under data parallelism every device's record is its own replica's, so
    the mapping keeps the per-device fields unrelated (the trunk's own
    staging rule); under tensor parallelism they all hold the same latent
    cache, so the mapping replicates.
    """
    mapping = DeviceMapping(mesh, (Unknown(),) if dp > 1 else (Replicated(),))
    kv: dict[str, PagedCacheValues] = {}
    for name in kv_params.children:
        per_device = (
            batch.kv_collections
            if name == WINDOW_LEAF
            else batch.passthrough_kv[name]
        )
        kv[name] = PagedCacheValues.from_upstream(per_device, mapping)
    return kv


def block_row_positions(
    block_offsets: Tensor,
    cache_lengths: Tensor,
    total: Dim,
) -> Tensor:
    """Each block row's absolute position: ``L_r + j``.

    The block's row ``j`` of request ``r`` sits at the request's post-commit
    cache length plus ``j`` -- the anchor token is that length's own
    position, since the commit covered it. The arithmetic is the trunk's
    own step positions (:func:`absolute_positions`), which preserves the
    histogram-cumsum batch ids, the zero-length end sink, and the
    padded-row clamping that keeps out-of-batch gathers in bounds.
    """
    return absolute_positions(block_offsets, cache_lengths, total)


def window_starts(swa: PagedCacheValues) -> Tensor:
    """Each request's first block position, read off the block cache.

    The block cache's lengths sit past the tokens the accept committed, so
    the anchor's position -- the window reads everything below it -- is the
    length itself.
    """
    return F.cast(swa.cache_lengths, DType.int32)


def _shard_block_ids(
    ids: Tensor, splits: Tensor, block_size: int, mapping: DeviceMapping
) -> Tensor:
    """The flat draft block ids as per-replica rows, request-major.

    One flat block holds ``K`` rows per request, so a replica's rows are
    the contiguous span its requests own: ``[splits[r] * K,
    splits[r+1] * K)``. The ids arrive on rank 0, so one broadcast puts
    every rank's copy in place before its span is cut out -- the same
    broadcast-then-slice the trunk runs on the merged window.
    """
    replicated = F.distributed_broadcast(ids, mapping.mesh)
    shards = []
    for rank, shard in enumerate(replicated.local_shards):
        shards.append(
            F.slice_tensor(
                shard,
                [
                    (
                        slice(
                            splits[rank] * block_size,
                            splits[rank + 1] * block_size,
                        ),
                        f"dspark_block_rows_r{rank}",
                    )
                ],
            )
        )
    return Tensor.from_shard_values(shards, mapping)


def _dp_gather_variable_rows(
    hidden: Tensor,
    row_offsets: Tensor,
    return_n_logits: Tensor,
    mesh: DeviceMesh,
) -> Tensor:
    """Each request's last ``n`` hidden rows, gathered per rank then joined.

    The data-parallel form of the framework's ``gather_variable_rows``:
    the gather runs once per rank over its own rows (rule-less functional
    body, so no per-shard shape arithmetic reaches graph IR), and one
    allgather reconstitutes the global batch in rank order -- which is
    global request order by construction of the splits. Allgathering the
    whole merged window would move every prefill row four times; only
    each request's last ``n`` rows ride the collective.
    """
    n_range = F.range(
        return_n_logits[0],
        0,
        -1,
        out_dim="return_n_logits_range",
        dtype=DType.int64,
        device=mesh.devices[0],
    )
    n_range = F.distributed_broadcast(n_range, mesh)

    def _body(
        h: TensorValue, offsets: TensorValue, n: TensorValue
    ) -> TensorValue:
        rows = ops.unsqueeze(offsets[1:], -1) - n
        return ops.gather(h, ops.reshape(rows, (-1,)), axis=0)

    variable_h = F.functional(_body)(
        hidden, row_offsets, n_range
    ).rebind_mapping(hidden.mapping)
    return F.allgather(variable_h, tensor_axis=0)


def context_store_origins(
    row_offsets: Tensor, positions: Tensor, dtype: DType
) -> Tensor:
    """Each request's storage origin: its first payload row's position.

    A narrowed verify keeps each request's last window rows, and the
    payload's ``positions`` are those rows' absolute positions -- already
    advanced past the rows the narrowing dropped. The block driver's
    context cache lengths are the engine-staged pre-iteration lengths,
    unshifted by the drop, so storing from them would write kept row ``i``
    of request ``r`` at ``L_r + i`` while its true position -- the one the
    window read resolves through the lookup table -- is
    ``L_r + shift_r + i``: the drafter's window ring rotates by
    ``shift mod window`` until the decode steps' stores have overwritten
    it. The first payload row's own position is the origin the store
    needs: ``L_r + shift_r`` on a narrowed chunk, and exactly ``L_r`` on
    every unshifted one (decode steps, chunks no longer than the
    window), where it equals the staged length the store used before.

    Under data parallelism the offsets and positions are per-rank
    sharded tensors whose row axes the shape rules cannot project
    (slicing an expression-dim shard reshard-rises), so the gather runs
    per rank over its own shards -- the ``_per_rank_positions`` pattern.
    """
    if row_offsets.is_distributed:
        shards = [
            _context_store_origins_local(offsets, pos, dtype)
            for offsets, pos in zip(
                row_offsets.local_shards,
                positions.local_shards,
                strict=True,
            )
        ]
        return Tensor.from_shard_values(shards, positions.mapping)
    return _context_store_origins_local(row_offsets, positions, dtype)


def _context_store_origins_local(
    row_offsets: Tensor, positions: Tensor, dtype: DType
) -> Tensor:
    """The storage origins over one rank's rows."""
    starts = F.cast(row_offsets[:-1], positions.dtype)
    return F.gather(positions, starts, axis=0).cast(dtype)


def _sample_next(
    sampler: DraftSampler,
) -> Callable[[Tensor, int], Tensor]:
    """Bridges the driver's per-position sampler into the drafter's loop.

    The bridge passes each step's mesh logits through unchanged, so it
    serves only samplers that take distributed ``Tensor``s; the greedy
    :class:`ArgmaxDraftSampler` is routed around it in :meth:`head`.
    """

    # The framework's sampler protocol spells its inputs as graph values
    # while the samplers this bridge serves take mesh ``Tensor``s (the
    # ``_tensor_compat`` boundary is migrating off), so the logits pass
    # through unchanged and the call itself crosses the boundary the same
    # way ``_tensor_compat`` does: dynamically, on the runtime types.
    def sample(step_logits: Tensor, position: int) -> Tensor:
        sample_next: Any = sampler.sample_next
        picked: Tensor = sample_next(step_logits, position)
        return picked

    return sample


class DeepseekV41BlockTarget:
    """Verifies the previous drafts with the trunk over the merged window.

    Every prompt and draft token of the merged window runs through the
    trunk, so their KV is written whatever the accept later commits: rows
    past the committed length are overwritten by the next verify before any
    read, since that verify starts at the committed length and every read
    sits below the reader's own row. The VARIABLE logits are the last
    ``num_speculative_tokens + 1`` rows of each request -- the recovered,
    the drafted, and the bonus positions -- and the taps ride to the
    proposer with the rows and positions that produced them.
    """

    def __init__(
        self,
        target: DeepseekV41,
        kv_params: MultiKVCacheParams,
        *,
        ep_batch_manager: Any | None = None,
    ) -> None:
        self.target = target
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager
        self.takes_distributed_inputs = target.dp > 1

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[TrunkTaps]:
        """Runs the trunk over the merged window and returns its taps."""
        assert isinstance(batch, BlockBatch)
        target = self.target
        dp = target.dp
        comm_buffers = None
        if self.ep_batch_manager is not None:
            if batch.ep_inputs is None:
                raise ValueError(
                    "the trunk runs expert-parallel but the batch carries "
                    "no EP collective inputs"
                )
            comm_buffers = self.ep_batch_manager.comm_buffers(
                [as_tensors(value) for value in batch.ep_inputs]
            )
        input_row_offsets_i64 = None
        if dp > 1:
            if batch.host_merged_offsets is None:
                raise ValueError(
                    "data parallelism needs the host mirror of the merged "
                    "row offsets"
                )
            # The trunk splits the merged window by request from the int64
            # host mirror (gather indices are int64); the canonical
            # signature declares the mirror uint32, so widen it here.
            input_row_offsets_i64 = F.cast(
                as_tensors(batch.host_merged_offsets), DType.int64
            )
        b = target.batch_from_parts(
            as_tensors(batch.merged_tokens),
            as_tensors(batch.merged_offsets),
            _leaf_kv(batch, self.kv_params, target.mesh, dp),
            _extras(batch, target.interface_config, dp),
            data_parallel_splits=(
                as_tensors(batch.data_parallel_splits)
                if batch.data_parallel_splits is not None
                else None
            ),
            input_row_offsets_i64=input_row_offsets_i64,
            comm_buffers=comm_buffers,
            # The merged frame's draft rows per request, read off the
            # draft_tokens input's own width so the narrowing can widen
            # its kept-row bound by exactly what this step appended: a
            # verify chunk's K placeholder rows must not crowd real rows
            # out of the drafter's stored window (0 on a pure prefill,
            # where nothing is appended and the bound is the bare window).
            draft_width=as_tensors(
                ops.shape_to_tensor([batch.num_draft_tokens])[0]
            ),
        )
        stream, pre = target.embed_stream(b)
        stream, pre, taps = target.run_trunk_with_taps(b, stream, pre)
        hidden = target.hidden_states(stream, pre)
        # The acceptance sampler reads one flat request-major block of
        # num_steps + 1 logit rows per request, which is exactly what the
        # last-K+1 gather produces over the merged window. Under data
        # parallelism the gather runs per rank over its own rows and one
        # allgather reconstitutes the global batch -- the trunk's own
        # ``logits`` pattern.
        if dp > 1:
            var_h = _dp_gather_variable_rows(
                hidden,
                b.row_offsets,
                as_tensors(batch.return_n_logits),
                target.mesh,
            )
        else:
            var_h, _ = gather_variable_rows(
                hidden,
                b.row_offsets,
                as_tensors(batch.return_n_logits),
                target.mesh,
            )
        return Verified(
            logits=as_graph_values(target.logits_all_rows(var_h)),
            hidden=(taps, b.row_offsets, b.positions),
            roll_forward=(
                self._record_lookback
                if "engram_lookback_history" in batch.extra
                else None
            ),
        )

    def _record_lookback(self, batch: BlockBatch) -> None:
        """Records each row's lookback window for the next step to read.

        Only that step reads it, so it may be recorded before this step's
        draft runs. The window is read back out of the history the same way
        the trunk read it; the compiler folds the two reads into one.
        """
        assert batch.num_accepted is not None
        history = batch.extra["engram_lookback_history"].buffer
        record_engram_lookback(
            history,
            read_engram_lookback(
                batch.extra["engram_lookback_ids"].tensor,
                batch.extra["engram_lookback_source"].tensor,
                history,
            ),
            batch.tokens,
            batch.input_row_offsets,
            batch.draft_tokens,
            batch.num_accepted,
        )


class DSparkDeepseekV41Proposer:
    """Drafts a block with the DSpark stages; every position drafts.

    The block is the committed token then ``block_size - 1`` noise slots,
    and the drafter embeds it through the target's own embedding, so
    :meth:`embed_block` passes the ids through. The stages read their
    context windows from the sliding-window leaf and attend over them plus
    the block itself.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self,
        drafter: DSparkDrafter,
        target: DeepseekV41,
        config: DeepseekV41Config,
    ) -> None:
        if config.dspark_noise_token_id is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs dspark_noise_token_id to fill "
                "the draft block"
            )
        self.drafter = drafter
        self.target = target
        self.config = config
        self.mesh = drafter.mesh
        self.block_size = config.dspark_block_size
        self.mask_token_id = config.dspark_noise_token_id
        # The trunk's replica count: TP1 per rank under data parallelism,
        # so the drafter's stage weights replicate and the proposer slices
        # the block per replica instead of broadcasting it.
        self.dp = drafter.data_parallel_degree
        # The vocab-parallel draft is the data-parallel route: each rank
        # drafts over its own vocab shard of the trunk's lm_head and only
        # the per-row ``(max, index)`` winners cross the mesh. The joined
        # head (which allgathers the bf16 logits onto rank 0 and
        # broadcasts them back as float32) stays as the greedy head's
        # tensor-parallel route.
        self.vocab_parallel_draft = self.dp > 1

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: TrunkTaps,
        ctx_kv: list[UpstreamPagedCacheValues],
    ) -> None:
        """Stores each stage's context KV for every verified token.

        The staged context cache carries the unshifted pre-iteration
        lengths, so the store's base is replaced with the payload's own
        storage origins -- the first retained row's position of each
        request -- which the narrowed verify has already advanced past
        the dropped rows. The block cache (``pre + commit``) is the
        driver's, untouched.
        """
        taps, row_offsets, positions = target_hidden
        mapping = (
            DeviceMapping(self.mesh, (Unknown(),))
            if self.dp > 1
            else DeviceMapping.replicated(self.mesh)
        )
        swa = PagedCacheValues.from_upstream(ctx_kv, mapping)
        self.drafter.store_context(
            replace(
                swa,
                cache_lengths=context_store_origins(
                    row_offsets, positions, swa.cache_lengths.dtype
                ),
            ),
            taps,
            row_offsets,
            positions,
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Hands the ids through; the drafter embeds them itself."""
        return [block_ids]

    def _dp_embed(self, splits: Tensor) -> Callable[[Tensor], Tensor]:
        """The DP embed: full batch in, per-replica rows out.

        The target's embedding is vocab-sharded over the replica axis, so
        its gather all-reduces across the mesh and every rank must hold the
        same rows -- the trunk's own rule (embed the full replicated
        batch, split afterwards). The drafter embeds inside its forward,
        so the callable it receives re-broadcasts the sharded ids, embeds
        the global batch once, and cuts each rank's K-strided span back
        out.
        """

        def embed(sharded_ids: Tensor) -> Tensor:
            # One allgather reconstitutes the global block in rank order --
            # global request order by construction of the splits -- so the
            # vocab-sharded gather all-reduces while every rank still
            # holds the same rows.
            full = F.allgather(sharded_ids, tensor_axis=0)
            hidden = self.target.embed(full)
            k = self.block_size
            shards = []
            for rank, shard in enumerate(hidden.local_shards):
                start = splits[rank] * k
                end = splits[rank + 1] * k
                shards.append(
                    F.slice_tensor(
                        shard,
                        [
                            (
                                slice(start, end),
                                f"dspark_block_rows_r{rank}",
                            )
                        ],
                    )
                )
            return Tensor.from_shard_values(
                shards, DeviceMapping(self.mesh, (Sharded(0),))
            )

        return embed

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[UpstreamPagedCacheValues],
    ) -> Tensor:
        """Runs the DSpark stages over each request's draft block.

        Each stage attends over its window of stored context -- everything
        below the block's start -- plus the block's own rows.

        Under data parallelism each rank drafts its own replica's block:
        the ids, offsets and windows carry per-rank rows, and the embed
        runs on the full replicated batch first (the vocab-sharded gather
        all-reduces while every rank still holds the same rows).
        """
        if self.dp == 1:
            ids = F.distributed_broadcast(as_tensors(embeds[0]), self.mesh)
            block_offsets = F.distributed_broadcast(
                as_tensors(offsets[0]), self.mesh
            )
            swa = PagedCacheValues.from_upstream(
                block_kv, DeviceMapping.replicated(self.mesh)
            )
            starts = window_starts(swa)
            positions = block_row_positions(
                block_offsets, swa.cache_lengths, ids.shape[0]
            )
            windows = self.drafter.windows(swa, starts)
            return self.drafter.forward_block(
                ids, positions, windows, embed=self.target.embed
            )
        if batch.data_parallel_splits is None:
            raise ValueError(
                "a data-parallel block batch must carry data_parallel_splits"
            )
        row_sharded = DeviceMapping(self.mesh, (Sharded(0),))
        splits = as_tensors(batch.data_parallel_splits)
        # One flat request-major block, K rows per request, so a replica's
        # rows are the contiguous span its requests own.
        ids = _shard_block_ids(
            as_tensors(embeds[0]), splits, self.block_size, row_sharded
        )
        embed = self._dp_embed(splits)
        block_offsets = Tensor.from_shard_values(
            [as_tensors(offset) for offset in offsets], row_sharded
        )
        swa = PagedCacheValues.from_upstream(
            block_kv, DeviceMapping(self.mesh, (Unknown(),))
        )
        # The per-device cache lengths are each replica's own record, so
        # the starts build per shard: a rule-based cast over the
        # Unknown-mapped field has no feasible plan on the replica axis.
        starts = Tensor.from_shard_values(
            [
                F.cast(shard, DType.int32)
                for shard in swa.cache_lengths.local_shards
            ],
            row_sharded,
        )
        positions = Tensor.from_shard_values(
            [
                absolute_positions(offsets, lengths, shard.shape[0])
                for offsets, lengths, shard in zip(
                    block_offsets.local_shards,
                    swa.cache_lengths.local_shards,
                    ids.local_shards,
                    strict=True,
                )
            ],
            row_sharded,
        )
        windows = self.drafter.windows(swa, starts)
        return self.drafter.forward_block(ids, positions, windows, embed=embed)

    def head(
        self,
        batch: BlockBatch,
        block_hs: Tensor,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Turns the block's hidden states into draft tokens, one at a time."""
        block_hs = as_tensors(block_hs)
        # The committed anchors arrive on rank 0 (the accept runs there),
        # so stage them onto the mesh where the serial loop and the
        # replicated Markov head live. Under data parallelism the block
        # rows are sharded, so one allgather reconstitutes the global batch
        # in rank order -- global request order by construction of the
        # splits -- before the head runs, the trunk's own ``logits``
        # pattern.
        if self.dp > 1 and block_hs.is_distributed:
            block_hs = F.allgather(block_hs, tensor_axis=0)
        anchors = F.distributed_broadcast(
            as_tensors(accepted.next_tokens), self.mesh
        )
        greedy = type(sampler) is ArgmaxDraftSampler
        if greedy and self.vocab_parallel_draft:
            # Every rank already holds the whole batch's hidden rows and
            # its own vocab shard of the head, so the greedy loop runs
            # vocab-parallel and exchanges only each row's per-shard
            # winner -- not the ``[b * block_size, vocab]`` float32 logits
            # the joined head would broadcast to every rank.
            return as_graph_values(
                self.drafter.draft_vocab_parallel(
                    block_hs, self.target.head.weight, anchors
                )
            )
        # The joined head's logits land on rank 0 (its allgather takes
        # rank 0's copy), so they are re-staged onto the mesh too.
        logits = F.distributed_broadcast(
            self.drafter.logits(block_hs, self.target.head.weight),
            self.mesh,
        )
        # The driver's greedy sampler draws through ``ops.argmax``, a
        # TensorValue op that rejects the mesh-replicated logits staged
        # here (its ``Tensor.__tensorvalue__`` bridge refuses anything
        # distributed). The drafter's own greedy is the same argmax over
        # ``F.argmax``, whose SPMD reduction rule runs replicated on the
        # mesh, so exactly the greedy class bypasses the bridge; a
        # subclass that overrides ``sample_next`` is its own sampler and
        # still goes through it.
        drafts, _ = self.drafter.draft(
            logits,
            anchors,
            sampler=None if greedy else _sample_next(sampler),
        )
        return as_graph_values(_to_lead(drafts))


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> DeepseekV41BlockTarget:
    """Returns the verify of a DeepSeek-V4.1 trunk, returning DSpark's taps.

    Raises:
        ValueError: If the pairing doesn't read selected layers, as only
            block drafts do.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A DeepSeek-V4.1 target verifies only for block drafts, which"
            f" read selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, DeepseekV41)
    return DeepseekV41BlockTarget(
        target, target.kv_params, ep_batch_manager=target.ep_batch_manager
    )


def block_proposer(
    verify: Any,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> DSparkDeepseekV41Proposer:
    """Returns the proposer of the DSpark stages, on the verify's trunk."""
    del speculative
    assert isinstance(verify, DeepseekV41BlockTarget)
    assert isinstance(draft, DSparkDrafter)
    assert isinstance(draft_config, DeepseekV41Config)
    return DSparkDeepseekV41Proposer(draft, verify.target, draft_config)


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the routed experts' expert-parallel inputs, if any."""
    assert isinstance(target, DeepseekV41)
    if target.ep_batch_manager is None:
        return ()
    return target.ep_batch_manager.input_types()


def trailing_input_types(
    target_config: Any,
) -> list[TensorType | BufferType]:
    """Returns the trunk's extras the graph takes past the canonical tail."""
    assert isinstance(target_config, DeepseekV41Config)
    return [dtype for _, dtype in target_config.dspark_trailing_inputs]


def batch_trailing_buffers(target_inputs: object) -> list[Buffer]:
    """Returns the extras the trunk's batch processor prepared."""
    assert isinstance(target_inputs, DeepseekV41Inputs)
    return list(target_inputs.extras)


def extra(target_config: Any, trailing: Sequence[Value[Any]]) -> dict[str, Any]:
    """Returns the trunk's extras by name, as its verify reads them."""
    assert isinstance(target_config, DeepseekV41Config)
    names = [name for name, _ in target_config.dspark_trailing_inputs]
    return dict(zip(names, trailing, strict=True))


ADAPTER = TargetAdapter(
    primary_kv=WINDOW_LEAF,
    verify=verify,
    block_proposer=block_proposer,
    ep_input_types=ep_input_types,
    # The graph takes no in-thinking input, as the fused graph the serving
    # engine binds doesn't.
    relaxed_thinking_acceptance=False,
    prepares_every_batch=True,
    trailing_input_types=trailing_input_types,
    batch_trailing_buffers=batch_trailing_buffers,
    extra=extra,
)
