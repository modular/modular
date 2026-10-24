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
"""Adapts DeepSeek-V4.1-Flash and its DSpark drafter to the block driver.

The drafter is not a separate model with its own cache: its stages live in
the same module as the target model and store their context KV in the target
model's sliding-window cache. Each step verifies the previous drafts with the
target model, stores the drafter's context for the verified tokens, and
drafts the next block.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.comm.ep import EPBatchManager
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import ReturnLogits, logits_postprocess
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from ..deepseekV41.deepseekV41 import (
    DeepseekV41,
    DeepseekV41Trunk,
    VisionMerge,
    gather_rows,
)
from ..deepseekV41.dspark_drafter import (
    DeepseekV41DSparkDrafter,
    DeepseekV41WithDSpark,
)
from ..deepseekV41.kv_leaves import V41KVLeafSpec
from ..deepseekV41.layers.attention import CSA2Cache
from ..deepseekV41.layers.ragged import ReplicaTokens
from ..deepseekV41.layers.ragged import V41RaggedRows as RaggedRows
from ..deepseekV41.model import csa2_cache
from ..deepseekV41.model_config import DeepseekV41Config

__all__ = [
    "ENGRAM_LOOKBACK",
    "WINDOW_LEAF",
    "DSparkDeepseekV41Proposer",
    "DeepseekV41BlockTarget",
    "TrunkTaps",
    "leaf_caches",
]

WINDOW_LEAF = "swa"
"""Name of the sliding-window KV cache, which also holds the drafter's KV."""

ENGRAM_LOOKBACK = "engram_lookback_ids"
"""Key in :attr:`BlockBatch.extra` for the Engram lookback window.

The window is ``[batch, max_ngram_size - 1]`` int64 on device 0, and is
absent when the config has no Engram layers.
"""

TrunkTaps = tuple[list[TensorValue], list[RaggedRows]]
"""Per device: captured target-model hidden states and their row layout.

The states are ``[1, T, d * n_taps]``; the rows describe the verification
pass that produced them.
"""


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
        list(window) if spec.key == WINDOW_LEAF else list(others[spec.key])
        for spec in specs
    ]
    return [csa2_cache(specs, groups, i) for i in range(len(window))]


class DeepseekV41BlockTarget:
    """Runs the target model over prompt and draft tokens for verification.

    ``replicas`` is the model, or its per-device copies under tensor
    parallelism; empty for a signature-only driver, which never builds a
    forward.
    """

    def __init__(
        self,
        model: DeepseekV41WithDSpark | None,
        config: DeepseekV41Config,
        specs: Sequence[V41KVLeafSpec],
    ) -> None:
        self.replicas: Sequence[DeepseekV41WithDSpark] = (
            [model] if model is not None else []
        )
        self.config = config
        self.specs = list(specs)

    def verify(self, batch: BlockBatch) -> Verified[TrunkTaps]:
        """Runs the target model from each request's cached length.

        Writes every prompt and draft token into every KV cache and returns
        the logits at every position, plus the hidden states entering each
        DSpark target layer. Rows written past what is accepted are never
        read before the next step overwrites them: that step starts at the
        first unaccepted position, and every read is below the reader's own
        position.
        """
        caches = leaf_caches(
            self.specs, batch.kv_collections, batch.passthrough_kv
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
        if batch.data_parallel_degree > 1:
            return self._verify_data_parallel(batch, caches, visions)
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
        lookback = batch.extra.get(ENGRAM_LOOKBACK)
        lead = self.replicas[0]
        tap_layer_ids = self.config.dspark_target_layer_ids
        if len(self.replicas) == 1:
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
                    [replica.trunk for replica in self.replicas],
                    tokens,
                    rows,
                    caches,
                    batch.signal_buffers,
                    tap_layer_ids=tap_layer_ids,
                    engram_operands=DeepseekV41._engram_operands_tensor_parallel(
                        self.replicas,
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
        return Verified(
            logits=outputs.logits["logits"], hidden=(taps_per_dev, rows)
        )

    def _verify_data_parallel(
        self,
        batch: BlockBatch,
        caches: Sequence[CSA2Cache],
        visions: Sequence[VisionMerge],
    ) -> Verified[TrunkTaps]:
        """Runs :meth:`verify` with each device on its own replica's requests.

        The merged batch is replica-major; each device's taps and rows cover
        its own requests, and the logits cover the whole batch on device 0.
        ``visions`` hold the whole batch's image rows on every device, and
        each device merges its own.
        """
        assert batch.host_merged_offsets is not None
        splits = batch.data_parallel_splits
        assert splits is not None
        manager = self.replicas[0].ep_batch_manager
        if manager is not None:
            assert batch.ep_inputs is not None
            manager.fetch_buffers(batch.ep_inputs)
        lookback = batch.extra.get(ENGRAM_LOOKBACK)
        split = DeepseekV41.split_replicas(
            self.replicas,
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
        xs, taps = DeepseekV41WithDSpark.trunk_with_taps_data_parallel(
            self.replicas,
            split,
            caches,
            batch.signal_buffers,
            visions=[
                vision.for_rows(offsets, splits[i], splits[i + 1])
                for i, (vision, offsets) in enumerate(
                    zip(visions, batch.merged_offsets_per_dev, strict=True)
                )
            ],
        )
        lead = self.replicas[0]
        outputs = logits_postprocess(
            gather_rows(xs, batch.merged_tokens.shape[0], batch.signal_buffers),
            batch.merged_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return Verified(
            logits=outputs.logits["logits"], hidden=(taps, split.rows)
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.config.ep_config is None:
            return ()
        return EPBatchManager(self.config.ep_config).input_types()


class DSparkDeepseekV41Proposer:
    """Drafts a block of tokens with the DSpark stages.

    Every block position drafts, including the first (the last accepted
    token). The drafter has no embedding of its own and uses the target
    model's, so :meth:`embed_block` passes the token ids through and
    :meth:`forward_block` embeds them.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self, model: DeepseekV41WithDSpark | None, config: DeepseekV41Config
    ) -> None:
        if config.dspark_noise_token_id is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs dspark_noise_token_id to fill the "
                "draft block"
            )
        self.replicas: Sequence[DeepseekV41WithDSpark] = (
            [model] if model is not None else []
        )
        self.config = config
        self.block_size = config.dspark_block_size
        self.mask_token_id = config.dspark_noise_token_id

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: TrunkTaps,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Stores each stage's context KV for every verified token."""
        taps, rows = target_hidden
        for replica, tap, row, swa in zip(
            self.replicas, taps, rows, ctx_kv, strict=True
        ):
            replica.dspark.store_context(tap, row, swa)

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
        rows = [
            RaggedRows.from_offsets(
                off, ids.shape[1], ops.cast(swa.cache_lengths, DType.int32)
            )
            for off, ids, swa in zip(offsets, embeds, block_kv, strict=True)
        ]
        windows = [
            replica.dspark.windows(row.starts, swa)
            for replica, row, swa in zip(
                self.replicas, rows, block_kv, strict=True
            )
        ]
        if len(self.replicas) == 1:
            lead = self.replicas[0]
            return lead.dspark.forward_block(
                embeds[0], rows[0], windows[0], lead.trunk.embed
            )
        if batch.data_parallel_degree > 1:
            total = batch.input_row_offsets.shape[0] - 1
            ranges = self._block_ranges(batch)
            hs = DeepseekV41DSparkDrafter.forward_block_data_parallel(
                [replica.dspark for replica in self.replicas],
                embeds,
                rows,
                windows,
                [replica.trunk.embed for replica in self.replicas],
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
            [replica.dspark for replica in self.replicas],
            embeds,
            rows,
            windows,
            [replica.trunk.embed for replica in self.replicas],
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
        lead = self.replicas[0]
        logits = lead.dspark.logits(block_hs, lead.head.weight)
        sample: Callable[[TensorValue, int], TensorValue] = (
            lambda step_logits, position: sampler.sample_next(
                step_logits, position
            )
        )
        drafts, _ = lead.dspark.draft(
            logits, accepted.next_tokens, sampler=sample
        )
        return ops.cast(drafts, accepted.next_tokens.dtype)
