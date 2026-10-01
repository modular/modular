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
"""Block spec-decode adapters for DeepSeek-V4 and its in-checkpoint DSpark.

The draft is not a second model with its own cache: the three DSpark stages
are submodules of :class:`DeepseekV4` and their windows are layers
``num_hidden_layers ..`` of the trunk's own window leaf. So the driver's
"draft cache" is that leaf, at the two lengths the driver gives it, and the
trunk's other leaves ride in :attr:`BlockBatch.passthrough_kv`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import ReturnLogits, logits_postprocess
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from ..deepseekV4.deepseekV4 import DeepseekV4
from ..deepseekV4.layers import DeepseekV4Cache, RaggedRows
from ..deepseekV4.model_config import DeepseekV4Config

__all__ = [
    "VERIFY_WINDOWS",
    "WINDOW_LEAF",
    "DSparkDeepseekV4Proposer",
    "DeepseekV4BlockTarget",
    "TrunkHidden",
    "leaf_caches",
]

WINDOW_LEAF = "swa"
"""The leaf the stages' windows live in; the driver's primary and draft KV."""

VERIFY_WINDOWS = "verify_windows"
"""The :attr:`BlockBatch.extra` key of the verify forward's per-ratio window
counts (:attr:`RaggedRows.windows`)."""

TrunkHidden = tuple[list[TensorValue], list[RaggedRows]]
"""Per device: the trunk states DSpark reads, ``[1, T, d * n_targets]``, and
the rows of the verify forward that produced them."""


def leaf_caches(
    config: DeepseekV4Config,
    window: Sequence[PagedCacheValues],
    others: Mapping[str, list[PagedCacheValues]],
) -> list[DeepseekV4Cache]:
    """Reassembles the trunk's leaves per device, taking the window leaf as
    given."""
    return [
        DeepseekV4Cache.from_groups(
            config,
            [
                [window[i]]
                if spec.key == WINDOW_LEAF
                else [others[spec.key][i]]
                for spec in config.kv_leaf_specs()
            ],
        )
        for i in range(len(window))
    ]


class DeepseekV4BlockTarget:
    """The trunk over the merged prompt + draft tokens.

    The verify forward writes every merged token into every leaf. The rows past
    what the accept commits are written at positions the next iteration writes
    again before anything reads them: every read is at a position below the
    reader's own, and the next forward starts at the committed length.

    ``replicas`` is the model, or its per-device copies under tensor
    parallelism (:meth:`DeepseekV4.tensor_parallel_replicas`).
    """

    def __init__(self, model: DeepseekV4, config: DeepseekV4Config) -> None:
        self.replicas: Sequence[DeepseekV4] = [model]
        self.config = config

    def verify(self, batch: BlockBatch) -> Verified[TrunkHidden]:
        caches = leaf_caches(
            self.config, batch.kv_collections, batch.passthrough_kv
        )
        t = batch.merged_tokens.shape[0]
        windows = batch.extra[VERIFY_WINDOWS]
        rows = [
            RaggedRows.from_offsets(offsets, t, cache.cache_lengths, windows)
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
        lead = self.replicas[0]
        if len(self.replicas) == 1:
            x, main_hidden = lead.trunk(tokens[0], rows[0], caches[0])
            hidden = [main_hidden]
        else:
            x, hidden = DeepseekV4.trunk_tensor_parallel(
                self.replicas, tokens, rows, caches, batch.signal_buffers
            )
        outputs = logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            batch.merged_offsets,
            batch.return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=ReturnLogits.VARIABLE,
        )
        return Verified(logits=outputs[1], hidden=(hidden, rows))

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DSparkDeepseekV4Proposer:
    """The three DSpark stages: every block position drafts, anchor included.

    The block is embedded by the trunk's own embedding inside
    :meth:`DeepseekV4.dspark_block`, which also needs the ids for the stages'
    (unread) routing input, so :meth:`embed_block` hands the ids through.
    """

    samples_from_anchor = True

    def __init__(self, model: DeepseekV4, config: DeepseekV4Config) -> None:
        self.replicas: Sequence[DeepseekV4] = [model]
        self.config = config
        self.block_size = config.dspark_block_size
        self.mask_token_id = config.dspark_noise_token_id

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: TrunkHidden,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        hidden, rows = target_hidden
        caches = leaf_caches(self.config, ctx_kv, batch.passthrough_kv)
        for replica, main_hidden, row, cache in zip(
            self.replicas, hidden, rows, caches, strict=True
        ):
            replica.fill_dspark_cache(main_hidden, row, cache)

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        return [
            ops.reshape(ids, [1, -1])
            for ids in broadcast_per_device(
                block_ids, batch.signal_buffers, batch.n_devs
            )
        ]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> TensorValue:
        caches = leaf_caches(self.config, block_kv, batch.passthrough_kv)
        rows = [
            RaggedRows.from_offsets(off, ids.shape[1], cache.cache_lengths)
            for off, ids, cache in zip(offsets, embeds, caches, strict=True)
        ]
        if len(self.replicas) == 1:
            return self.replicas[0].dspark_block(embeds[0], rows[0], caches[0])
        return DeepseekV4.dspark_block_tensor_parallel(
            self.replicas, embeds, rows, caches, batch.signal_buffers
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        drafts, _ = self.replicas[0].dspark_head(
            block_hs, accepted.next_tokens, sampler=sampler.sample_next
        )
        return drafts
