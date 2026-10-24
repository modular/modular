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
"""DeepSeek-V4's call, and the DSpark stages its checkpoint ships.

The stages are no second model with a cache of their own: their windows are
layers ``num_hidden_layers ..`` of the trunk's window leaf, and they embed
and project through the trunk's embedding and head. So the draft's cache is
that leaf, at the two lengths the driver gives it, and the proposer runs the
stages on the trunk's own per-device copies, which only the verify here
makes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.graph import DeviceRef, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
)
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

PRIMARY_KV = "swa"
"""The window leaf, which holds the DSpark stages' windows too."""


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a DeepSeek-V4 trunk, returning DSpark's states.

    Raises:
        ValueError: If the trunk wasn't built to return selected layers,
            which only block drafts read.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A DeepSeek-V4 target verifies only for block drafts, which read"
            f" selected layers. Got {hidden_states.value}."
        )
    return _Verify(target)


def block_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[list[TensorValue], TensorValue]:
    """Returns the proposer of the DSpark stages, on the verify's trunk."""
    del speculative
    assert isinstance(verify, _Verify)
    return _DSparkProposer(verify, draft, draft_config)


def _leaves(
    batch: BlockBatch, window: list[PagedCacheValues]
) -> dict[str, list[PagedCacheValues]]:
    """The trunk's leaves by key, the window leaf as given."""
    return {PRIMARY_KV: window, **batch.passthrough_kv}


class _Verify:
    """The trunk over the merged prompt and draft tokens.

    The verify forward writes every merged token into every leaf. The rows
    past what the accept commits are written at positions the next iteration
    writes again before anything reads them: every read is at a position
    below the reader's own, and the next forward starts at the committed
    length.
    """

    def __init__(self, target: Any) -> None:
        self.target = target
        self._replicas: list[Any] | None = None

    def replicas(self, devices: Sequence[DeviceRef]) -> list[Any]:
        """The trunk, or one copy of it per device.

        The copies are made once, in the graph, after the weights are
        loaded, as the trunk requires.
        """
        if self._replicas is None:
            self._replicas = (
                [self.target]
                if len(devices) == 1
                else self.target.tensor_parallel_replicas(devices)
            )
        return self._replicas

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        assert isinstance(batch, BlockBatch)
        outputs = self.target.serve_paged(
            self.replicas(batch.devices),
            broadcast_per_device(
                batch.merged_tokens, batch.signal_buffers, batch.n_devs
            ),
            list(batch.merged_offsets_per_dev),
            batch.return_n_logits,
            _leaves(batch, batch.kv_collections),
            batch.signal_buffers,
        )
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors()),
        )


class _DSparkProposer:
    """The DSpark stages: every block position drafts, anchor included.

    The trunk's embedding embeds the block inside the stages, which also
    route on its ids, so :meth:`embed_block` hands the ids through.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(self, verify: _Verify, draft: Any, draft_config: Any) -> None:
        self._verify = verify
        self.draft = draft
        self.block_size = draft_config.block_size
        self.mask_token_id = draft_config.mask_token_id
        self._stages: list[Any] | None = None

    def stages(self, devices: Sequence[DeviceRef]) -> list[Any]:
        """The stages, or one copy of them per device."""
        if self._stages is None:
            self._stages = (
                [self.draft]
                if len(devices) == 1
                else self.draft.tensor_parallel_replicas(devices)
            )
        return self._stages

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        self.draft.fill_cache(
            self.stages(batch.devices),
            target_hidden,
            list(batch.merged_offsets_per_dev),
            _leaves(batch, ctx_kv),
        )

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
        trunks = self._verify.replicas(batch.devices)
        return self.draft.block(
            self.stages(batch.devices),
            [trunk.embed for trunk in trunks],
            embeds,
            offsets,
            _leaves(batch, block_kv),
            batch.signal_buffers,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        drafts, _ = self.stages(batch.devices)[0].head(
            block_hs,
            accepted.next_tokens,
            self._verify.replicas(batch.devices)[0].head,
            sampler=sampler.sample_next,
        )
        return drafts


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    block_proposer=block_proposer,
)
