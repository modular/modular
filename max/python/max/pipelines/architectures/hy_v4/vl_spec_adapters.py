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
"""How spec decoding calls Hy4 VL, and how its DFly2 drafter proposes.

The verify runs the target over the merged window and keeps every row's
final normalized hidden state, M-RoPE position and the top-k of the
drafter's index layers. The block then reads them at each request's last
committed row, as vLLM's ``source_row`` does. The drafter attends in the
target's own MLA cache, so it writes no context KV.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.graph_groups import MRoPEPositions
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.context import TextContext
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    BlockProposer,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    SequentialBatch,
    gather_variable_rows,
)
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
    verify_lengths,
)
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor

from .batch_processor import context_positions
from .decoder import HYV4VLDecoder
from .dfly import HYV4DFlyDrafter
from .hy_v4 import HYV4TextModel
from .model_config import HYV4VLConfig
from .spec_adapters import (
    INDEXER_TARGET_KV,
    PRIMARY_KV,
    _batch_context_length,
    _caches,
    _first_device,
    _on_mesh,
    _per_device,
    _replicated,
)

__all__ = ["ADAPTER"]

POSITION_IDS = "position_ids"
"""The key of a driver batch's ``extra`` that holds the merged window's
``[3, rows]`` M-RoPE positions."""


@dataclass(frozen=True)
class DFlyTaps:
    """What the drafter reads off the verify, for every merged row."""

    hidden: Tensor
    """``[rows, hidden]`` final normalized hidden states."""
    topk: Tensor
    """``[layers, rows, index_topk]`` int32 physical cache rows, one top-k
    per drafter layer."""
    positions: Tensor
    """``[3, rows]`` M-RoPE positions."""


class HYV4VLBlockTarget:
    """Runs the Hy4 VL target over the merged window and its image rows.

    Returns the logits of the last ``return_n_logits`` positions per request,
    on the first device, and the :class:`DFlyTaps` of every row.
    """

    # The attention blocks take the batch's context length.
    takes_distributed_inputs = True

    def __init__(
        self, target: HYV4TextModel, topk_layers: Sequence[int]
    ) -> None:
        self.target = target
        self.topk_layers = tuple(topk_layers)

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[DFlyTaps]:
        """Runs the target over the batch to score the drafts."""
        assert isinstance(batch, BlockBatch)
        mesh = self.target.embed_tokens.weight.mesh
        tokens = _on_mesh(batch.merged_tokens, mesh)
        offsets = _replicated(batch.merged_offsets_per_dev, mesh)
        positions = _on_mesh(
            ops.rebind(
                batch.extra[POSITION_IDS], [3, batch.merged_tokens.shape[0]]
            ),
            mesh,
        )
        mla, indexer = _caches(
            batch.kv_collections,
            batch.passthrough_kv[INDEXER_TARGET_KV],
            tokens.mapping,
        )
        streams, topk = self.target.tapped_streams(
            tokens,
            mla,
            indexer,
            offsets,
            _batch_context_length(batch),
            positions,
            VisionInputs(
                embeddings=_replicated(batch.vision_embeddings, mesh),
                scatter_indices=_replicated(batch.vision_scatter_indices, mesh),
            ),
            topk_layers=self.topk_layers,
        )
        hidden = self.target.head(streams)
        n = Tensor.from_graph_value(batch.return_n_logits)
        rows, _ = gather_variable_rows(hidden, offsets, n, mesh)
        return Verified(
            logits=_first_device(self.target.logits(rows)),
            hidden=DFlyTaps(hidden, F.stack(topk), positions),
        )


class HYV4DFlyProposer:
    """The DFly2 block: the target's embedding and head, the drafter's layers.

    The bonus slot does not draft; mask slot ``k`` drafts the token at
    ``anchor + k`` from its corrected hidden state and the token before it.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(self, draft: HYV4DFlyDrafter, target: HYV4TextModel) -> None:
        self.draft = draft
        self.target = target
        self.mesh = target.embed_tokens.weight.mesh
        self.block_size = draft.width
        self.mask_token_id = draft.mask_token_id
        self._taps: DFlyTaps | None = None

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: DFlyTaps,
        ctx_kv: list[Any],
    ) -> None:
        """Keeps the verify's taps for the block.

        The drafter's layers share the target's cache, which the verify
        wrote, so there is no context KV to write.
        """
        del batch, ctx_kv
        self._taps = target_hidden

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds ``block_ids`` with the target's embedding, per device."""
        del batch
        return _per_device(
            self.target.embed_tokens(_on_mesh(block_ids, self.mesh))
        )

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[Any],
    ) -> Tensor:
        """Runs the drafter over the block; returns its hidden states.

        ``block_kv``'s cache length is each request's anchor, past the
        tokens this step commits.
        """
        taps = self._taps
        assert taps is not None
        w = self.block_size
        # The last committed row of each request's merged window.
        anchor = block_kv[0].cache_lengths
        committed = anchor - ops.rebind(
            batch.kv_collections[0].cache_lengths, anchor.shape
        )
        starts = ops.rebind(batch.merged_offsets[:-1], anchor.shape)
        source = _on_mesh((starts + committed - 1).cast(DType.int64), self.mesh)

        hidden = F.gather(taps.hidden, source, axis=0)
        batch_size, hidden_size = hidden.shape
        steps = F.range(1, w + 1, dtype=DType.int64, device=self.mesh)
        # The drafter's layer subgraph takes named dims only.
        positions = F.rebind(
            (
                F.gather(taps.positions, source, axis=1).unsqueeze(-1) + steps
            ).reshape([3, -1]),
            [3, "dfly_block_rows"],
        )
        block = F.rebind(
            _replicated(embeds, self.mesh), [batch_size * w, hidden_size]
        ).reshape([batch_size, w, hidden_size])
        first = self.draft.mtp_fusion(block[:, 0], hidden)
        block = F.concat([first.unsqueeze(1), block[:, 1:]], axis=1)
        return self.draft(
            block.reshape([-1, hidden_size]),
            F.gather(taps.topk, source, axis=1),
            PagedCacheValues.from_upstream(
                block_kv_with_dispatch(block_kv, w), hidden.mapping
            ),
            self.target.rope.freqs_cis.to(self.mesh),
            positions,
            _replicated(offsets, self.mesh),
            _batch_context_length(batch),
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: Tensor,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Drafts the block's tokens one corrected slot at a time."""
        del batch
        hidden_size = block_hs.shape[-1]
        batch_size = accepted.next_tokens.shape[0]
        states = F.rebind(
            block_hs, [batch_size * self.block_size, hidden_size]
        ).reshape([batch_size, self.block_size, hidden_size])
        previous = accepted.next_tokens
        tokens = []
        for k in range(1, self.block_size):
            corrected = self.draft.hidden_correction.correct(
                states[:, k],
                self.target.embed_tokens(_on_mesh(previous, self.mesh)),
            )
            logits = _first_device(self.target.logits(corrected))
            previous = sampler.sample_next(logits, k - 1)
            tokens.append(ops.unsqueeze(previous, axis=1))
        return ops.concat(tokens, axis=1)


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> HYV4VLBlockTarget:
    """Returns the verify, tapping the top-k of the drafter's index layers."""
    del target_config, hidden_states
    assert isinstance(target, HYV4VLDecoder)
    assert isinstance(draft_config, HYV4VLConfig)
    return HYV4VLBlockTarget(
        target.language_model, draft_config.dfly.target_index_layer
    )


def block_proposer(
    verify: Any, draft: Any, draft_config: Any, speculative: SpeculativeConfig
) -> BlockProposer[DFlyTaps, Tensor]:
    """Returns the DFly2 proposer, on the verify's target."""
    del draft_config, speculative
    assert isinstance(verify, HYV4VLBlockTarget)
    assert isinstance(draft, HYV4DFlyDrafter)
    return HYV4DFlyProposer(draft, verify.target)


def _position_tree(target_config: Any) -> OrderedDict[str, object]:
    """Names the merged window's M-RoPE positions."""
    return OrderedDict(
        mrope=MRoPEPositions(
            position_ids=TensorType(
                DType.int64,
                shape=[3, "merged_total_seq_len"],
                device=target_config.devices[0],
            )
        )
    )


def _position_types(target_config: Any) -> list[TensorType | BufferType]:
    return tree.leaves(_position_tree(target_config))


def _position_extra(
    target_config: Any, trailing: Sequence[Any]
) -> dict[str, Any]:
    del target_config
    return {POSITION_IDS: trailing[0].tensor}


def _position_buffers(
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns each request's positions over its tokens and its drafts."""
    merge = target_config.vision.spatial_merge_size
    positions = np.concatenate(
        [np.empty((3, 0), np.int64)]
        + [
            context_positions(ctx, merge, length)
            for ctx, length in zip(
                contexts, verify_lengths(contexts), strict=True
            )
        ],
        axis=1,
    )
    return [Buffer.from_numpy(positions).to(devices[0])]


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    # Accepts strictly in thinking too, as Hy4's MTP pairing does.
    relaxed_thinking_acceptance=False,
    block_proposer=block_proposer,
    # The verify and the block read M-RoPE positions past the canonical
    # tail.
    trailing_input_types=_position_types,
    trailing_input_tree=_position_tree,
    extra=_position_extra,
    trailing_buffers=_position_buffers,
)
