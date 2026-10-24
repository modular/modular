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
"""The DFlash2 proposer: DFlash's block, then one path through candidates.

DFlash2 drafts its block as DFlash does. Its head keeps the target head's
top-k candidates at every drafted slot and traces one coherent path through
them, scoring each adjacent pair against the slot's hidden state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from max.graph import TensorValue

from .block_driver import Accepted, BlockBatch, DraftSampler
from .dflash import DFlashDraft, DFlashProposer

__all__ = ["DFlash2Draft", "DFlash2Proposer"]


class CandidateSelector(Protocol):
    """Scores a block's candidate pairs and walks the best path."""

    def score_edges(
        self,
        candidate_ids: TensorValue,
        unary_logits: TensorValue,
        hidden_states: TensorValue,
        anchor_token_ids: TensorValue,
    ) -> TensorValue:
        """Scores every adjacent candidate pair in the block."""
        ...

    def select_path(
        self,
        scores: TensorValue,
        candidate_ids: TensorValue,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> TensorValue:
        """Walks the scores left to right, following the best successor."""
        ...


@runtime_checkable
class DFlash2Draft(DFlashDraft, Protocol):
    """What the DFlash2 proposer calls on its draft, beyond DFlash's calls."""

    @property
    def candidate_selector(self) -> CandidateSelector:
        """The draft's path selector."""
        ...

    def compute_candidates(
        self, logits: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Returns each slot's top-k candidate ids and their logits."""
        ...


class DFlash2Proposer(DFlashProposer):
    """Drafts a DFlash2 block through the target's embedding and head."""

    draft: DFlash2Draft

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores candidate paths through the selector and samples the best."""
        drafted = self._drafted(batch, block_hs)
        candidate_ids, unary_logits = self.draft.compute_candidates(
            self._logits(batch, drafted)
        )
        selector = self.draft.candidate_selector
        (hidden_states,) = drafted
        scores = selector.score_edges(
            candidate_ids, unary_logits, hidden_states, accepted.next_tokens
        )
        return selector.select_path(
            scores,
            candidate_ids,
            sampler=lambda row, step: sampler.sample_next(
                row, step, token_ids=candidate_ids[:, step]
            ),
        ).rebind(["batch_size", self.num_drafts])
