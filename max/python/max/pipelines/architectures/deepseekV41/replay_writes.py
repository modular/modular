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
"""Certifies eager DSpark writes without synchronizing speculative acceptance."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from max.pipelines.context import TextContext

from .replay import ReplayPlan


@dataclass(frozen=True)
class DSparkWriteWitness:
    """Associates a submitted verify/materialize with its eventual accepted prefix.

    Positions beyond ``base_end`` are conditional on later acceptance: the
    draft block overwrites the unaccepted suffix of the drafter's cache.
    """

    context: TextContext
    base_end: int
    verified_width: int
    request_id: Hashable


@dataclass(frozen=True)
class _WriteRow:
    context: TextContext
    request_id: Hashable
    start: int
    end: int
    kept_start: int
    generated: bool
    optimistic_width: int
    previous: DSparkWriteWitness | None


@dataclass(frozen=True)
class DSparkWriteBatch:
    """Freezes one eager invocation's rows and immediately prior witnesses."""

    rows: tuple[_WriteRow, ...]
    verified_width: int
    partial: bool
    leaf_id: str

    @classmethod
    def capture(
        cls,
        contexts: Sequence[TextContext],
        plan: ReplayPlan | None,
        *,
        leaf_id: str,
        verified_width: int,
        previous: Sequence[DSparkWriteWitness],
    ) -> DSparkWriteBatch:
        """Captures coordinates without asserting that a graph was submitted."""
        if plan is not None and len(plan.kept_skip) != len(contexts):
            raise ValueError("Replay plan batch differs from write batch")
        if verified_width < 0:
            raise ValueError("Verification width must be nonnegative")
        by_identity = {id(witness.context): witness for witness in previous}
        return cls(
            tuple(
                _WriteRow(
                    context=ctx,
                    request_id=ctx.request_id,
                    start=ctx.tokens.processed_length,
                    end=ctx.tokens.processed_length + ctx.tokens.active_length,
                    kept_start=(
                        ctx.tokens.processed_length + int(plan.kept_skip[i])
                        if plan is not None
                        else ctx.tokens.processed_length
                    ),
                    generated=bool(ctx.tokens.generated_length),
                    optimistic_width=len(
                        ctx.spec_decoding_state.maybe_accepted_draft_tokens
                    ),
                    previous=by_identity.get(id(ctx)),
                )
                for i, ctx in enumerate(contexts)
            ),
            verified_width,
            plan is not None,
            leaf_id,
        )

    def publish(
        self, contexts: Sequence[TextContext]
    ) -> tuple[DSparkWriteWitness, ...]:
        """Publishes coverage after the selected eager graph returns normally.

        Unsupported rows omit reports after partial mode has started. C keeps
        the old interval in that case, so new pages cannot become cache hits.
        A first unsupported trim must explicitly enter partial mode instead.
        """
        if len(contexts) != len(self.rows) or any(
            row.context is not ctx or row.request_id != ctx.request_id
            for row, ctx in zip(self.rows, contexts, strict=True)
        ):
            if self.partial:
                for ctx in contexts:
                    end = ctx.tokens.processed_length
                    ctx.kv_cache_write_ranges[self.leaf_id] = (end, end)
            return ()

        witnesses: list[DSparkWriteWitness] = []
        for row in self.rows:
            ctx = row.context
            processed = ctx.tokens.processed_length
            start: int | None = None
            if not ctx.is_done:
                if not row.generated:
                    if row.optimistic_width == 0 and processed == row.end:
                        start = row.kept_start
                elif row.end - row.start == 1:
                    previous = row.previous
                    shift = processed - row.end
                    if (
                        previous is not None
                        and previous.request_id == row.request_id
                        and previous.base_end <= row.start
                        and row.start - previous.base_end + row.optimistic_width
                        <= previous.verified_width
                        and 0 <= shift <= row.optimistic_width
                    ):
                        # Before a terminal/clipped update, host and device
                        # acceptance agree. The previous accepted prefix joins
                        # the rebased current write at row.start + shift.
                        start = previous.base_end
                    elif row.optimistic_width == 0 and shift == 0:
                        start = row.start
            if start is not None and 0 <= start <= processed:
                ctx.kv_cache_write_ranges[self.leaf_id] = (start, processed)
                witnesses.append(
                    DSparkWriteWitness(
                        ctx, processed, self.verified_width, row.request_id
                    )
                )
            elif self.partial:
                ctx.kv_cache_write_ranges[self.leaf_id] = (processed, processed)
        return tuple(witnesses)
