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
"""How many of its carried drafts a speculative step verifies."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .adaptive_width import parse_adaptive_widths
from .config import SpeculativeConfig
from .depth_schedule import build_depth_lookup

__all__ = ["SpecWidthPolicy"]

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class SpecWidthPolicy:
    """How many of its carried drafts a speculative step verifies.

    Resolved once from the config, so every setting that bears on a step's
    width is combined in one place.
    """

    num_speculative_tokens: int
    """The configured draft depth, which caps every width."""

    by_batch_size: Sequence[int] | None = None
    """Dense ``batch_size -> width``, index 0 unused. ``None`` verifies every
    carried draft at every batch size."""

    mixed: int | None = None
    """The width a mixed prefill+decode step verifies: ``0`` when mixed steps
    do not verify, ``None`` to follow the batch-size schedule."""

    adaptive: Sequence[int] | None = None
    """The widths a decode step chooses among by measured tokens per second,
    ascending. ``None`` keeps the width static."""

    adaptive_min_batch_size: int = 1
    """Smallest decode batch size that captures every adaptive width. Smaller
    batches capture only the widest."""

    @classmethod
    def from_config(
        cls,
        spec_config: SpeculativeConfig | None,
        num_speculative_tokens: int,
        max_batch_size: int,
        *,
        mixed_steps_verify: bool,
    ) -> SpecWidthPolicy:
        """Resolves the policy for a pipeline.

        Args:
            spec_config: The pipeline's speculative config, which carries the
                schedule or the adaptive widths, and the mixed-batch width.
            num_speculative_tokens: The configured draft depth, which caps
                every width. For block drafters this is the checkpoint's fixed
                block width. The schedule narrows only how much of that block
                the target verifies, not how much the draft produces.
            max_batch_size: Largest decode batch size the schedule must cover.
            mixed_steps_verify: Whether a mixed prefill+decode step verifies
                drafts at all.

        Returns:
            The resolved policy.
        """
        by_batch_size: list[int] | None = None
        adaptive: list[int] | None = None
        adaptive_min_batch_size = 1
        configured_mixed: int | None = None
        if spec_config is not None and num_speculative_tokens > 0:
            # Parsed here rather than at config read because ``dflash`` reads
            # its ceiling from the draft checkpoint.
            if spec_config.adaptive_speculative_widths is not None:
                adaptive = parse_adaptive_widths(
                    spec_config.adaptive_speculative_widths,
                    num_speculative_tokens,
                )
                adaptive_min_batch_size = (
                    spec_config.adaptive_speculative_min_batch_size
                )
                if adaptive_min_batch_size > max_batch_size:
                    logger.warning(
                        "adaptive_speculative_min_batch_size=%d exceeds "
                        "max_batch_size=%d, so every step verifies %d drafts.",
                        adaptive_min_batch_size,
                        max_batch_size,
                        adaptive[-1],
                    )
            if spec_config.verify_width_schedule is not None:
                by_batch_size = build_depth_lookup(
                    spec_config.verify_width_schedule,
                    max_batch_size=max(1, max_batch_size),
                    max_depth=num_speculative_tokens,
                )
            if spec_config.num_speculative_tokens_mixed_batch is not None:
                configured_mixed = min(
                    spec_config.num_speculative_tokens_mixed_batch,
                    num_speculative_tokens,
                )
        if not mixed_steps_verify:
            if configured_mixed is not None:
                logger.warning(
                    "num_speculative_tokens_mixed_batch is set but mixed "
                    "batches do not verify drafts here, so it has no effect."
                )
            mixed: int | None = 0
        else:
            mixed = configured_mixed
        return cls(
            num_speculative_tokens=num_speculative_tokens,
            by_batch_size=by_batch_size,
            mixed=mixed,
            adaptive=adaptive,
            adaptive_min_batch_size=adaptive_min_batch_size,
        )

    @property
    def verifies_mixed_steps(self) -> bool:
        """Whether a mixed prefill+decode step verifies any drafts."""
        return self.mixed != 0

    def log_summary(self) -> None:
        """Logs every setting that narrows a step below the full draft depth."""
        if self.adaptive is not None:
            logger.info(
                "Choosing among %s drafts to verify by measured tokens per "
                "second.",
                list(self.adaptive),
            )
            if self.adaptive_min_batch_size > 1:
                logger.info(
                    "Verifying %d drafts below decode batch size %d.",
                    self.adaptive[-1],
                    self.adaptive_min_batch_size,
                )
        if self.by_batch_size is not None:
            logger.info(
                "Verifying %s of %d drafted tokens by decode batch size.",
                sorted(set(self.by_batch_size[1:])),
                self.num_speculative_tokens,
            )
        if self.mixed is not None and self.verifies_mixed_steps:
            logger.info(
                "Verifying %d of %d drafted tokens on mixed prefill+decode "
                "batches.",
                self.mixed,
                self.num_speculative_tokens,
            )

    def scheduled_width(self, batch_size: int) -> int:
        """The schedule's width at a decode batch size.

        Args:
            batch_size: The per-replica decode batch size. Sizes past the end
                of the schedule take its last width, and 0 takes the width of
                1.

        Returns:
            The width.
        """
        if self.by_batch_size is None:
            return self.num_speculative_tokens
        return self.by_batch_size[
            min(max(batch_size, 1), len(self.by_batch_size) - 1)
        ]

    def verify_width(self, batch_size: int, *, mixed_step: bool) -> int:
        """The width a step verifies, once it is known to verify at all.

        Args:
            batch_size: The per-replica decode batch size.
            mixed_step: Whether the step is a mixed prefill+decode step, which
                takes :attr:`mixed` over the schedule when it is set.

        Returns:
            The width.
        """
        if mixed_step and self.mixed is not None:
            return self.mixed
        return self.scheduled_width(batch_size)

    def reachable_widths(self) -> list[int]:
        """Every width any step could verify, sorted.

        Includes the mixed-batch width, which is reachable at any batch size.
        The constrained-decoding bitmask allocates one buffer per width from
        this, and a width missing here fails the step with "no allocated
        buffer" rather than degrading.
        """
        if self.adaptive is not None:
            widths = set(self.adaptive)
        elif self.by_batch_size is not None:
            widths = set(self.by_batch_size[1:])
        else:
            widths = {self.num_speculative_tokens}
        if self.mixed is not None:
            widths.add(self.mixed)
        return sorted(widths)

    def widths_by_batch_size(self, max_batch_size: int) -> list[list[int]]:
        """Dense ``batch_size -> widths a pure decode step may verify``.

        The table graph capture records: each batch size captures exactly its
        own row, which holds every adaptive width at or above
        :attr:`adaptive_min_batch_size` and only the widest below it. Index 0
        is unused.

        Args:
            max_batch_size: The largest decode batch size to cover.

        Returns:
            A list of length ``max_batch_size + 1``.
        """
        if self.adaptive is not None:
            return [
                self.captured_adaptive_widths(batch_size)
                for batch_size in range(max(1, max_batch_size) + 1)
            ]
        return [
            [self.scheduled_width(batch_size)]
            for batch_size in range(max(1, max_batch_size) + 1)
        ]

    def captured_adaptive_widths(self, batch_size: int) -> list[int]:
        """The adaptive widths graph capture records at a decode batch size.

        Small batches are memory-bound, so a narrower width barely saves time
        there but would still cost a captured graph per width.

        Args:
            batch_size: The per-replica decode batch size, where 0 takes the
                widths of 1.

        Returns:
            The widths, ascending.
        """
        assert self.adaptive is not None
        if max(batch_size, 1) >= self.adaptive_min_batch_size:
            return list(self.adaptive)
        return list(self.adaptive[-1:])

    def decode_widths(self, max_batch_size: int) -> list[int]:
        """The widths a pure decode step up to ``max_batch_size`` verifies.

        Args:
            max_batch_size: The largest decode batch size to cover.

        Returns:
            The distinct widths, sorted.
        """
        if self.adaptive is not None:
            return list(self.adaptive)
        if self.by_batch_size is None:
            return [self.num_speculative_tokens]
        return sorted(set(self.by_batch_size[1 : max_batch_size + 1]))
