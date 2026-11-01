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
"""Declared aliasing between a unified spec-decode draft and its target."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from dataclasses import dataclass

__all__ = ["NO_ALIASES", "DraftAliases", "validate_draft_state_dict"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftAliases:
    """The weights a draft inherits from its target instead of loading."""

    always: tuple[str, ...] = ()
    """Prefixes the draft inherits regardless of what the checkpoint holds."""

    when_absent: tuple[str, ...] = ()
    """Prefixes inherited only when the checkpoint provides no weight for them."""

    def resolve(self, provided: Collection[str]) -> tuple[str, ...]:
        """Returns the prefixes actually aliased for one draft checkpoint."""
        detected = tuple(
            prefix
            for prefix in self.when_absent
            if not any(name.startswith(prefix) for name in provided)
        )
        return self.always + detected


NO_ALIASES = DraftAliases()
"""A draft that loads every weight itself."""


def validate_draft_state_dict(
    expected: Iterable[str],
    provided: Collection[str],
    aliases: DraftAliases = NO_ALIASES,
) -> tuple[str, ...]:
    """Checks a draft checkpoint covers every weight the draft module needs.

    Args:
        expected: Weight names the draft module declares, typically
            ``draft.raw_state_dict().keys()``.
        provided: Weight names the draft checkpoint supplies.
        aliases: Modules the draft inherits from the target, which the
            checkpoint is therefore not expected to supply.

    Returns:
        The prefixes resolved for this checkpoint, for the caller to reuse
        when prefixing the draft's weights with ``draft.``.

    Raises:
        ValueError: If the draft expects a weight that is neither aliased from
            the target nor present in the checkpoint.
    """
    expected_set = set(expected)
    provided_set = set(provided)
    aliased = aliases.resolve(provided_set)

    missing = {
        name
        for name in expected_set - provided_set
        if not name.startswith(aliased)
    }
    if missing:
        raise ValueError(
            f"Draft model has unloaded non-shared weights: {sorted(missing)}"
        )

    extra = provided_set - expected_set
    if extra:
        logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

    return aliased
