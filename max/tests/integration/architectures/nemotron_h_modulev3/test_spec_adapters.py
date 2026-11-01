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
"""Guards the speculative settings the Nemotron-H target adapter accepts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.pipelines.architectures.nemotron_h_modulev3.spec_adapters import (
    ADAPTER,
)


def _config(**fields: Any) -> Any:
    return cast(Any, SimpleNamespace(synthetic_acceptance_rate=None, **fields))


@pytest.mark.parametrize("draft_proposal", ["argmax", "sampled"])
def test_both_draft_proposals_are_accepted(draft_proposal: str) -> None:
    ADAPTER.check(_config(draft_proposal=draft_proposal), False)


def test_synthetic_acceptance_is_rejected() -> None:
    config = _config(draft_proposal="argmax")
    config.synthetic_acceptance_rate = 0.5
    with pytest.raises(ValueError, match="Synthetic acceptance"):
        ADAPTER.check(config, False)
