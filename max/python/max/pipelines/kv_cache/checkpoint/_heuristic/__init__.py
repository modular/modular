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
"""Heuristics that decide where to checkpoint a request."""

from __future__ import annotations

from typing import Protocol

from .._triggers import Triggers
from ._at_event import AtEventHeuristic, end_of_prompt, release


class Heuristic(Protocol):
    """Decides where to checkpoint."""

    def subscribe(self, triggers: Triggers) -> None:
        """Registers when this heuristic runs."""
        ...


__all__ = ["AtEventHeuristic", "Heuristic", "end_of_prompt", "release"]
