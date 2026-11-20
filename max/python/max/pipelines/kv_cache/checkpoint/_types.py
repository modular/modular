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
"""Types shared by the checkpointer, its context, and heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from max.pipelines.modeling.types import RequestID


class EventKind(Enum):
    """A point in a request's lifecycle."""

    END_OF_PROMPT = auto()
    RELEASE = auto()


@dataclass(frozen=True)
class Event:
    """Something that happened to a request at a token position."""

    request_id: RequestID
    kind: EventKind
    position: int
    """The exact token index the event happened at."""


@dataclass(frozen=True)
class Checkpoint:
    """A token position of a request to persist the cache at."""

    request_id: RequestID
    position: int
    """The exact token index to checkpoint at."""
