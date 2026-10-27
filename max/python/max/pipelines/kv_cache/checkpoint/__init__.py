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
"""Decides where to checkpoint a request's cache."""

from ._checkpointer import Checkpointer, CheckpointerFactory
from ._context import Context as CheckpointContext
from ._heuristic import Heuristic as CheckpointHeuristic
from ._heuristic import end_of_prompt as checkpoint_at_end_of_prompt
from ._heuristic import release as checkpoint_at_release
from ._types import Checkpoint
from ._types import Event as CheckpointEvent
from ._types import EventKind as CheckpointEventKind

__all__ = [
    "Checkpoint",
    "CheckpointContext",
    "CheckpointEvent",
    "CheckpointEventKind",
    "CheckpointHeuristic",
    "Checkpointer",
    "CheckpointerFactory",
    "checkpoint_at_end_of_prompt",
    "checkpoint_at_release",
]
