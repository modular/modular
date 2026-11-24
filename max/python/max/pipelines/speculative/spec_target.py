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
"""What a spec-decode driver's verify callable returns.

A target is the model being verified, so nothing here depends on how the draft
proposes. The drafting strategy shows up in one place only: which batch type
the driver hands its verify callable.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Generic

from max import tree
from max.graph import TensorValue
from typing_extensions import TypeVar

__all__ = ["DistributedVerify", "Verified"]

_TargetHiddenT = TypeVar("_TargetHiddenT")
"""The target's hidden-state payload, opaque to whichever driver carries it."""

_LogitsT = TypeVar("_LogitsT", default=TensorValue)
"""The logits' value type: a graph value, or an experimental ``Tensor``."""


@tree.dataclass(frozen=True)
class Verified(Generic[_TargetHiddenT, _LogitsT]):
    """What the target contributes: verification logits and hidden states.

    Naming the two results is the point. The target output tuple is
    ``[last_logits?][logits, offsets?][hidden per device...]``, and whether the
    leading entry exists depends on ``emit_last_token_logits`` -- which is why
    structurally identical code reads ``[1]`` and ``[3 : 3 + n]`` in one arch
    and ``[0]`` and ``[2 : 2 + n]`` in another. The adapter owns that layout.
    """

    logits: _LogitsT
    hidden: _TargetHiddenT
    roll_forward: Callable[[Any], None] | None = None
    """Lands state no cache length rewinds on the accepted prefix, for a
    target that keeps any. The driver calls it with its batch once the
    accepted count is set, before the draft runs."""
    replica_logits: list[_LogitsT] | None = None
    """The verify logits each data-parallel replica's leader holds for its own
    rows, ``[rows * (num_steps + 1), vocab]``, for a driver that judges the
    proposals where the rows live. ``None`` when only :attr:`logits` exists."""


class DistributedVerify(partial[Verified[Any]]):
    """A verify that reads the distributed inputs.

    Those are the host row offsets, data-parallel splits, per-device context
    lengths and expert-parallel buffers. A fused graph declares them only
    when its verify or its proposer reads them.
    """

    takes_distributed_inputs = True
