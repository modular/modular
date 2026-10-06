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

"""Chat template kwargs that toggle model thinking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

THINKING_TOGGLE_KEYS = ("enable_thinking", "thinking")
"""Chat templates disagree on the name of the thinking toggle."""


def thinking_requested(kwargs: Mapping[str, Any]) -> bool | None:
    """Reads the thinking toggle from chat template kwargs.

    Returns:
        The toggle under whichever spelling is set, or ``None`` when neither is.
    """
    for key in THINKING_TOGGLE_KEYS:
        if key in kwargs:
            return bool(kwargs[key])
    return None


def with_thinking(kwargs: Mapping[str, Any], enabled: bool) -> dict[str, Any]:
    """Returns ``kwargs`` with every spelling of the thinking toggle set."""
    return {**kwargs, **dict.fromkeys(THINKING_TOGGLE_KEYS, enabled)}


def without_thinking(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Returns ``kwargs`` with thinking off, refusing a request to turn it on.

    Raises:
        ValueError: If either spelling of the toggle is set to anything but
            false.
    """
    for key in THINKING_TOGGLE_KEYS:
        if key in kwargs and kwargs[key] is not False:
            raise ValueError(
                f"chat_template_kwargs sets {key!r} to {kwargs[key]!r}, but "
                "it must be false or unset here"
            )
    return with_thinking(kwargs, False)
