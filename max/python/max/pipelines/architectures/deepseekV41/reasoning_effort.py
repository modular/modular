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
"""Reasoning-effort normalization for DeepSeek-V4.1-Flash.

V4.1 conditions thinking-mode prompts on a numeric reasoning-effort budget
in [1, 100]. This module turns a request's chat-template options into a
thinking mode and that budget, accepting either a number or a named effort.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from max.pipelines.context.exceptions import InputError

#: Named efforts and their budgets; ``"xhigh"`` is an alias of ``"high"``.
NAMED_EFFORTS: Mapping[str, int] = {
    "low": 50,
    "high": 75,
    "xhigh": 75,
    "max": 100,
}

#: Budget used when no effort was requested.
DEFAULT_EFFORT_BUDGET = NAMED_EFFORTS["high"]


def _resolve_budget(effort: object) -> int:
    """Resolves a raw ``reasoning_effort`` value to a budget in [1, 100].

    Names always resolve to an int, so the encoder never sees a name it
    does not know (such as ``"xhigh"``).

    Raises:
        InputError: If ``effort`` is not an in-range int or a name in
            ``NAMED_EFFORTS``. This rejects ``"medium"``, ``"minimal"`` and
            ``"none"``.
    """
    if effort is None:
        return DEFAULT_EFFORT_BUDGET
    if type(effort) is int and 1 <= effort <= 100:
        return effort
    if isinstance(effort, str) and effort in NAMED_EFFORTS:
        return NAMED_EFFORTS[effort]
    raise InputError(
        f"Invalid reasoning_effort {effort!r}. Send "
        'chat_template_kwargs={"reasoning_effort": <int 1-100 or '
        f'{"/".join(NAMED_EFFORTS)}>}}, or "none" to disable thinking.'
    )


def resolve_reasoning_effort(
    chat_template_options: Mapping[str, Any],
) -> tuple[str, int | None]:
    """Resolves a request's thinking mode and numeric reasoning-effort budget.

    Thinking is on unless ``thinking`` or ``enable_thinking`` is given and
    false. ``reasoning_effort="none"`` selects chat mode.

    Args:
        chat_template_options: Request chat-template kwargs. Reads
            ``thinking``, ``enable_thinking``, and ``reasoning_effort``
            (an int, one of ``NAMED_EFFORTS``, ``"none"``, or absent/``None``).

    Returns:
        ``(thinking_mode, budget)``: ``thinking_mode`` is ``"thinking"`` or
        ``"chat"``; ``budget`` is an int in [1, 100] in thinking mode, else
        ``None``.

    Raises:
        InputError: If ``reasoning_effort`` is not ``None``, ``"none"``, an
            int in [1, 100], or a name in ``NAMED_EFFORTS``. Checked even
            outside thinking mode.
    """
    if (
        "thinking" in chat_template_options
        or "enable_thinking" in chat_template_options
    ):
        thinking = bool(chat_template_options.get("thinking")) or bool(
            chat_template_options.get("enable_thinking")
        )
    else:
        thinking = True

    effort = chat_template_options.get("reasoning_effort")
    if effort == "none":
        return "chat", None

    budget = _resolve_budget(effort)
    thinking_mode = "thinking" if thinking else "chat"
    return thinking_mode, (budget if thinking else None)
