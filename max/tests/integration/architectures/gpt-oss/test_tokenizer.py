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

"""gpt-oss reasoning-effort handling, matching the orchestrator's contract."""

from __future__ import annotations

from typing import Any

import pytest
from max.pipelines.architectures.gpt_oss.tokenizer import (
    GptOssTokenizer,
    normalize_gpt_oss_reasoning_effort,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import TextTokenizer


@pytest.mark.parametrize(
    ("effort", "rendered"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        (" Max ", "high"),
        ("turbo", "turbo"),
    ],
)
def test_levels_map_onto_the_three_gpt_oss_knows(
    effort: str, rendered: str
) -> None:
    assert normalize_gpt_oss_reasoning_effort({"reasoning_effort": effort}) == {
        "reasoning_effort": rendered
    }


@pytest.mark.parametrize("effort", ["none", " None "])
def test_none_is_refused(effort: str) -> None:
    with pytest.raises(InputError, match='reasoning_effort "none"'):
        normalize_gpt_oss_reasoning_effort({"reasoning_effort": effort})


def test_no_effort_is_left_alone() -> None:
    options = {"enable_thinking": False, "thinking": False}
    assert normalize_gpt_oss_reasoning_effort(options) == options


def test_the_tokenizer_renders_the_mapped_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def render(self: Any, messages: Any, tools: Any, **options: Any) -> str:
        seen.update(options)
        return ""

    monkeypatch.setattr(TextTokenizer, "apply_chat_template", render)
    tokenizer = object.__new__(GptOssTokenizer)
    tokenizer.apply_chat_template([], None, reasoning_effort="xhigh")
    assert seen == {"reasoning_effort": "high"}
