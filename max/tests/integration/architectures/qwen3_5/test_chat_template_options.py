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
"""Tests that Qwen3_5Tokenizer forwards every chat template option."""

from __future__ import annotations

import pytest
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import TextGenerationRequestMessage


class _RecordingDelegate:
    """Records the keyword arguments of each chat template call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> str:
        self.calls.append(kwargs)
        return "rendered"


def _render(**chat_template_options: object) -> dict[str, object]:
    # Skips __init__, which loads a real checkpoint's tokenizer and config.
    tokenizer = object.__new__(Qwen3_5Tokenizer)
    delegate = _RecordingDelegate()
    object.__setattr__(tokenizer, "delegate", delegate)
    messages = [TextGenerationRequestMessage(role="user", content="Hi")]
    rendered = tokenizer.apply_chat_template(
        messages, None, **chat_template_options
    )
    assert rendered == "rendered"
    [kwargs] = delegate.calls
    return kwargs


def test_defaults_think_and_open_the_assistant_turn() -> None:
    assert _render() == {
        "tools": None,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": False},
        {"add_generation_prompt": False},
        {"preserve_thinking": True, "custom_flag": "x"},
    ],
)
def test_caller_options_reach_the_template(
    options: dict[str, object],
) -> None:
    kwargs = _render(**options)
    for key, value in options.items():
        assert kwargs[key] == value


@pytest.mark.parametrize("key", ["chat_template", "tokenize"])
def test_rejects_options_that_replace_the_template(key: str) -> None:
    with pytest.raises(InputError, match=f"cannot set '{key}'"):
        _render(**{key: "x"})
