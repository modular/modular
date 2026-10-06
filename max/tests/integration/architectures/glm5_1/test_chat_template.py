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

"""Parity tests for GLM-5.3's linear-time tool-result ordering.

The patched template plus :func:`order_tool_results` must render exactly the
prompt the checkpoint's own template renders, because the model was trained on
that format. Each case renders both ways through HuggingFace's renderer and
compares the strings. The official template is quadratic in tool calls, so
parity is checked on small conversations and speed separately.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.pipelines.architectures.glm5_1.chat_template import (
    linearize_tool_results,
    order_tool_results,
)
from max.pipelines.architectures.glm5_1.tokenizer import GlmTokenizer
from max.pipelines.modeling.types import (
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

_OFFICIAL = (
    Path(os.environ["PIPELINES_TESTDATA"]) / "glm_5_3_chat_template.jinja"
).read_text()

_TYPED_TOOLS = [
    TextGenerationRequestTool(
        type="function",
        function=TextGenerationRequestFunction(
            name=name,
            description=f"Runs {name}.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    )
    for name in ("read", "edit", "bash")
]
_TOOLS = [dict(tool) for tool in _TYPED_TOOLS]


def _patched() -> str:
    patched = linearize_tool_results(_OFFICIAL)
    assert patched is not None
    return patched


def _render(
    template: str,
    messages: list[TextGenerationRequestMessage],
    **options: Any,
) -> str:
    rendered, _ = render_jinja_template(
        conversations=[[m.flatten_content() for m in messages]],
        tools=_TOOLS,
        chat_template=template,
        add_generation_prompt=True,
        **options,
    )
    return rendered[0]


def _call(call_id: str | None, name: str = "read") -> dict[str, Any]:
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": {"path": f"{name}.py"}},
    }
    if call_id is not None:
        call["id"] = call_id
    return call


def _assistant(*calls: dict[str, Any]) -> TextGenerationRequestMessage:
    return TextGenerationRequestMessage(
        role="assistant",
        content="Let me look.",
        reasoning_content="thinking",
        tool_calls=list(calls),
    )


def _result(
    call_id: str | None, content: str | None = None
) -> TextGenerationRequestMessage:
    return TextGenerationRequestMessage(
        role="tool",
        content=content if content is not None else f"result of {call_id}",
        tool_call_id=call_id,
    )


def _user(text: str = "Fix the bug.") -> TextGenerationRequestMessage:
    return TextGenerationRequestMessage(role="user", content=text)


def _turn(
    call_ids: list[str | None], result_ids: list[str | None]
) -> list[TextGenerationRequestMessage]:
    return [
        _user(),
        _assistant(*(_call(i) for i in call_ids)),
        *(_result(i) for i in result_ids),
    ]


_CASES: dict[str, list[TextGenerationRequestMessage]] = {
    "no_tools": [_user(), _assistant(), _user("Thanks.")],
    "in_order": _turn(["a", "b", "c"], ["a", "b", "c"]),
    "reversed": _turn(["a", "b", "c"], ["c", "b", "a"]),
    "shuffled": _turn(["a", "b", "c", "d", "e"], ["d", "a", "e", "c", "b"]),
    "subset_of_calls": _turn(["a", "b", "c", "d"], ["d", "b"]),
    "duplicate_result_id": _turn(["a", "b", "c"], ["c", "a", "c"]),
    "result_without_id": _turn(["a", "b", "c"], ["c", None, "a"]),
    "result_id_matches_no_call": _turn(["a", "b"], ["b", "z"]),
    "call_without_id": _turn(["a", None, "c"], ["c", "a"]),
    "duplicate_call_id": _turn(["a", "b", "a"], ["b", "a"]),
    "results_after_user": [_user(), _result("b"), _result("a")],
    "results_first": [_result("b"), _result("a"), _user()],
    "assistant_without_calls": [
        _user(),
        TextGenerationRequestMessage(role="assistant", content="Done."),
        _result("b"),
        _result("a"),
    ],
    "empty_result_content": [
        _user(),
        _assistant(_call("a"), _call("b")),
        _result("b", content=""),
        _result("a", content=""),
    ],
    "multi_turn": [
        TextGenerationRequestMessage(role="system", content="Be careful."),
        *_turn(["a", "b"], ["b", "a"]),
        _assistant(_call("c"), _call("d", name="edit")),
        _result("d"),
        _result("c"),
        *_turn(["e", "f"], ["f", "f"]),
        _assistant(_call("g", name="bash")),
        _result("g"),
        _user("Now run the tests."),
    ],
}


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"clear_thinking": True},
        {"clear_thinking": True, "reasoning_effort": "low"},
    ],
    ids=["defaults", "clear_thinking", "low_effort"],
)
def test_matches_the_official_template(
    case: str, options: dict[str, Any]
) -> None:
    """The patched render is byte-identical to the checkpoint's template."""
    messages = _CASES[case]
    expected = _render(_OFFICIAL, messages, **options)
    actual = _render(_patched(), order_tool_results(messages), **options)
    assert actual == expected


def test_matches_on_random_conversations() -> None:
    """Parity holds across many random mixes of the cases above."""
    rng = random.Random(4326)
    ids = list("abcdefgh")
    for _ in range(200):
        messages: list[TextGenerationRequestMessage] = [_user()]
        for _ in range(rng.randint(1, 4)):
            calls: list[str | None] = rng.sample(ids, rng.randint(1, 6))
            results: list[str | None] = rng.sample(
                calls, rng.randint(1, len(calls))
            )
            if rng.random() < 0.2:
                results.append(rng.choice([*ids, None]))
            if rng.random() < 0.1:
                calls[rng.randrange(len(calls))] = None
            messages += [
                _assistant(*(_call(i) for i in calls)),
                *(_result(i) for i in results),
            ]
            if rng.random() < 0.5:
                messages.append(_user("Continue."))
        expected = _render(_OFFICIAL, messages, clear_thinking=True)
        actual = _render(
            _patched(), order_tool_results(messages), clear_thinking=True
        )
        assert actual == expected


def test_many_tool_calls_render_quickly() -> None:
    """A turn with thousands of calls renders in well under a second.

    The official template takes about 70 s at 2,000 calls, and a degenerate
    model turn in production carried about 20,000.
    """
    n = 20_000
    call_ids = [f"call_{i}" for i in range(n)]
    result_ids = list(reversed(call_ids))
    messages = _turn(list(call_ids), list(result_ids))
    start = time.perf_counter()
    prompt = _render(_patched(), order_tool_results(messages))
    elapsed = time.perf_counter() - start
    assert prompt.count("<tool_response>") == n
    assert prompt.index("result of call_0") < prompt.index("result of call_1")
    assert elapsed < 10, f"rendering {n} tool calls took {elapsed:.1f}s"


def test_order_tool_results_keeps_other_messages_in_place() -> None:
    """Only tool-result runs move; every message is kept exactly once."""
    messages = _CASES["multi_turn"]
    ordered = order_tool_results(messages)
    assert sorted(map(id, ordered)) == sorted(map(id, messages))
    for original, reordered in zip(messages, ordered, strict=True):
        if str(original.role) != "tool":
            assert reordered is original


def test_a_changed_template_is_not_patched() -> None:
    """Any edit to the matched block leaves the template untouched."""
    edited = _OFFICIAL.replace("ns_chk.can_sort = false", "ns_chk.x = 1", 1)
    assert linearize_tool_results(edited) is None
    assert linearize_tool_results("{{ messages }}") is None


def _glm_tokenizer(tmp_path: Path, chat_template: str | None) -> GlmTokenizer:
    """A GlmTokenizer over a tiny offline vocabulary.

    Rendering never looks at the vocabulary, so this exercises the tokenizer's
    template handling without a checkpoint. With ``chat_template`` unset, the
    official template is the checkpoint's own.
    """
    delegate = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                vocab={"<unk>": 0, "<eos>": 1, "<think>": 2, "</think>": 3},
                unk_token="<unk>",
            )
        ),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    delegate.chat_template = _OFFICIAL
    delegate.save_pretrained(tmp_path)
    config = MagicMock()
    config.tokenizer_impl = None
    config.draft_model = None
    config.model.huggingface_config.eos_token_id = [1]
    config.model.generation_config.eos_token_id = [1]
    return GlmTokenizer(
        str(tmp_path), config, max_length=1 << 20, chat_template=chat_template
    )


@pytest.mark.parametrize(
    "chat_template", [None, _OFFICIAL], ids=["checkpoint", "override"]
)
def test_tokenizer_renders_like_the_official_template(
    tmp_path: Path, chat_template: str | None
) -> None:
    """GlmTokenizer patches the template and orders results before rendering.

    Covers both the checkpoint's template and an operator's
    ``--chat-template``. Results arrive out of call order, so dropping either
    the template patch or the Python ordering changes the prompt.
    """
    tokenizer = _glm_tokenizer(tmp_path, chat_template)
    assert tokenizer.delegate.chat_template == _patched()
    messages = _CASES["reversed"]
    actual = tokenizer.apply_chat_template(messages, _TYPED_TOOLS)
    assert actual == _render(_OFFICIAL, messages, clear_thinking=True)


def test_tokenizer_warns_when_it_cannot_patch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A GLM-5.3-style template MAX cannot rewrite is logged, not hidden."""
    edited = _OFFICIAL.replace("ns_chk.can_sort = false", "ns_chk.x = 1", 1)
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        tokenizer = _glm_tokenizer(tmp_path, edited)
    assert tokenizer.delegate.chat_template == edited
    assert "quadratic macros" in caplog.text
