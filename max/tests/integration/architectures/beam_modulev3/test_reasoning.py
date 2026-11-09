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
"""Beam's reasoning parser, over a stand-in vocabulary.

Every Harmony marker is one token, as in the checkpoint's vocabulary; other
text is one token per word or run of spaces.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pytest
from max.pipelines.architectures.beam_modulev3 import harmony
from max.pipelines.architectures.beam_modulev3.reasoning import (
    BeamReasoningParser,
)

_MARKERS = (
    harmony.START,
    harmony.END,
    harmony.MESSAGE,
    harmony.CHANNEL,
    harmony.CONSTRAIN,
    harmony.RETURN,
    harmony.CALL,
)
_SPLIT = re.compile(
    "(" + "|".join(re.escape(m) for m in _MARKERS) + r")|(\s*[^\s<]+|\s+|<)"
)


class _Vocab:
    def __init__(self) -> None:
        self.texts: list[str] = list(_MARKERS)

    def encode(self, text: str) -> list[int]:
        ids = []
        for match in _SPLIT.finditer(text):
            piece = match.group()
            if piece not in self.texts:
                self.texts.append(piece)
            ids.append(self.texts.index(piece))
        assert self.decode(ids) == text
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.texts[i] for i in ids)


R = harmony.REASONING_OPEN
A = harmony.END + harmony.START + harmony.ROLE + harmony.ANSWER_OPEN
CALL = (
    harmony.END
    + harmony.START
    + harmony.ROLE
    + " to=get_weather"
    + harmony.TOOL_HEADER_SUFFIX
    + '<rfl:parameter name="city">\nParis\n</rfl:parameter>'
)


def _parse(text: str, eos: str | None, chunk: int) -> tuple[str, str]:
    vocab = _Vocab()
    ids = vocab.encode(text) + (vocab.encode(eos) if eos else [])
    parser = BeamReasoningParser(list(range(len(_MARKERS))), vocab.decode)
    parser.reset()
    still = parser.will_reason_after_prompt(
        vocab.encode(harmony.START + harmony.ROLE)
    )
    reasoning, content = [], []
    for i in range(0, len(ids), chunk):
        delta = ids[i : i + chunk]
        # Log-probabilities ride along: a held token's must stay with it.
        logprobs = [float(t) for t in delta]
        parsed = parser.stream(delta, is_currently_reasoning=still)
        still = parsed.is_still_reasoning
        content_ids = parsed.span.extract_content(delta)
        assert parsed.span.extract_content(logprobs) == [
            float(t) for t in content_ids
        ]
        reasoning_ids = parsed.span.extract_reasoning(delta)
        reasoning.append(vocab.decode(reasoning_ids))
        content.append(vocab.decode(content_ids))
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize("chunk", [1, 2, 3, 1000])
@pytest.mark.parametrize(
    "text, eos, reasoning, content",
    [
        (
            R + "Add them." + A + "2 + 2 = 4.",
            "<|return|>",
            "Add them.",
            "2 + 2 = 4.",
        ),
        # The output can end inside reasoning; nothing is answer then.
        (
            R + "Add them. The answer is 4.",
            "<|return|>",
            "Add them. The answer is 4.",
            "",
        ),
        (R + "Add them.<|end|>", "<|return|>", "Add them.", ""),
        (R + "Add them.<|end|>", None, "Add them.", ""),
        # Only <|end|><|start|>assistant (or <|start|>assistant) closes it.
        (
            R + "a<|end|>b<|start|>user c" + A + "d",
            "<|return|>",
            "a<|end|>b<|start|>user c",
            "d",
        ),
        (
            R + "a<|end|><|end|><|start|>assistant" + harmony.ANSWER_OPEN + "d",
            "<|return|>",
            "a<|end|>",
            "d",
        ),
        (
            R + "a" + harmony.START + harmony.ROLE + harmony.ANSWER_OPEN + "d",
            "<|return|>",
            "a",
            "d",
        ),
        (R + "a<|start|>", "<|return|>", "a<|start|>", ""),
        # Text outside a message, and an unknown channel's, is dropped.
        ("junk " + R + "a" + A + "d<|end|> trailing", "<|return|>", "a", "d"),
        ("<|channel|>commentary<|message|>c" + A + "d", "<|return|>", "", "d"),
        (R + A + "d", "<|return|>", "", "d"),
        # A call message goes to content whole, for the tool parser.
        (
            R + "a" + CALL,
            "<|call|>",
            "a",
            CALL[len(harmony.END + harmony.START + harmony.ROLE) :]
            + "<|call|>",
        ),
        (
            R + "a" + A + "Done." + CALL,
            "<|call|>",
            "a",
            "Done."
            + CALL[len(harmony.END + harmony.START + harmony.ROLE) :]
            + "<|call|>",
        ),
    ],
)
def test_routes_each_message(
    text: str, eos: str | None, reasoning: str, content: str, chunk: int
) -> None:
    assert _parse(text, eos, chunk) == (reasoning, content)


def test_a_prompt_that_opens_reasoning_starts_in_it() -> None:
    prompt = harmony.START + harmony.ROLE + R
    vocab = _Vocab()
    parser = BeamReasoningParser(list(range(len(_MARKERS))), vocab.decode)
    parser.reset()
    assert parser.will_reason_after_prompt(vocab.encode(prompt))
    ids = vocab.encode("more" + A + "d")
    parsed = parser.stream(ids, is_currently_reasoning=True)
    assert vocab.decode(parsed.span.extract_reasoning(ids)) == "more"
    assert vocab.decode(parsed.span.extract_content(ids)) == "d"


def test_a_prompt_that_opens_the_answer_does_not_reason() -> None:
    vocab = _Vocab()
    parser = BeamReasoningParser(list(range(len(_MARKERS))), vocab.decode)
    parser.reset()
    prompt = vocab.encode(harmony.START + harmony.ROLE + harmony.ANSWER_OPEN)
    assert not parser.will_reason_after_prompt(prompt)
    ids = vocab.encode("d")
    parsed = parser.stream(ids, is_currently_reasoning=False)
    assert vocab.decode(parsed.span.extract_content(ids)) == "d"


def test_a_prompt_without_an_assistant_turn_is_all_answer() -> None:
    # A raw /v1/completions prompt: the model continues plain text.
    vocab = _Vocab()
    parser = BeamReasoningParser(list(range(len(_MARKERS))), vocab.decode)
    parser.reset()
    prompt = vocab.encode("Question: what is 2+2? Answer:")
    still = parser.will_reason_after_prompt(prompt)
    assert not still
    text = " 4. The answer is 4."
    ids = vocab.encode(text)
    parsed = parser.stream(ids, is_currently_reasoning=still)
    assert vocab.decode(parsed.span.extract_content(ids)) == text
    assert parsed.span.extract_reasoning(ids) == []
