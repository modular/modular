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

"""Beam's Harmony wire format, as its ``response_template`` declares it.

An assistant turn is a run of messages. The generation prompt ends at
``<|start|>assistant``, and the model writes the rest::

    <|channel|>analysis<|message|>REASONING<|end|>
    <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>

A tool call is a message of its own on the ``analysis`` channel, addressed to
the tool, whose body is ``rfl:parameter`` XML::

    <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>
    <rfl:parameter name="KEY">
    VALUE
    </rfl:parameter><|call|>

Parallel calls end with ``<|end|>`` and the last with ``<|call|>``. Every
marker is a single token marked ``special: false``, so decoding keeps its
text; ``<|return|>`` and ``<|call|>`` are also the end-of-sequence tokens.

These are the same literals as ``formatspec.json``, the format's FormatSpec
document for the serving engine; an edit to one belongs in the other.
"""

from __future__ import annotations

import re

START = "<|start|>"
END = "<|end|>"
MESSAGE = "<|message|>"
CHANNEL = "<|channel|>"
CONSTRAIN = "<|constrain|>"
RETURN = "<|return|>"
CALL = "<|call|>"
ROLE = "assistant"

REASONING_OPEN = CHANNEL + "analysis" + MESSAGE
ANSWER_OPEN = CHANNEL + "final" + MESSAGE
TOOL_HEADER_PREFIX = " to="
TOOL_HEADER_SUFFIX = CHANNEL + "analysis " + CONSTRAIN + "xml" + MESSAGE
TOOL_HEADER = re.compile(
    re.escape(TOOL_HEADER_PREFIX)
    + r"(?P<name>[^\s<]+)"
    + re.escape(TOOL_HEADER_SUFFIX)
)

# The response template closes reasoning on any of these, a tool call on
# ``<|end|><|start|>assistant`` or ``<|call|>``, and an answer on
# ``<|return|>``, ``<|end|>`` or ``<|call|>``.
REASONING_CLOSES = (END + START + ROLE, CONSTRAIN + START + ROLE, START + ROLE)
TOOL_CLOSE = END + START + ROLE

PARAMETER = re.compile(
    r'<rfl:parameter name="(?P<key>[^"]+)">\n(?P<value>.*?)\n'
    r'(?:</rfl:parameter>|(?=<rfl:parameter name="))',
    re.DOTALL,
)

_NAME = re.compile(r"[^\s<]*")


def _may_grow_into_tool_header(text: str) -> bool:
    if len(text) <= len(TOOL_HEADER_PREFIX):
        return TOOL_HEADER_PREFIX.startswith(text)
    if not text.startswith(TOOL_HEADER_PREFIX):
        return False
    rest = text[len(TOOL_HEADER_PREFIX) :]
    name = _NAME.match(rest)
    assert name is not None
    tail = rest[name.end() :]
    return not tail or (name.end() > 0 and TOOL_HEADER_SUFFIX.startswith(tail))


def tool_header_start(text: str) -> int | None:
    """Returns where the suffix of ``text`` that may grow into a tool header begins.

    Returns ``None`` when no suffix can. A header holds two spaces, the one
    that leads it and the one before ``<|constrain|>``, so it starts at one of
    the last two.
    """
    first = text.rfind(" ")
    if first < 0:
        return None
    second = text.rfind(" ", 0, first)
    for start in (second, first):
        if start >= 0 and _may_grow_into_tool_header(text[start:]):
            return start
    return None
