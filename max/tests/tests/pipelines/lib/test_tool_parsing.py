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

import pytest
from max.pipelines.architectures.deepseekV3.tool_parser import (
    DeepseekV3_1ToolParser,
    DeepseekV3ToolParser,
)
from max.pipelines.architectures.gemma4.tool_parser import Gemma4ToolParser
from max.pipelines.architectures.glm5_1.tool_parser import GlmToolParser
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.architectures.minimax_m2.tool_parser import (
    MinimaxM2ToolParser,
)
from max.pipelines.lib.tool_parsing import (
    _TOOL_PARSERS,
    StructuralTagToolParser,
    create,
)
from max.serve.parser.llama_tool_parser import LlamaToolParser


def test_create_returns_registered_llama_parser() -> None:
    assert isinstance(create("llama"), LlamaToolParser)


def test_create_returns_registered_kimi_parser() -> None:
    assert isinstance(create("kimik2_5"), KimiToolParser)


def test_create_returns_registered_minimax_parser() -> None:
    assert isinstance(create("minimax_m2"), MinimaxM2ToolParser)


def test_create_unknown_parser_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool parser"):
        create("unknown_parser")


def test_llama_parser_parse_complete_smoke() -> None:
    parser = create("llama")
    parsed = parser.parse_complete(
        'text {"name":"get_weather","parameters":{"location":"Boston"}}'
    )

    assert parsed.content is None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"


# ----- Releasing content withheld as a partial start marker -----------------
#
# ``_extract_content_delta`` withholds a trailing run of the buffer that
# matches a prefix of the start marker, since the next token may complete it.
# Nothing can complete it once generation stops, so ``flush`` releases it.
# These check that a reply is not truncated when it ends in its parser's
# marker prefix.
#
# Membership below is mechanical: a parser inherits the holdback exactly when
# it subclasses ``StructuralTagToolParser``, so every such parser belongs
# here. Parsers that implement ``parse_delta`` themselves, such as Llama,
# Laguna and Qwen 3.5, withhold nothing and are unaffected.
# ``test_every_registered_structural_parser_is_covered`` keeps the list honest.

STRUCTURAL_PARSERS: list[type[StructuralTagToolParser]] = [
    KimiToolParser,
    MinimaxM2ToolParser,
    DeepseekV3ToolParser,
    DeepseekV3_1ToolParser,
    Gemma4ToolParser,
    GlmToolParser,
    InklingToolParser,
]


def _ids(cls: type[StructuralTagToolParser]) -> str:
    return cls.__name__


def _marker(parser: StructuralTagToolParser) -> str:
    """The start marker as a plain ``str``.

    A parser may declare its markers as a ``str`` subclass rather than a bare
    string, and a ``str``-subclass enum renders as its member name under
    f-string interpolation. Every string operation the parser itself performs
    is unaffected; only building fixture text by interpolation is, so flatten
    it here rather than special-casing the parsers that do this.
    """
    return str.__str__(parser._start_marker)


def _stream(parser: StructuralTagToolParser, text: str, chunk: int = 1) -> str:
    """Feeds ``text`` through ``parse_delta`` and returns the content emitted."""
    out: list[str] = []
    for i in range(0, len(text), chunk):
        for delta in parser.parse_delta(text[i : i + chunk]) or []:
            if delta.content:
                out.append(delta.content)
    return "".join(out)


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_releases_a_withheld_marker_prefix(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """A reply ending in the marker's first byte survives the stream ending."""
    parser = parser_cls()
    lead = _marker(parser)[0]
    text = f"answer{lead}"

    streamed = _stream(parser, text)
    # Without the flush this is where the character is lost.
    assert streamed == "answer"

    assert parser.flush() == lead
    # Streaming now agrees with the non-streaming parse, which returns the
    # whole reply as content because no marker ever completed.
    assert streamed + lead == parser_cls().parse_complete(text).content


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_returns_none_when_nothing_was_withheld(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """Ordinary text streams in full, so the flush has nothing to add."""
    parser = parser_cls()
    text = "a complete reply with nothing held back"

    assert _stream(parser, text) == text
    assert parser.flush() is None


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_never_emits_tool_call_structure_as_content(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """Once the marker completes, the rest is structure and must not leak."""
    parser = parser_cls()
    text = f"before{_marker(parser)}"

    assert _stream(parser, text) == "before"
    assert parser.flush() is None


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_does_not_repeat_itself(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """The cursor advances, so a second flush yields nothing to re-send."""
    parser = parser_cls()
    _stream(parser, f"answer{_marker(parser)[0]}")

    assert parser.flush()
    assert parser.flush() is None


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_releases_a_multi_character_marker_prefix(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """The withheld run is not limited to one byte.

    ``partial_tag_overlap`` matches as much of the marker as the tail allows,
    up to one character short of the whole thing, so a reply trailing off part
    way into a marker has that entire run withheld. All of it has to come
    back, not just the last character.
    """
    parser = parser_cls()
    marker = _marker(parser)
    # One short of the full marker: the longest run that is still a prefix.
    prefix = marker[:-1]
    text = f"answer{prefix}"

    streamed = _stream(parser, text)
    assert streamed == "answer"
    assert parser.flush() == prefix


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_releases_a_reply_that_is_only_a_marker_prefix(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """Nothing precedes the withheld run, so the whole reply is at stake."""
    parser = parser_cls()
    lead = _marker(parser)[0]

    assert _stream(parser, lead) == ""
    assert parser.flush() == lead


@pytest.mark.parametrize("parser_cls", STRUCTURAL_PARSERS, ids=_ids)
def test_flush_on_an_untouched_parser_returns_none(
    parser_cls: type[StructuralTagToolParser],
) -> None:
    """A stream that produced nothing has nothing to release."""
    assert parser_cls().flush() is None


def test_every_registered_structural_parser_is_covered() -> None:
    """``STRUCTURAL_PARSERS`` must not drift as parsers are added.

    The holdback comes from the base class, so a parser is affected exactly
    when it subclasses it. Anything registered under that rule needs a row
    above, or the sweep silently stops covering a model. Cascade's in-tree
    doubles are excluded as they ship no model.
    """
    registered = {
        cls
        for cls in _TOOL_PARSERS.values()
        if issubclass(cls, StructuralTagToolParser)
        and not cls.__module__.startswith("max.experimental")
    }

    assert registered <= set(STRUCTURAL_PARSERS), (
        "registered structural parsers missing from STRUCTURAL_PARSERS: "
        f"{sorted(c.__name__ for c in registered - set(STRUCTURAL_PARSERS))}"
    )
