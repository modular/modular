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
"""Cost and exactness of every token mask along one tool call, per parser.

The decode loop fills a token bitmask once per generated token, so a single
slow position stalls every step that reaches it. Each case walks one
representative call through its parser's compiled xgrammar tool grammar and
checks, at every position, that the fill stays under the budget and that the
filled mask equals token-by-token acceptance.

CI uses synthetic RAW vocabularies. To also run each case on its real
tokenizer, point ``TOOL_GRAMMAR_FILL_HF_CACHE`` at a HuggingFace hub cache and
run the test on this host, where the cache is::

    ./bazelw test //max/tests/integration/pipelines/tokenization:test_tool_grammar_fill_cost \
        --strategy=TestRunner=local \
        --test_env=TOOL_GRAMMAR_FILL_HF_CACHE=$HOME/.cache/huggingface/hub

A case whose tokenizer is not in that cache skips.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
import transformers
from _grammar_fill_harness import (
    BUDGET_S,
    EOS,
    assert_mask_is_exact,
    encode,
    fill_costs_along,
    synthetic_vocab,
)
from _grammar_harness import MARKERS
from max import _xgrammar as xgr
from max.pipelines.architectures.gemma4.tool_parser import Gemma4ToolParser
from max.pipelines.architectures.glm5_1.tool_parser import GlmToolParser
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.architectures.minimax_m2.tool_parser import (
    MinimaxM2ToolParser,
)
from max.pipelines.architectures.muse_glimmer.tool_parser import (
    MuseGlimmerToolParser,
)
from max.pipelines.architectures.qwen3_5.tool_parser import Qwen3_5ToolParser
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    XgrammarMatcher,
    make_grammar_backend,
)

_COMMAND = "cd /tmp && ls -la | grep -v total"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }
]


@dataclass(frozen=True)
class _Case:
    parser: str
    """The registered tool parser name."""
    generate_grammar: Callable[..., str]
    """The parser's ``generate_tool_call_grammar``."""
    call: str
    """One call to ``bash`` exactly as the model emits it."""
    markers: tuple[str, ...]
    """Strings the real vocabulary holds as single tokens."""
    special: tuple[str, ...] = ()
    """The markers the serving backend registers as special tokens."""
    hf_repo: str | None = None
    """The HuggingFace repo of the real tokenizer, if one is public."""


_GEMMA4_CHANNEL = ("<|channel>", "<channel|>")
_GEMMA4_MARKERS = ("<|tool_call>", "<tool_call|>", '<|"|>', *_GEMMA4_CHANNEL)
_INKLING_MARKERS = tuple(MARKERS)
_GLM_ARG_MARKERS = ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")
_THINK = ("<think>", "</think>")

_CASES = [
    _Case(
        parser="minimax_m2",
        generate_grammar=MinimaxM2ToolParser.generate_tool_call_grammar,
        # The grammar opens with the empty think block MAX emits when
        # reasoning is off.
        call=(
            "\n</think>\n\n\n\n<minimax:tool_call>\n"
            '<invoke name="bash">\n'
            f'<parameter name="command">{_COMMAND}</parameter>\n'
            '<parameter name="timeout">30</parameter>\n'
            "</invoke>\n</minimax:tool_call>"
        ),
        markers=("<minimax:tool_call>", "</minimax:tool_call>", *_THINK),
        hf_repo="MiniMaxAI/MiniMax-M2.7",
    ),
    _Case(
        parser="kimik2_5",
        generate_grammar=KimiToolParser.generate_tool_call_grammar,
        call=(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0"
            f'<|tool_call_argument_begin|>{{"command": "{_COMMAND}", '
            '"timeout": 30}<|tool_call_end|><|tool_calls_section_end|>'
        ),
        markers=(
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            *_THINK,
        ),
        hf_repo="moonshotai/Kimi-K2.5",
    ),
    _Case(
        parser="glm45",
        generate_grammar=GlmToolParser.generate_tool_call_grammar,
        call=(
            "<tool_call>bash<arg_key>command</arg_key>"
            f"<arg_value>{_COMMAND}</arg_value><arg_key>timeout</arg_key>"
            "<arg_value>30</arg_value></tool_call>"
        ),
        markers=("<tool_call>", "</tool_call>", *_GLM_ARG_MARKERS, *_THINK),
        special=_GLM_ARG_MARKERS,
        hf_repo="zai-org/GLM-5.1",
    ),
    _Case(
        parser="qwen3_5",
        generate_grammar=Qwen3_5ToolParser.generate_tool_call_grammar,
        call=(
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            f"{_COMMAND}\n</parameter>\n<parameter=timeout>\n30\n"
            "</parameter>\n</function>\n</tool_call>"
        ),
        markers=("<tool_call>", "</tool_call>", *_THINK),
        hf_repo="Qwen/Qwen3.5-9B",
    ),
    _Case(
        parser="gemma4",
        generate_grammar=Gemma4ToolParser.generate_tool_call_grammar,
        call=(
            f'<|tool_call>call:bash{{command:<|"|>{_COMMAND}<|"|>,'
            "timeout:30}<tool_call|>"
        ),
        markers=_GEMMA4_MARKERS,
        special=_GEMMA4_MARKERS,
        hf_repo="google/gemma-4-31B-it",
    ),
    _Case(
        parser="inkling",
        generate_grammar=InklingToolParser.generate_tool_call_grammar,
        # The generation prompt ends at <|message_model|>, so the first call
        # starts at its bare name.
        call=(
            "bash<|content_invoke_tool_json|>"
            f'{{"name":"bash","args":{{"command":"{_COMMAND}","timeout":30}}}}'
            "<|end_message|>"
        ),
        markers=_INKLING_MARKERS,
        special=_INKLING_MARKERS,
    ),
    _Case(
        parser="muse_glimmer",
        generate_grammar=MuseGlimmerToolParser.generate_tool_call_grammar,
        # Under auto the matcher sees no message header: <|start|> and
        # <|message|> are special, so free text before the section rejects
        # them.
        call=(
            '<atem:function_calls>\n<atem:invoke name="bash">\n'
            f'<atem:parameter name="command">{_COMMAND}</atem:parameter>\n'
            '<atem:parameter name="timeout">30</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        ),
        markers=(
            "<|start|>",
            "<|message|>",
            "<atem:function_calls>",
            "</atem:function_calls>",
        ),
        special=("<|start|>", "<|message|>"),
    ),
]


@dataclass
class _Walk:
    """A matcher at the start of a call and the call's token ids."""

    matcher: xgr.GrammarMatcher
    vocab_size: int
    tokens: list[int]

    def positions(self) -> Iterator[int]:
        """Yields each position, the matcher left there, then accepts on."""
        for i, token in enumerate(self.tokens):
            yield i
            assert self.matcher.accept_token(token), f"rejected at {i}"
        yield len(self.tokens)


def _grammar(case: _Case) -> str:
    return case.generate_grammar(tools=_TOOLS, tool_choice="auto")


# Large enough that a fill re-matching most of the vocabulary costs several
# milliseconds, as it does on a real 150k-260k vocabulary.
_SYNTHETIC_VOCAB_SIZE = 60_000


@functools.cache
def _synthetic_compiled(
    case: _Case,
) -> tuple[xgr.CompiledGrammar, dict[str, int]]:
    vocab = synthetic_vocab(case.markers, _SYNTHETIC_VOCAB_SIZE)
    ids = {token: i for i, token in enumerate(vocab)}
    info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[ids[EOS]],
        special_token_ids=[ids[m] for m in case.special],
    )
    return xgr.GrammarCompiler(info).compile_structural_tag(_grammar(case)), ids


def _synthetic_walk(case: _Case) -> _Walk:
    compiled, ids = _synthetic_compiled(case)
    markers = sorted(case.markers, key=len, reverse=True)
    return _Walk(
        xgr.GrammarMatcher(compiled),
        len(ids),
        encode(case.call, ids, markers),
    )


def _real_walk(case: _Case) -> _Walk:
    cache = os.environ.get("TOOL_GRAMMAR_FILL_HF_CACHE")
    if not cache:
        pytest.skip("set TOOL_GRAMMAR_FILL_HF_CACHE to a hub cache to run")
    if case.hf_repo is None:
        pytest.skip(f"{case.parser} has no public tokenizer")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            case.hf_repo,
            cache_dir=cache,
            local_files_only=True,
            trust_remote_code=True,
        )
    except OSError as e:
        pytest.skip(f"{case.hf_repo} is not loadable from {cache}: {e}")
    # The serving backend, so the vocabulary and special tokens match decode.
    backend = make_grammar_backend(
        "xgrammar", tokenizer, len(tokenizer), tool_parser_name=case.parser
    )
    matcher = backend.create_matcher(_grammar(case))
    assert isinstance(matcher, XgrammarMatcher)
    return _Walk(
        matcher._matcher,
        len(tokenizer),
        tokenizer.encode(case.call, add_special_tokens=False),
    )


def _fill_costs(walk: _Walk) -> list[float]:
    return fill_costs_along(walk.matcher, walk.vocab_size, walk.tokens)


def _assert_under_budget(costs: list[float]) -> None:
    worst = max(range(len(costs)), key=costs.__getitem__)
    assert costs[worst] < BUDGET_S, (
        f"fill at position {worst} took {costs[worst] * 1e3:.3f} ms"
    )


def _assert_masks_exact(walk: _Walk) -> None:
    for i in walk.positions():
        assert_mask_is_exact(walk.matcher, walk.vocab_size, f"at {i}")


_IDS = [case.parser for case in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_synthetic_fill_stays_under_budget(case: _Case) -> None:
    _assert_under_budget(_fill_costs(_synthetic_walk(case)))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_synthetic_mask_matches_token_by_token_acceptance(case: _Case) -> None:
    _assert_masks_exact(_synthetic_walk(case))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_real_tokenizer_fill_stays_under_budget(case: _Case) -> None:
    _assert_under_budget(_fill_costs(_real_walk(case)))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_real_tokenizer_mask_matches_token_by_token_acceptance(
    case: _Case,
) -> None:
    _assert_masks_exact(_real_walk(case))
