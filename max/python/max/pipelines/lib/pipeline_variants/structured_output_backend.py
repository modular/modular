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
"""Pluggable grammar backend for constrained decoding.

Abstracts the grammar/matcher engine behind two protocols so MAX can support
more than one structured-output backend:

* :class:`GrammarBackend` owns grammar compilation, matcher construction, and
  bitmask allocation/filling (the engine-level, non-per-token entry points).
* :class:`GrammarMatcher` is the per-request object stepped each decode step.
  Method names mirror llguidance's ``LLMatcher`` so the hot decode path is
  backend-agnostic by duck typing.

``llguidance`` is the original (and default) backend; its native ``LLMatcher``
already satisfies :class:`GrammarMatcher`, so :class:`LlguidanceBackend` is a
thin pass-through with no behavior change.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import llguidance
import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max import _xgrammar as xgrammar
from max.pipelines.context import GrammarMatcher
from max.pipelines.context.exceptions import InputError
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


class _TikTokenAdapter:
    """Adapter to make TikToken-based tokenizers compatible with llguidance.

    llguidance's TokenizerWrapper expects a tokenizer object with specific
    attributes (eos_token_id, bos_token_id, tokens, special_token_ids) and
    a callable interface for encoding. This adapter wraps TikToken-based
    tokenizers (which don't inherit from PreTrainedTokenizerFast) to provide
    that interface.

    Raises:
        ValueError: If the tokenizer is not a TikToken-based tokenizer.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if "TikToken" not in type(tokenizer).__name__:
            raise ValueError(
                f"Structured output requires PreTrainedTokenizerFast or "
                f"TikToken-based tokenizers, but got {type(tokenizer).__name__}"
            )

        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.special_token_ids = getattr(tokenizer, "all_special_ids", [])

        # convert_ids_to_tokens returns the byte->unicode surface form, not the
        # token's true bytes; reverse it via the tokenizer's byte_decoder, or
        # llguidance masks the wrong bytes and leaks control chars into output.
        byte_decoder = getattr(tokenizer, "byte_decoder", None)
        if byte_decoder is None:
            raise ValueError(
                "TikToken-based structured output requires a tokenizer with a "
                "`byte_decoder` (byte-level BPE inverse map); "
                f"{type(tokenizer).__name__} does not provide one."
            )
        vocab_size = len(tokenizer.get_vocab())
        self._tokens: list[bytes] = []
        for i in range(vocab_size):
            token_str = tokenizer.convert_ids_to_tokens(i)
            if token_str is None:
                self._tokens.append(b"")
            else:
                try:
                    self._tokens.append(
                        bytes(byte_decoder[c] for c in token_str)
                    )
                except KeyError:
                    self._tokens.append(
                        token_str.encode("utf-8", errors="replace")
                    )

    @property
    def tokens(self) -> list[bytes]:
        """Returns byte representation of each token in vocabulary."""
        return self._tokens

    def __call__(self, text: str | bytes) -> list[int]:
        """Encode text to token IDs."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        return self._tokenizer.encode(text, allow_special_tokens=True)


class GrammarBackend(Protocol):
    """Engine-level entry points: compile grammars, build matchers, bitmasks."""

    name: str

    def compile_json_schema(self, json_schema: str) -> Any:
        """Compile a JSON schema to a grammar handle for this backend."""
        ...

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        """Build a matcher from a compiled grammar (backend-specific handle)."""
        ...

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        ...

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens."""
        ...


class LlguidanceBackend:
    """llguidance backend. Thin pass-through over the native ``LLMatcher``."""

    name = "llguidance"

    def __init__(self, tokenizer_info: Any) -> None:
        self._tokenizer_info = tokenizer_info

    @classmethod
    def from_tokenizer_delegate(
        cls,
        tokenizer_delegate: PreTrainedTokenizerBase,
        vocab_size: int,
    ) -> LlguidanceBackend:
        """Build the llguidance tokenizer info from a tokenizer delegate."""
        if isinstance(tokenizer_delegate, PreTrainedTokenizerFast):
            tokenizer_info = llguidance.hf.from_tokenizer(
                tokenizer_delegate, n_vocab=vocab_size
            )
        else:
            adapter = _TikTokenAdapter(tokenizer_delegate)
            wrapper = TokenizerWrapper(adapter)
            tokenizer_info = LLTokenizer(wrapper, n_vocab=vocab_size)
        return cls(tokenizer_info)

    def compile_json_schema(self, json_schema: str) -> Any:
        """Compile a JSON schema to a grammar handle for this backend."""
        return LLMatcher.grammar_from_json_schema(
            json_schema, overrides={"whitespace_pattern": ""}
        )

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        """Build a matcher from a compiled grammar (backend-specific handle)."""
        return LLMatcher(self._tokenizer_info, grammar)

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        return llguidance.numpy.allocate_token_bitmask(batch_size, vocab_size)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens."""
        assert isinstance(matcher, LLMatcher)
        llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, index=index)


class XgrammarMatcher:
    """Adapter exposing an xgrammar ``GrammarMatcher`` as a :class:`GrammarMatcher`."""

    def __init__(self, matcher: Any) -> None:
        self._matcher = matcher

    def try_consume_tokens(self, tokens: list[int]) -> int:
        """Advance the matcher; returns the number of tokens consumed."""
        consumed = 0
        for token in tokens:
            if not self._matcher.accept_token(token):
                break
            consumed += 1
        return consumed

    def is_accepting(self) -> bool:
        """Whether the matcher is at an accepting (stoppable) state."""
        return bool(self._matcher.is_completed())

    def is_stopped(self) -> bool:
        """Whether the matcher has reached a terminal state."""
        return bool(self._matcher.is_terminated())

    def get_error(self) -> str | None:
        """Error message for the last rejection, if any (diagnostics)."""
        return None

    def get_grammar_warnings(self) -> Any:
        """Grammar compilation warnings, if any (diagnostics)."""
        return None

    def deep_copy(self) -> XgrammarMatcher:
        """Independent copy for speculative walks (never mutates the original)."""
        return XgrammarMatcher(self._matcher.fork())


class XgrammarBackend:
    """xgrammar backend.

    Compiles JSON schemas with full ``$ref``/``$defs``/``anyOf``/type-list
    enforcement (where llguidance fails open). The packed int32 bitmask layout
    matches llguidance's, and ``fill_next_token_bitmask`` writes numpy arrays
    directly, so the decode hot path stays torch-free.
    """

    name = "xgrammar"

    def __init__(self, compiler: Any) -> None:
        self._compiler = compiler

    @classmethod
    def from_tokenizer_delegate(
        cls,
        tokenizer_delegate: PreTrainedTokenizerBase,
        vocab_size: int,
    ) -> XgrammarBackend:
        """Build the xgrammar tokenizer info and compiler from a delegate."""
        if isinstance(tokenizer_delegate, PreTrainedTokenizerFast):
            tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
                tokenizer_delegate, vocab_size=vocab_size
            )
        else:
            adapter = _TikTokenAdapter(tokenizer_delegate)
            stop_token_ids = (
                [adapter.eos_token_id]
                if adapter.eos_token_id is not None
                else None
            )
            tokenizer_info = xgrammar.TokenizerInfo(
                adapter.tokens,
                vocab_type=xgrammar.VocabType.RAW,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
            )
        return cls(xgrammar.GrammarCompiler(tokenizer_info))

    def compile_json_schema(self, json_schema: Any) -> Any:
        """Compile a JSON schema (str or dict) to a grammar handle."""
        schema = (
            json_schema
            if isinstance(json_schema, str)
            else json.dumps(json_schema)
        )
        # any_whitespace=True allows flexible whitespace (including none); it
        # enforces the schema structure without blocking valid JSON. NOTE:
        # any_whitespace=False is NOT "compact" — it mandates a space after
        # ':'/',' and would reject compact output. This matches vLLM's default.
        return self._compiler.compile_json_schema(schema, any_whitespace=True)

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        """Build a matcher from a compiled grammar or structural-tag JSON."""
        if isinstance(grammar, xgrammar.CompiledGrammar):
            compiled = grammar
        elif isinstance(grammar, str):
            tag = xgrammar.StructuralTag.model_validate_json(grammar)
            compiled = self._compiler.compile_structural_tag(tag)
        else:
            raise InputError(
                f"The xgrammar backend received an unsupported grammar of "
                f"type {type(grammar).__name__}."
            )
        return XgrammarMatcher(xgrammar.GrammarMatcher(compiled))

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        # -1 == all bits set == unconstrained (matches llguidance + MAX's
        # bitmask convention); fill_next_token_bitmask overwrites filled rows.
        words = (vocab_size + 31) // 32
        return np.full((batch_size, words), -1, dtype=np.int32)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens."""
        assert isinstance(matcher, XgrammarMatcher)
        matcher._matcher.fill_next_token_bitmask(bitmask, index)


def build_xgrammar_tool_grammar(
    model_format: str,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
    reasoning: bool = False,
) -> str:
    """Build a serialized xgrammar tool-call grammar (StructuralTag JSON).

    Uses xgrammar's built-in per-model tool-call format (e.g. ``"kimi"``),
    which frames the model's tool-call envelope and constrains each call's
    arguments to that tool's JSON schema. The returned JSON string is passed
    as a grammar to :meth:`XgrammarBackend.create_matcher`.

    Args:
        model_format: xgrammar model-format key (e.g. ``"kimi"``).
        tools: OpenAI-style tool dicts (``{"type": "function", "function": ...}``).
        tool_choice: ``"auto"``, ``"required"``, or a named choice.
        reasoning: Whether the model interleaves reasoning before tool calls.

    Returns:
        The StructuralTag serialized as a JSON string.
    """
    tag = xgrammar.get_builtin_structural_tag(
        model_format,
        tools=tools,
        tool_choice=tool_choice,
        reasoning=reasoning,
    )
    return tag.model_dump_json()


def make_grammar_backend(
    name: str,
    tokenizer_delegate: PreTrainedTokenizerBase,
    vocab_size: int,
) -> GrammarBackend:
    """Construct the structured-output backend selected by ``name``.

    Args:
        name: Backend identifier (``"llguidance"`` or ``"xgrammar"``).
        tokenizer_delegate: HuggingFace/TikToken tokenizer to build vocab info.
        vocab_size: Vocabulary size from the tokenizer.

    Returns:
        A configured :class:`GrammarBackend`.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    if name == "llguidance":
        return LlguidanceBackend.from_tokenizer_delegate(
            tokenizer_delegate, vocab_size
        )
    if name == "xgrammar":
        return XgrammarBackend.from_tokenizer_delegate(
            tokenizer_delegate, vocab_size
        )
    raise ValueError(
        f"unknown structured output backend: {name!r} "
        f"(supported: 'llguidance', 'xgrammar')"
    )
