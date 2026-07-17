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
"""Tests for pipeline_variants/utils.py."""

from __future__ import annotations

from typing import Any

import max.pipelines.lib.pipeline_variants.structured_output_backend as _sob
import numpy as np
import numpy.typing as npt
import pytest
from max import _xgrammar as xgrammar
from max.pipelines.context import (
    GenerationStatus,
    GrammarMatcher,
    StructuredOutputRegionDelimiters,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    GrammarBackend,
)
from max.pipelines.lib.pipeline_variants.utils import (
    StructuredOutputHelper,
    build_response,
)
from max.pipelines.lib.tool_parsing import StructuralTagToolParser, register
from max.pipelines.modeling.types import ParsedToolCall, RequestID
from max.pipelines.sampling import DEFAULT_STRUCTURED_OUTPUT_BACKEND


class _RecordingMatcher(GrammarMatcher):
    """Minimal GrammarMatcher that records the tokens fed to it."""

    def __init__(self) -> None:
        self.consumed: list[list[int]] = []

    def try_consume_tokens(self, tokens: list[int]) -> int:
        self.consumed.append(list(tokens))
        return len(tokens)

    def is_accepting(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    def get_error(self) -> str | None:
        return None

    def get_grammar_warnings(self) -> Any:
        return None

    def deep_copy(self) -> _RecordingMatcher:
        # Speculative walks (Part 2) use a copy, never the original.
        return _RecordingMatcher()


class _NoopBackend(GrammarBackend[Any]):
    """GrammarBackend stub so Part 2's fills don't touch llguidance."""

    name = "noop"

    def compile_json_schema(self, json_schema: str) -> Any:
        return None

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        return _RecordingMatcher()

    def validate_grammar(self, grammar: Any) -> None:
        return None

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        return np.zeros((batch_size, (vocab_size + 31) // 32), dtype=np.int32)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        pass


def create_text_context(prompt_len: int, max_length: int) -> TextContext:
    """Create a TextContext for testing."""
    tokens = np.arange(prompt_len, dtype=np.int64)
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(tokens),
    )


def advance_to_processed(ctx: TextContext) -> None:
    """Advance context so prompt tokens are marked as processed.

    After this call, processed_length equals the original token count.
    """
    ctx.update_with_future_token()
    ctx.realize_future_token(new_token=99, log_probabilities=None)


@register("_test_section_parser")
class _SectionParser(StructuralTagToolParser):
    """Minimal section-wrapped parser for region-tag derivation tests."""

    SECTION_BEGIN = "<sec>"
    SECTION_END = "</sec>"

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        return []


@register("_test_to_eos_parser")
class _ToEosParser(_SectionParser):
    """Section parser that opts into enforcement-to-EOS."""

    ENFORCE_TOOL_REGION_TO_EOS = True


class TestGetToolRegionTags:
    """``StructuredOutputHelper._get_tool_region_tags``."""

    def test_section_parser_returns_section_pair(self) -> None:
        tags = StructuredOutputHelper._get_tool_region_tags(
            "_test_section_parser"
        )
        assert tags == ("<sec>", "</sec>")

    def test_to_eos_parser_has_no_end_tag(self) -> None:
        """With no end tag, enforcement never flips off: the completed
        grammar masks everything but EOS, so the turn ends with its single
        tool-call section (e.g. MiniMax-M3; CENG-718)."""
        tags = StructuredOutputHelper._get_tool_region_tags(
            "_test_to_eos_parser"
        )
        assert tags == ("<sec>", None)


class TestBuildResponse:
    """Tests for build_response function."""

    def test_marks_maximum_length_when_at_limit(self) -> None:
        """Context is marked MAXIMUM_LENGTH when at the boundary."""
        max_seq_len = 100
        # Create context with 99 tokens
        ctx = create_text_context(prompt_len=99, max_length=max_seq_len)

        # Advance to processed state: processed_length = 99
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 99

        # current_length = 99 + 1 = 100
        # With max_growth_per_step=1: 100 + 1 = 101 > 100 → MAXIMUM_LENGTH
        build_response([ctx], max_seq_len=max_seq_len, max_growth_per_step=1)

        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH

    def test_does_not_mark_when_below_limit(self) -> None:
        """Context is not marked when there's room for growth."""
        max_seq_len = 100
        ctx = create_text_context(prompt_len=50, max_length=max_seq_len)

        # Advance to processed state: processed_length = 50
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 50

        # current_length = 50 + 1 = 51
        # With max_growth_per_step=1: 51 + 1 = 52 <= 100 → not done
        build_response([ctx], max_seq_len=max_seq_len, max_growth_per_step=1)

        assert ctx.status != GenerationStatus.MAXIMUM_LENGTH

    def test_max_growth_per_step_for_speculative_decoding(self) -> None:
        """Larger max_growth_per_step triggers earlier termination.

        This is the core logic for speculative decoding: with 3 spec tokens,
        max_growth_per_step = 4, so we stop earlier to prevent KV cache overflow.
        """
        max_seq_len = 100
        max_growth = 4  # e.g., 3 speculative tokens + 1 bonus

        # At length 96, after advance: processed_length = 96
        # current_length = 96 + 1 = 97
        # With max_growth_per_step=4: 97 + 4 = 101 > 100 → MAXIMUM_LENGTH
        ctx_near_limit = create_text_context(
            prompt_len=96, max_length=max_seq_len
        )
        advance_to_processed(ctx_near_limit)
        assert ctx_near_limit.tokens.processed_length == 96

        build_response(
            [ctx_near_limit],
            max_seq_len=max_seq_len,
            max_growth_per_step=max_growth,
        )
        assert ctx_near_limit.status == GenerationStatus.MAXIMUM_LENGTH

        # With default max_growth_per_step=1: 97 + 1 = 98 <= 100 → not done
        ctx_with_default = create_text_context(
            prompt_len=96, max_length=max_seq_len
        )
        advance_to_processed(ctx_with_default)

        build_response(
            [ctx_with_default], max_seq_len=max_seq_len, max_growth_per_step=1
        )
        assert ctx_with_default.status != GenerationStatus.MAXIMUM_LENGTH

    def test_respects_per_request_max_length(self) -> None:
        """Per-request max_length is respected when lower than global."""
        global_max_seq_len = 100
        per_request_max = 50

        # Create context with per-request limit of 50
        ctx = create_text_context(prompt_len=49, max_length=per_request_max)
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 49

        # current_length = 49 + 1 = 50
        # With max_growth_per_step=1: 50 + 1 = 51 > 50 → MAXIMUM_LENGTH
        build_response(
            [ctx], max_seq_len=global_max_seq_len, max_growth_per_step=1
        )
        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH


class TestTokensForConsume:
    """``StructuredOutputHelper._tokens_for_consume``.

    On the conditional-enforcement flip-on (``tool_choice=auto``), the fresh
    matcher must consume the whole tool-call start marker, not just the token
    that completed it — otherwise multi-token / namespace-prefixed markers
    (e.g. MiniMax-M3's ``NS<tool_call>``) reject and enforcement falls open.
    """

    @staticmethod
    def _helper(start_token_ids: list[int]) -> StructuredOutputHelper:
        return StructuredOutputHelper(
            tool_call_region_delimiters=StructuredOutputRegionDelimiters(
                start_token_ids=start_token_ids,
                end_token_ids=[999],
            )
        )

    def test_flip_on_feeds_full_multitoken_marker(self) -> None:
        # M3-style NS<tool_call> = two tokens; flip-on feeds both.
        helper = self._helper([200058, 200052])
        assert helper._tokens_for_consume(200052, was_enforced=False) == [
            200058,
            200052,
        ]

    def test_already_enforced_feeds_single_token(self) -> None:
        helper = self._helper([200058, 200052])
        assert helper._tokens_for_consume(77, was_enforced=True) == [77]

    def test_single_token_marker_is_noop(self) -> None:
        # Single-token markers (e.g. Kimi) feed just the token even on flip-on.
        helper = self._helper([42])
        assert helper._tokens_for_consume(42, was_enforced=False) == [42]

    def test_no_delimiters_feeds_single_token(self) -> None:
        helper = StructuredOutputHelper()
        assert helper._tokens_for_consume(5, was_enforced=False) == [5]


class TestAdvanceFsmAndComputeBitmasks:
    """``StructuredOutputHelper.advance_fsm_and_compute_bitmasks``.

    A row that can't be attributed to a producing-batch slot (absent, or reset
    to an initial prompt after enqueue) must be degraded to the all-valid -1
    reset rather than raising -- a raise blanket-unconstrains the whole batch.
    """

    @staticmethod
    def _decoding_ctx() -> TextContext:
        """A continuing (non-initial-prompt) unconstrained decode row."""
        ctx = create_text_context(prompt_len=4, max_length=128)
        # Mirror handle_prefill_response on the decode engine: applying the
        # first generated token makes generated_length > 0 and clears
        # is_initial_prompt -- exactly the state of a KV-transferred row.
        ctx.update(new_token=99)
        assert ctx.tokens.generated_length > 0
        assert not ctx.is_initial_prompt
        return ctx

    @staticmethod
    def _empty_bitmask(num_rows: int, num_positions: int) -> np.ndarray:
        # Packed int32 bitmask, [rows, K+1, ceil(vocab/32)]. Vocab is small;
        # the callback resets each row to -1 before any fill, so the only
        # requirement is a writable rectangle of the right outer shape.
        return np.zeros((num_rows, num_positions, 1), dtype=np.int32)

    def test_row_absent_from_producing_batch_degrades_not_raises(self) -> None:
        """A row absent from the producing batch is degraded, not raised on."""
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        producer = self._decoding_ctx()
        transferred = self._decoding_ctx()

        producing_batch = [producer]
        consuming_batch = [transferred, producer]

        # Producing-batch-shaped spec-decode arrays: K=1 draft token.
        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        bonus = np.full((1,), 99, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        # Sentinel: every reached row is reset to -1, so a row still holding 7
        # afterwards was never reached (the loop aborted early).
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # No raise; the degraded row and the trailing row are both reached.
        assert (bitmask_out == -1).all()

    def test_row_preempted_in_flight_degrades_not_raises(self) -> None:
        """Regression: a row reset (preempted) after enqueue is degraded, not
        raised on, so the rest of the batch keeps its constraints."""
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        survivor = self._decoding_ctx()
        preempted = self._decoding_ctx()

        producing_batch = [preempted, survivor]
        consuming_batch = [preempted, survivor]

        # Mirror an in-flight preemption: the scheduler resets the context
        # (requeuing it to context encoding) after the callback was enqueued.
        preempted.reset()
        assert preempted.is_initial_prompt

        accepted = np.zeros((2, 1), dtype=np.int64)
        num_accepted = np.zeros((2,), dtype=np.int64)
        bonus = np.full((2,), 99, dtype=np.int64)
        next_draft = np.zeros((2, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # No raise; the preempted row and the survivor row are both reached.
        assert (bitmask_out == -1).all()

    def test_preempted_in_flight_row_matcher_not_advanced(self) -> None:
        """Regression: a constrained row reset (preempted) after enqueue must
        NOT have its matcher advanced in Part 1.

        ``reset()`` rewinds the token buffer (dropping the in-flight committed
        token) but preserves ``ctx.matcher``. Advancing the matcher through that
        dropped token would leave it one token ahead of the sequence the row
        re-primes from on resume -- a silent desync that yields schema-shaped
        but invalid output. The matcher must be left untouched.
        """
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_NoopBackend()
        )

        matcher = _RecordingMatcher()
        preempted = self._decoding_ctx()
        preempted.set_matcher(matcher)
        preempted.grammar_enforced = True

        producing_batch = [preempted]
        consuming_batch = [preempted]

        # Mirror an in-flight preemption after the callback was enqueued.
        preempted.reset()
        assert preempted.is_initial_prompt
        assert preempted.matcher is matcher  # reset preserves the matcher

        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        # A non-EOS committed (bonus) token that, absent the skip, the matcher
        # would be advanced through.
        bonus = np.full((1,), 5, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)
        bitmask_out[:] = 7

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # The preempted row's matcher was never advanced (Part 1 skipped it),
        # and its bitmask row was left unconstrained (Part 2 skipped it).
        assert matcher.consumed == []
        assert (bitmask_out == -1).all()

    def test_continuing_row_matcher_is_advanced(self) -> None:
        """Control: a constrained row NOT preempted still has its matcher
        advanced through the committed token -- the skip is preemption-only."""
        helper = StructuredOutputHelper(
            enabled=True, vocab_size=16, backend=_NoopBackend()
        )

        matcher = _RecordingMatcher()
        ctx = self._decoding_ctx()
        ctx.set_matcher(matcher)
        ctx.grammar_enforced = True
        assert not ctx.is_initial_prompt

        producing_batch = [ctx]
        consuming_batch = [ctx]

        accepted = np.zeros((1, 1), dtype=np.int64)
        num_accepted = np.zeros((1,), dtype=np.int64)
        bonus = np.full((1,), 5, dtype=np.int64)
        next_draft = np.zeros((1, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # Part 1 advanced the original matcher through the bonus token.
        assert matcher.consumed == [[5]]

    def test_all_consumer_rows_present_in_producing_batch_ok(self) -> None:
        """Control: steady decode->decode, every consumer row attributable.

        When no row was admitted from outside the producing batch (the
        aggregated steady-state path), the callback attributes every consumer
        row and does not assert.
        """
        helper = StructuredOutputHelper(enabled=True, vocab_size=16)

        row_a = self._decoding_ctx()
        row_b = self._decoding_ctx()

        producing_batch = [row_a, row_b]
        consuming_batch = [row_a, row_b]

        accepted = np.zeros((2, 1), dtype=np.int64)
        num_accepted = np.zeros((2,), dtype=np.int64)
        bonus = np.full((2,), 99, dtype=np.int64)
        next_draft = np.zeros((2, 1), dtype=np.int64)
        bitmask_out = self._empty_bitmask(len(consuming_batch), num_positions=2)

        helper.advance_fsm_and_compute_bitmasks(
            context_batch=producing_batch,
            accepted_draft_tokens=accepted,
            num_accepted=num_accepted,
            bonus_tokens=bonus,
            next_draft_tokens=next_draft,
            bitmask_out=bitmask_out,
            output_context_batch=consuming_batch,
        )

        # Unconstrained rows: callback resets every row to all-valid (-1).
        assert (bitmask_out == -1).all()


class _RaisingBackend(GrammarBackend[Any]):
    """GrammarBackend stub whose compiles always raise, to exercise the
    validator's exception translation."""

    name = "boom"

    def compile_json_schema(self, json_schema: Any) -> Any:
        raise ValueError("cannot compile schema")

    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        raise ValueError("cannot compile grammar")

    def validate_grammar(self, grammar: Any) -> None:
        return None

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        raise NotImplementedError

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        raise NotImplementedError


class TestGrammarValidation:
    """A backend's GrammarValidator checks turn a compile failure into an
    InputError (400)."""

    def test_tool_grammar_ok_does_not_raise(self) -> None:
        _NoopBackend().check_tool_grammar("<grammar>")

    def test_tool_grammar_uncompilable_raises_input_error(self) -> None:
        with pytest.raises(InputError, match="boom"):
            _RaisingBackend().check_tool_grammar("<grammar>")

    def test_json_schema_ok_does_not_raise(self) -> None:
        _NoopBackend().check_json_schema('{"type": "object"}')

    def test_json_schema_uncompilable_raises_input_error(self) -> None:
        with pytest.raises(InputError, match="boom"):
            _RaisingBackend().check_json_schema('{"type": "object"}')

    def test_make_validator_none_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A None backend_name (unresolved config) builds the validator with
        the default backend -- mirroring StructuredOutputHelper.from_tokenizer
        -- so admission still fires when the worker would otherwise silently
        fall back to xgrammar on an unresolved config and crash."""
        captured: dict[str, Any] = {}

        def fake_make(
            name: Any, delegate: Any, vocab_size: Any
        ) -> GrammarBackend[Any]:
            captured["name"] = name
            return _NoopBackend()

        monkeypatch.setattr(_sob, "make_grammar_backend", fake_make)
        _sob.make_grammar_validator(None, object(), 128)
        assert captured["name"] == DEFAULT_STRUCTURED_OUTPUT_BACKEND


class TestXgrammarCacheBound:
    """xgrammar's compiled-grammar cache is bounded (vLLM's VLLM_XGRAMMAR_CACHE_MB
    analog) so a long-running server can't grow it without limit."""

    def test_default_limit_is_512_mb(self) -> None:
        assert _sob._xgrammar_cache_limit_bytes() == 512 * 1024 * 1024

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODULAR_XGRAMMAR_CACHE_MB", "64")
        assert _sob._xgrammar_cache_limit_bytes() == 64 * 1024 * 1024

    def test_compiler_honors_byte_limit(self) -> None:
        vocab = [f"tok{i}".encode() for i in range(16)]
        tokenizer_info = xgrammar.TokenizerInfo(
            vocab, vocab_type=xgrammar.VocabType.RAW, stop_token_ids=[0]
        )
        limit = 8 * 1024 * 1024
        compiler = xgrammar.GrammarCompiler(
            tokenizer_info, max_memory_bytes=limit
        )
        assert compiler._impl.cache_limit_bytes == limit
