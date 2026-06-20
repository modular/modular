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

"""Provides utility functions for computing allowed generation steps in pipeline variants."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import llguidance
import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.context import (
    GenerationStatus,
    LogProbabilities,
    StructuredOutputRegionDelimiters,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    get_parser_cls,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.types import RequestID
from max.profiler import Tracer, traced
from max.support.math import ceildiv
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from max.pipelines.modeling.types import PipelineTokenizer

logger = logging.getLogger("max.pipelines")


def _count_token_subsequence(
    content: Sequence[int], special_tags: Sequence[int]
) -> int:
    """Counts non-overlapping occurrences of ``special_tags`` in ``content``.

    Used only on the matcher-rejection diagnostic path to count how many
    tool-call section markers were already committed. ``special_tags`` is the
    section-begin (or -end) token-id sequence — a single token for most
    parsers, so this is effectively a count. Runs O(len(content)); acceptable
    because it fires only when a rejection has already occurred.
    """
    width = len(special_tags)
    if width == 0:
        return 0
    tag_ids = list(special_tags)
    count = 0
    i = 0
    last_start = len(content) - width
    while i <= last_start:
        if list(content[i : i + width]) == tag_ids:
            count += 1
            i += width
        else:
            i += 1
    return count


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

        # Build byte representation for each token (required by TokenizerWrapper).
        # convert_ids_to_tokens(i) returns the token's byte->unicode *surface
        # form* (e.g. raw newline 0x0A -> 'Ċ', space 0x20 -> 'Ġ'); .encode("utf-8")
        # then gives the UTF-8 bytes of those placeholder characters (b'\xc4\x8a'),
        # not the token's true bytes (b'\n'). Feeding those to llguidance makes it
        # mask against the wrong bytes and admit control-char tokens as legal JSON
        # string content, leaking raw newlines into structured output. Reverse the
        # map via the tokenizer's byte_decoder to recover the true bytes.
        # byte_decoder is integral to a byte-level BPE tokenizer (its own decode
        # depends on it).
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
                    # A char outside the byte->unicode map (rare; e.g. some
                    # special tokens, like an emoji): fall back to the UTF-8 encoding.
                    # This fallback is not expected to be used for standard TikToken vocabs.
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

        # TikToken tokenizers use allow_special_tokens (not add_special_tokens)
        return self._tokenizer.encode(text, allow_special_tokens=True)


def calculate_num_steps(
    context: TextGenerationContextType,
    num_steps: int,
    max_seq_len: int,
) -> int:
    """Compute the number of generation steps allowed for a context.

    The value is clamped by the remaining capacity with respect to
    the model's configured ``max_seq_len``.

    Args:
        context: The context whose sequence length constraints apply.
        num_steps: Desired number of steps to attempt.
        max_seq_len: The maximum allowed sequence length for the model.

    Returns:
        The number of steps to execute for this context (>= 1).

    Raises:
        ValueError: If the current request length is already >= ``max_seq_len``.
    """
    num_available_steps = context.compute_num_available_steps(max_seq_len)

    if num_available_steps <= 0:
        raise ValueError(
            f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
        )

    return min(num_available_steps, num_steps)


def build_response(
    context_batch: list[TextGenerationContextType],
    max_seq_len: int,
    max_growth_per_step: int = 1,
) -> dict[RequestID, TextGenerationOutput]:
    """Build response from updated contexts.

    Args:
        context_batch: The list of context objects.
        max_seq_len: The maximum sequence length.
        max_growth_per_step: Maximum tokens that can be added in the next step.
            For standard decoding this is 1. For speculative decoding this is
            num_speculative_tokens + 1 (all drafts accepted + bonus token).

    Returns:
        Dictionary mapping request IDs to TextGenerationOutput objects.
    """
    res: dict[RequestID, TextGenerationOutput] = {}

    for context in context_batch:
        context_max_length = upper_bounded_default(
            upper_bound=max_seq_len, default=context.max_length
        )

        # Mark as done if the next step would exceed the max length.
        current_length = context.tokens.processed_length + 1
        if current_length + max_growth_per_step > context_max_length:
            context.status = GenerationStatus.MAXIMUM_LENGTH

        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


@traced
def update_context_and_prepare_responses(
    generated_tokens_host: npt.NDArray[np.int32],
    flat_batch: list[TextGenerationContextType],
    batch_log_probabilities: list[list[LogProbabilities | None]] | None = None,
    enable_log_probs: bool = False,
    overwrite_future: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed
            as [batch, 1] (single step).
        flat_batch: List of generation contexts, one per request, matching
            batch dimension.
        batch_log_probabilities: List of per-step log probability outputs (or
            None), each entry is a list per batch for that step.
        enable_log_probs: Whether to include log probability data in outputs.
        overwrite_future: Whether to overwrite future tokens in the context.

    Returns:
        A dictionary mapping request IDs to their respective generation outputs.
    """
    res: dict[RequestID, TextGenerationOutput] = {}
    for batch_index, context in enumerate(flat_batch):
        # Convert to a Python scalar to improve serialization performance.
        next_token = int(generated_tokens_host[batch_index, 0])

        # Get Log probs if needed.
        log_probs: LogProbabilities | None = None
        if enable_log_probs:
            assert batch_log_probabilities is not None
            if batch_log_probabilities:
                log_probs_for_step = batch_log_probabilities[0]
                if log_probs_for_step and batch_index < len(log_probs_for_step):
                    log_probs = log_probs_for_step[batch_index]

        if overwrite_future:
            # If generated_length is still 0, then there is no placeholder
            # future token. This is possible due to chunked prefill or preemption.
            if context.tokens.generated_length:
                context.realize_future_token(
                    new_token=next_token, log_probabilities=log_probs
                )
        else:
            context.advance_token_buffer(
                new_token=next_token, log_probabilities=log_probs
            )
            context.advance_fsm(next_token)

        # Only add the output if there are tokens to return.
        # It is possible that there are no generated tokens due to chunked prefill.
        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


@traced
def update_spec_decode_context_and_prepare_responses(
    draft_tokens: npt.NDArray[np.int32],
    next_draft_tokens: npt.NDArray[np.int32],
    num_accepted_draft_tokens: npt.NDArray[np.int32],
    next_tokens: npt.NDArray[np.int32],
    context_batch: list[TextGenerationContextType],
    max_seq_len: int,
    think_start_token_id: int | None = None,
    think_end_token_id: int | None = None,
    skip_fsm_advance: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after speculative decoding.

    When both boundary ids are provided, also toggles
    ``ctx.in_reasoning_phase`` from the just-committed tokens, in commit
    order so a ``<think>...</think>`` pair within one accept set ends
    correctly.

    Args:
        draft_tokens: Draft tokens verified this batch.
        next_draft_tokens: Next batch's draft tokens.
        num_accepted_draft_tokens: Count of accepted draft tokens per request.
        next_tokens: Bonus tokens per request.
        context_batch: List of generation contexts.
        max_seq_len: Maximum sequence length.
        think_start_token_id: Token ID that starts a reasoning phase.
        think_end_token_id: Token ID that ends a reasoning phase.
        skip_fsm_advance: When True, skip FSM advancement because a CUDA host
            callback already advanced the FSM. Token buffer is still updated.
    """
    num_draft_tokens_to_verify = draft_tokens.shape[1]
    num_speculative_tokens = next_draft_tokens.shape[1]

    assert num_accepted_draft_tokens.shape == (len(context_batch),)
    assert next_tokens.shape == (len(context_batch),)
    assert next_draft_tokens.shape == (
        len(context_batch),
        num_speculative_tokens,
    )
    assert all(
        num_accept <= num_draft_tokens_to_verify
        for num_accept in num_accepted_draft_tokens
    )

    track_phase = (
        think_start_token_id is not None and think_end_token_id is not None
    )

    # Handle chunked prefill case where there are no future tokens.
    for batch_idx, ctx in enumerate(context_batch):
        if not ctx.tokens.generated_length:
            continue

        maybe_accepted_draft_tokens: list[int] = draft_tokens[
            batch_idx
        ].tolist()
        num_accept = num_accepted_draft_tokens[batch_idx]
        tokens = maybe_accepted_draft_tokens[:num_accept]
        tokens += [next_tokens[batch_idx]]
        for i, token in enumerate(tokens):
            # The overlap scheduler leaves a FUTURE_TOKEN placeholder as the last
            # generated token; realize_future_token overwrites it in place. Calling
            # update() for that same index would append a duplicate (see
            # update_context_and_prepare_responses with overwrite_future).
            if i == 0:
                ctx.realize_future_token(token)
                # For structured output, advance FSM with the realized token.
                # realize_future_token only updates the token buffer, not the FSM.
                # Skip when a CUDA host callback already advanced the FSM.
                if ctx.matcher is not None and not skip_fsm_advance:
                    ctx.advance_fsm(token)
            elif ctx.is_done:
                break
            else:
                if skip_fsm_advance and ctx.matcher is not None:
                    # Token buffer must still advance; FSM was already advanced
                    # by the CUDA host callback. Only skip ctx.update() when
                    # there is a matcher — unconstrained contexts must still
                    # call ctx.update() so EOS detection fires normally.
                    ctx.advance_token_buffer(token)
                else:
                    ctx.update(token)

        if track_phase:
            for token in tokens:
                if token == think_start_token_id:
                    ctx.in_reasoning_phase = True
                elif token == think_end_token_id:
                    ctx.in_reasoning_phase = False

        ctx.spec_decoding_state.maybe_accepted_draft_tokens = []
        if not ctx.is_done:
            # Save draft tokens for verification in the next TG step.
            # Skipped when is_done=True: the context produces no further TG
            # steps so draft tokens are unnecessary.
            ctx.spec_decoding_state.draft_tokens_to_verify = next_draft_tokens[
                batch_idx
            ].tolist()

    # With speculative decoding, the next step can add up to
    # num_speculative_tokens (all drafts accepted) + 1 (bonus token).
    max_growth_per_step = num_speculative_tokens + 1
    result = build_response(
        context_batch=context_batch,
        max_seq_len=max_seq_len,
        max_growth_per_step=max_growth_per_step,
    )

    # Clear draft tokens for contexts that won't be processed further.
    for ctx in context_batch:
        if ctx.is_done:
            ctx.spec_decoding_state.draft_tokens_to_verify = []

    return result


def get_rope_theta(config: AutoConfig) -> float:
    """Gets rope_theta from a HuggingFace config, compatible with transformers v4 and v5.

    Transformers v5 moved rope_theta into config.rope_parameters["rope_theta"].
    This function checks rope_parameters first, then falls back to config.rope_theta.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return rope_params["rope_theta"]

    return config.rope_theta


def get_eos_tokens(hf_config: AutoConfig, eos_token_id: int) -> set[int]:
    """Returns the set of end-of-sequence token IDs from config or fallback.

    Args:
        hf_config: HuggingFace model configuration.
        eos_token_id: Default EOS token id when not present in config.

    Returns:
        Set of EOS token ids to use for generation.
    """
    # Expand eos tokens if more are provided in pipeline_config
    if "eos_token_id" not in hf_config:
        return set([eos_token_id])

    hf_eos_tokens = hf_config.eos_token_id
    if isinstance(hf_eos_tokens, int):
        if hf_eos_tokens != eos_token_id:
            msg = f"eos_token_id provided in huggingface config ({hf_eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
            logger.warning(msg)
        return set([hf_eos_tokens])
    elif isinstance(hf_eos_tokens, list):
        if eos_token_id in hf_eos_tokens:
            return set(hf_eos_tokens)
        else:
            return set([eos_token_id])
    else:
        msg = f"eos_token_id in huggingface_config is neither int or list: {hf_eos_tokens}"
        logger.warning(msg)
        return set([eos_token_id])


@dataclass
class StructuredOutputHelper:
    """Helper for structured output (constrained decoding) in text generation pipelines.

    Encapsulates grammar compilation and bitmask management, consolidating
    shared logic between TextGenerationPipeline and OverlapTextGenerationPipeline.

    Constrained decoding is used when:
    1. Feature enabled via feature flag (--enable-structured-output)
    2. Tool calling enforced via grammar (tool_choice=required, named function, or auto (conditional enforcement))
    """

    enabled: bool = False
    """Whether constrained decoding is available (tokenizer info initialized)."""
    enable_response_format_schema: bool = False
    """Whether user-provided json_schema is allowed."""
    vocab_size: int | None = None
    """Vocabulary size from the tokenizer, or None if disabled."""
    _tokenizer_info: Any = field(default=None, repr=False)
    tool_call_region_delimiters: StructuredOutputRegionDelimiters | None = None
    """Token sequences for tool call boundaries (conditional enforcement)."""
    # Serialises access to per-context ``ctx.matcher`` between the async
    # FSM-advance host callback and the synchronous spec-decode bitmask
    # path; concurrent calls into llguidance's ``LLInterpreter`` trip a
    # ``RuntimeError: Already borrowed`` and kill the worker coroutine.
    _matcher_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    @staticmethod
    def _get_tool_region_tags(
        tool_parser_name: str | None,
    ) -> tuple[str | None, str | None]:
        """Extract tool-calling *region* start/end tags from a registered parser.

        These tags control when ``GrammarEnforcementState`` toggles
        ``grammar_enforced``.  The start tag tells the state machine
        when to begin constraining (for ``tool_choice=auto`` where
        enforcement doesn't start immediately).  The end tag, if
        present, tells it when to stop.

        For **section-wrapped** parsers (e.g. Kimi K2.5, DeepSeek V3)
        that define ``SECTION_BEGIN``/``SECTION_END``, the outer
        section pair is returned — enforcement spans all tool calls.

        For **flat** parsers (e.g. Gemma 4) that only define
        ``CALL_BEGIN``/``CALL_END``, only ``CALL_BEGIN`` is returned
        as the start trigger; the end is ``None``.  ``CALL_END`` is a
        per-call delimiter, not a region boundary — using it would
        prematurely disable enforcement between consecutive tool calls,
        causing the model to generate unconstrained tokens before the
        next call.  With no end tag the grammar itself governs what
        follows each ``CALL_END`` (another call, or a terminal like
        ``<|tool_response>``).

        Args:
            tool_parser_name: Name of the registered tool parser, or None.

        Returns:
            A (start, end) pair of region tags.
        """
        parser_cls = get_parser_cls(tool_parser_name)
        if parser_cls is None:
            return (None, None)

        if not (
            isinstance(parser_cls, type)
            and issubclass(parser_cls, StructuralTagToolParser)
        ):
            return (None, None)

        if parser_cls.SECTION_BEGIN and parser_cls.SECTION_END:
            return (parser_cls.SECTION_BEGIN, parser_cls.SECTION_END)
        if parser_cls.CALL_BEGIN:
            return (parser_cls.CALL_BEGIN, None)

        return (None, None)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        enable_structured_output: bool,
        tool_parser_name: str | None = None,
    ) -> StructuredOutputHelper:
        """Create a helper from a tokenizer.

        Args:
            tokenizer: A pipeline tokenizer with a HuggingFace delegate attribute.
            enable_structured_output: Whether structured output is enabled
                (e.g. to constrain to response format json_schema).
            tool_parser_name: Name of the registered tool parser. Used to extract
                structural tags for tool call start/end markers.

        Returns:
            A configured StructuredOutputHelper instance.

            Note: Constrained decoding is used when tool calling grammar is forced or enable_structured_output=True.
        """
        if not hasattr(tokenizer, "delegate"):
            return cls(enabled=False)
        tokenizer_delegate = tokenizer.delegate
        vocab_size = len(tokenizer_delegate)

        if isinstance(tokenizer_delegate, PreTrainedTokenizerFast):
            # Fast path for HuggingFace fast tokenizers
            tokenizer_info = llguidance.hf.from_tokenizer(
                tokenizer_delegate, n_vocab=vocab_size
            )
        else:
            # Fallback for TikTokenTokenizer, used by KimiK2_5
            # Use adapter -> TokenizerWrapper -> LLTokenizer chain
            adapter = _TikTokenAdapter(tokenizer_delegate)
            wrapper = TokenizerWrapper(adapter)
            tokenizer_info = LLTokenizer(wrapper, n_vocab=vocab_size)

        # Extract structural tags from tool parser if available
        tool_start, tool_end = cls._get_tool_region_tags(tool_parser_name)

        # Tokenize start/end tags to get token ID sequences
        tool_call_region_delimiters: StructuredOutputRegionDelimiters | None = (
            None
        )
        if tool_start is not None or tool_end is not None:
            start_token_ids: list[int] | None = None
            end_token_ids: list[int] | None = None
            if tool_start is not None:
                start_token_ids = tokenizer_delegate.encode(
                    tool_start, add_special_tokens=False
                )
            if tool_end is not None:
                end_token_ids = tokenizer_delegate.encode(
                    tool_end, add_special_tokens=False
                )
            tool_call_region_delimiters = StructuredOutputRegionDelimiters(
                start_token_ids=start_token_ids,
                end_token_ids=end_token_ids,
            )

        return cls(
            enabled=True,
            enable_response_format_schema=enable_structured_output,
            vocab_size=vocab_size,
            _tokenizer_info=tokenizer_info,
            tool_call_region_delimiters=tool_call_region_delimiters,
        )

    def update_context(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Update context and bitmask for structured output.

        If a grammar is present, it is used directly. Otherwise,
        if a json_schema is present and no matcher is set, this compiles a
        grammar matcher and installs it on the context, then fills the
        per-request token bitmask.

        tool-call grammars (with any ``tool_choice`` setting) work
        regardless of ``enable_response_format_schema`` — the
        ``--enable-structured-output`` flag only gates user-supplied JSON
        schemas (via ``response_format``). Requests carrying a user schema
        set ``requires_structured_output_flag=True`` and are rejected when
        the flag is off.

        Args:
            context: Request context to update.
            bitmask: Preallocated bitmask buffer; updated in-place.
            index: Position in the bitmask for this request.

        Raises:
            InputError: If a JSON schema is provided but structured output is
                not enabled, or if constrained decoding is not available.
        """
        # Check for grammar first (e.g., tool call grammars from tool_choice=required)
        if context.grammar and context.matcher is None:
            if not self.enabled:
                raise InputError(
                    "grammar provided but constrained decoding is not available."
                )

            # ``--enable-structured-output`` gates user-supplied schemas (passed
            # via ``response_format``), not tool grammars.
            # Pure tool-call grammars work regardless of the flag. The combined
            # tool+schema case sets ``requires_structured_output_flag=True``.
            if (
                context.requires_structured_output_flag
                and not self.enable_response_format_schema
            ):
                raise InputError(
                    "response_format with a JSON schema requires "
                    "--enable-structured-output. Drop response_format to use "
                    "tool-call constraints only, or pass the flag to allow "
                    "schema-constrained responses."
                )

            try:
                with Tracer("tool_grammar_compile"):
                    matcher = LLMatcher(self._tokenizer_info, context.grammar)
                context.set_matcher(matcher)
                self.set_context_tool_region(context)
            except Exception as e:
                raise InputError(
                    f"Grammar provided in request cannot be compiled. "
                    f"From llguidance: {e}"
                ) from e

        # Fall back to json_schema if no grammar
        # json_schema requires enable_response_format_schema (--enable-structured-output flag)
        elif context.json_schema and context.matcher is None:
            if not self.enable_response_format_schema:
                raise InputError(
                    "json_schema provided but structured output is not enabled. "
                    "Pass --enable-structured-output to enable this feature."
                )

            try:
                # Compact JSON (no structural whitespace) to match
                # the tool-call grammar convention.
                grammar = LLMatcher.grammar_from_json_schema(
                    context.json_schema,
                    overrides={"whitespace_pattern": ""},
                )
                matcher = LLMatcher(self._tokenizer_info, grammar)
                context.set_matcher(matcher)
            except Exception as e:
                raise InputError(
                    f"JSON schema provided in request cannot be compiled to "
                    f"valid grammar. Update your JSON schema to produce valid "
                    f"structured output. From llguidance: {e}"
                ) from e

        if context.matcher:
            # Fill the bitmask for this context.
            self.fill_bitmask(context, bitmask, index)

    def allocate_bitmask(
        self,
        batch_size: int,
    ) -> npt.NDArray[np.int32]:
        """Allocate a token bitmask for the given batch size.

        Args:
            batch_size: Number of requests in the batch.

        Returns:
            A bitmask array of shape [batch_size, ceil(vocab_size/32)].

        Raises:
            ValueError: If vocab_size is not set.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set to allocate bitmask")
        return llguidance.numpy.allocate_token_bitmask(
            batch_size, self.vocab_size
        )

    def fill_bitmask(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill the bitmask for a context's matcher.

        Only fills the bitmask when the context has a matcher AND
        grammar_enforced is True. For conditional enforcement
        (tool_choice=auto), the bitmask is left unconstrained until
        the tool call start token is detected.

        Args:
            context: Request context with a matcher.
            bitmask: Bitmask buffer to update in-place.
            index: Position in the bitmask for this request.
        """
        if context.matcher and context.grammar_enforced:
            llguidance.numpy.fill_next_token_bitmask(
                context.matcher, bitmask, index=index
            )

    def set_context_tool_region(
        self,
        context: TextGenerationContextType,
    ) -> None:
        """Set the tool_region on context's grammar state if conditional enforcement.

        Called after setting the matcher to configure conditional enforcement
        for tool_choice=auto scenarios.

        Args:
            context: Request context with grammar state.
        """
        if self.tool_call_region_delimiters is not None:
            context.set_tool_region(
                start_token_ids=self.tool_call_region_delimiters.start_token_ids,
                end_token_ids=self.tool_call_region_delimiters.end_token_ids,
            )

    def _tokens_for_consume(self, token: int, was_enforced: bool) -> list[int]:
        """Tokens to feed the matcher for one conditional-enforcement step.

        Mirrors ``TextGenerationContext._tokens_for_consume`` for the async
        spec-decode paths: on the enforcement flip-on (``was_enforced`` was
        False), feed the whole start marker rather than just the token that
        completed it, so multi-token / namespace-prefixed markers (e.g.
        MiniMax-M3's ``NS<tool_call>``) align with the grammar's start rule
        instead of rejecting into fail-open. Single-token markers have
        ``start_token_ids == [token]``, so this is a no-op for them.
        """
        delims = self.tool_call_region_delimiters
        if not was_enforced and delims and delims.start_token_ids:
            return list(delims.start_token_ids)
        return [token]

    def _speculatively_fill_bitmask_window(
        self,
        ctx: TextGenerationContextType,
        drafts: npt.NDArray[np.int64],
        bitmask_window: npt.NDArray[np.int32],
    ) -> None:
        """Advance enforcement state through drafts, filling per-slot bitmasks.

        A draft that flips enforcement on mid-window causes downstream
        slots to be constrained: e.g. a ``</think>`` draft exits the
        thinking region, so the slot immediately after it gets a filled
        bitmask instead of staying unconstrained. The matcher is walked
        on a deep copy (never mutated), and enforcement state is restored
        at the end, so committed-token processing on the next batch
        replays the same transitions from a clean state.

        Out-of-vocab drafts stop the speculative advance and leave any
        remaining slots unconstrained; they are not treated as errors.

        Args:
            ctx: The request context.
            drafts: ``[K]`` candidate draft tokens for the next batch.
            bitmask_window: ``[K+1, packed_vocab]``, pre-initialized to
                ``-1`` (unconstrained). Slot 0 is the position
                immediately after the committed tokens; slot ``i+1`` is
                the position after consuming ``drafts[i]``. Slots stay
                ``-1`` wherever grammar enforcement is off.
        """
        assert ctx.matcher is not None
        fsm_snap = ctx.snapshot_grammar_state()

        # Speculatively consume drafts on a throwaway copy of the matcher.
        # LLMatcher.rollback() is not a perfect inverse when the consumed
        # span crosses a grammar rule/repetition boundary — e.g.
        # ``<|tool_call_begin|>`` can cause issues for rollback. Bypass this
        # issue by taking a deep copy instead.
        matcher_copy = ctx.matcher.deep_copy()

        # Slot 0: state immediately after committed tokens.
        if ctx.grammar_enforced:
            llguidance.numpy.fill_next_token_bitmask(
                matcher_copy,
                bitmask_window[0, :].reshape(1, -1),
                index=0,
            )

        vocab_size = self.vocab_size or 0
        for i in range(drafts.shape[0]):
            draft_token = int(drafts[i])
            if draft_token < 0 or draft_token >= vocab_size:
                break

            # EOS-class tokens are not part of the grammar — they signal end of
            # generation. Skip the matcher so it stays in a clean terminal
            # state. ``restore_grammar_state`` undoes this transient flip.
            # Drafts past EOS are pointless (the request ended), so exit the
            # loop and leave remaining slots unconstrained.
            if draft_token in ctx.eos_tracker.eos_token_ids:
                ctx.grammar_enforced = False
                break

            consumed = False
            was_enforced = ctx.grammar_enforced
            if ctx.update_enforcement_state(draft_token):
                tokens = self._tokens_for_consume(draft_token, was_enforced)
                if matcher_copy.try_consume_tokens(tokens) == len(tokens):
                    consumed = True
                else:
                    break

            if consumed or ctx.grammar_enforced:
                llguidance.numpy.fill_next_token_bitmask(
                    matcher_copy,
                    bitmask_window[i + 1, :].reshape(1, -1),
                    index=0,
                )

        ctx.restore_grammar_state(fsm_snap)

    def _rejection_diagnostics(
        self,
        ctx: TextGenerationContextType,
        committed_tokens: list[int],
        committed_idx: int,
    ) -> str:
        """Best-effort extra state for the matcher-rejection error log.

        Runs only on the (rare) rejection path and is fully guarded so a
        diagnostic failure can never crash the async worker thread. Surfaces
        whether the rejection landed in the middle of a tool call (a desync
        signature) versus at a clean grammar boundary:

        * ``matcher_accepting=False`` means the matcher was mid-structure
          (inside a call header / args), not at a stoppable boundary.
        * ``open_sections>0`` means more ``<|tool_calls_section_begin|>`` than
          ``...section_end|>`` are committed, i.e. an open tool-call section.
        * ``committed_token_ids`` is this spec-decode step's accepted-drafts +
          bonus token, as raw token IDs, so the exact desyncing batch can be
          reconstructed offline against the tokenizer.

        Only token IDs are logged (no decoded text), so no model output text
        reaches the logs; reconstruct decoded forms after the fact.
        """
        try:
            matcher = ctx.matcher
            snapshot = ctx.snapshot_grammar_state()

            # "Inside an open tool-call section": section-begins minus
            # section-ends committed so far.
            delims = self.tool_call_region_delimiters
            open_sections = -1
            if (
                delims is not None
                and delims.start_token_ids
                and delims.end_token_ids
            ):
                generated = [int(t) for t in ctx.tokens.generated]
                open_sections = _count_token_subsequence(
                    generated, delims.start_token_ids
                ) - _count_token_subsequence(generated, delims.end_token_ids)

            return (
                f"reject_idx={committed_idx}/{len(committed_tokens)} "
                f"matcher_accepting="
                f"{matcher.is_accepting() if matcher is not None else '?'} "
                f"matcher_stopped="
                f"{matcher.is_stopped() if matcher is not None else '?'} "
                f"enforced={ctx.grammar_enforced} "
                f"tools_forced={ctx.tools_forced} "
                f"in_thinking_region={snapshot.in_thinking_region} "
                f"open_sections={open_sections} "
                f"committed_token_ids={list(committed_tokens)}"
            )
        except Exception as e:
            return f"<diagnostics unavailable: {e!r}>"

    @traced
    def advance_fsm_and_compute_bitmasks(
        self,
        context_batch: list[TextGenerationContextType],
        accepted_draft_tokens: npt.NDArray[np.int64],
        num_accepted: npt.NDArray[np.int64],
        bonus_tokens: npt.NDArray[np.int64],
        next_draft_tokens: npt.NDArray[np.int64],
        bitmask_out: npt.NDArray[np.int32],
    ) -> None:
        """Advance FSM through accepted tokens, then compute bitmasks for the next batch.

        Combines FSM advancement (Part 1) with bitmask computation (Part 2) for
        use in a CUDA host callback. Must NOT call any CUDA APIs.

        Part 1 permanently advances the FSM through committed tokens from the
        current batch (accepted draft tokens followed by the bonus token). This
        mirrors what sync_and_process_outputs would do for structured output.

        Part 2 speculatively advances through the next batch's draft tokens to
        compute bitmasks, then rolls back to restore the FSM to the state after
        Part 1.

        Args:
            context_batch: List of generation contexts.
            accepted_draft_tokens: Draft tokens verified this batch, shape [batch, K].
            num_accepted: Count of accepted draft tokens per request, shape [batch].
            bonus_tokens: Bonus (target) tokens per request, shape [batch].
            next_draft_tokens: Draft tokens for the next batch, shape [batch, K].
            bitmask_out: Packed int32 bitmask output, shape [batch, K+1, packed_vocab].
                Initialized to -1 (unconstrained) per context before filling.
        """
        # This method runs on an AsyncRT worker thread. The main thread
        # may try to access the same ``ctx.matcher`` via
        # ``compute_speculative_bitmasks`` for the next iter while this
        # callback is still in flight; without serialisation llguidance
        # raises ``RuntimeError: Already borrowed`` and the worker dies.
        # See the comment on ``_matcher_lock``.
        with self._matcher_lock:
            for ctx_idx, ctx in enumerate(context_batch):
                bitmask_out[ctx_idx, :, :] = -1

                if ctx.matcher is None:
                    continue

                # Part 1: Advance the enforcement state machine through
                # committed tokens, one at a time so special tokens (e.g.
                # tool-call structural tags) can flip grammar enforcement
                # mid-sequence. This mirrors the synchronous
                # ``advance_fsm`` in ``context.py`` exactly:
                #
                #   * EOS-class tokens are not part of the grammar — they
                #     signal end of generation. Skip the matcher so it
                #     stays in a clean terminal state rather than getting
                #     a spurious rejection.
                #   * For everything else, gate on
                #     ``update_enforcement_state``'s return value, not on
                #     ``grammar_enforced``. The return value distinguishes
                #     ``</think>`` (flip enforcement on, do NOT consume —
                #     the thinking delimiter isn't grammar content) from
                #     ``<|tool_calls_section_end|>`` (flip enforcement
                #     off, DO consume — it's the grammar's terminal).
                #   * If the matcher rejects, log and disable enforcement
                #     for the rest of the request — continuing against a
                #     desynced matcher produces schema-shaped nonsense.
                n_accepted = int(num_accepted[ctx_idx])
                bonus_token = int(bonus_tokens[ctx_idx])
                committed_tokens = [
                    int(accepted_draft_tokens[ctx_idx, j])
                    for j in range(n_accepted)
                ]
                committed_tokens.append(bonus_token)

                for committed_idx, token in enumerate(committed_tokens):
                    if token in ctx.eos_tracker.eos_token_ids:
                        ctx.grammar_enforced = False
                        continue
                    was_enforced = ctx.grammar_enforced
                    if not ctx.update_enforcement_state(token):
                        continue
                    # On the enforcement flip-on, feed the matcher the whole
                    # start marker (multi-token / NS-prefixed markers like
                    # M3's NS<tool_call>), not just the completing token.
                    tokens = self._tokens_for_consume(token, was_enforced)
                    if ctx.matcher.try_consume_tokens(tokens) == len(tokens):
                        continue
                    # ``role`` distinguishes a rejection on the bonus
                    # token (sampled by target *with* bitmask, so a
                    # rejection here usually means a bitmask/matcher
                    # desync) from a rejection on an accepted draft
                    # (produced by the draft model and verified by
                    # target, where rejection more often reflects the
                    # target sampling outside the matcher's allowed
                    # set on a draft slot the speculative walk did
                    # not constrain).
                    role = (
                        "bonus"
                        if committed_idx == len(committed_tokens) - 1
                        else f"accepted_draft[{committed_idx}]"
                    )
                    logger.error(
                        "Async matcher rejected %d token(s) ending at %d "
                        "(request %s, role=%s); disabling enforcement "
                        "for the rest of the request. "
                        "matcher_errors=%s matcher_warnings=%s %s",
                        len(tokens),
                        token,
                        ctx.request_id,
                        role,
                        ctx.matcher.get_error(),
                        ctx.matcher.get_grammar_warnings(),
                        self._rejection_diagnostics(
                            ctx, committed_tokens, committed_idx
                        ),
                    )
                    ctx.grammar_enforced = False

                # Part 2: speculative window for the next batch's bitmasks.
                # A draft that flips enforcement on mid-window causes
                # downstream slots to be constrained.
                self._speculatively_fill_bitmask_window(
                    ctx,
                    drafts=next_draft_tokens[ctx_idx],
                    bitmask_window=bitmask_out[ctx_idx],
                )

    @traced
    def compute_speculative_bitmasks(
        self,
        context_batch: list[TextGenerationContextType],
        draft_tokens: npt.NDArray[np.int64],
        num_positions: int,
    ) -> npt.NDArray[np.int32]:
        """Compute speculative bitmasks for structured output in spec decode.

        For each draft position i, the bitmask at position i contains valid
        tokens given the FSM state after consuming draft[0:i-1]. The last
        position (num_positions - 1) is for the bonus token.

        This method speculatively advances the FSM through draft tokens to
        compute bitmasks, then rolls back to restore the original state.

        The bitmask is returned packed (1 bit per token, 32 tokens per int32
        word); the GPU acceptance sampler unpacks and applies it in one fused
        pass, so this method never unpacks to bool.

        Args:
            context_batch: List of generation contexts.
            draft_tokens: Draft tokens to verify, shape [batch, K].
            num_positions: Number of bitmask positions (K + 1, including bonus).

        Returns:
            Packed int32 bitmask array of shape
            ``[batch_size, num_positions, ceil(vocab_size / 32)]``. ``-1`` (all
            bits set) means all tokens are valid.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set for speculative bitmasks")

        batch_size = len(context_batch)
        packed_vocab_size = ceildiv(self.vocab_size, 32)

        # Check if any context has structured output
        has_structured_output = any(
            ctx.json_schema is not None
            or ctx.matcher is not None
            or ctx.grammar is not None
            for ctx in context_batch
        )

        if not has_structured_output:
            # Fast path: all unconstrained, return all-valid packed bitmask
            # (-1 = all bits set).
            return np.full(
                (batch_size, num_positions, packed_vocab_size),
                -1,
                dtype=np.int32,
            )

        # Allocate packed bitmask (int32) for llguidance
        packed_bitmask = llguidance.numpy.allocate_token_bitmask(
            batch_size * num_positions, self.vocab_size
        )
        packed_vocab_size = packed_bitmask.shape[1]
        packed_bitmask = packed_bitmask.reshape(
            batch_size, num_positions, packed_vocab_size
        )

        # Serialise against the async FSM-advance host callback
        # (``advance_fsm_and_compute_bitmasks``). Both paths touch the
        # same ``ctx.matcher`` LLInterpreter; concurrent access trips
        # llguidance's "Already borrowed" Rust panic and kills the
        # worker. See the comment on ``_matcher_lock``.
        with self._matcher_lock:
            # Initialize matchers for contexts with json_schema or grammar
            for ctx in context_batch:
                needs_matcher = ctx.matcher is None and (
                    ctx.json_schema or ctx.grammar is not None
                )
                if needs_matcher:
                    self.update_context(
                        ctx,
                        packed_bitmask[0, 0, :].reshape(
                            1, -1
                        ),  # Dummy, will be overwritten
                        index=0,
                    )

            # Fill bitmasks for each context. ``packed_bitmask`` is
            # initialized to -1 (all bits set = all tokens valid), so the
            # helper only needs to write slots where the FSM is enforced.
            for ctx_idx, ctx in enumerate(context_batch):
                if not ctx.matcher:
                    continue
                self._speculatively_fill_bitmask_window(
                    ctx,
                    drafts=draft_tokens[ctx_idx],
                    bitmask_window=packed_bitmask[ctx_idx],
                )
        # Return the packed int32 bitmask directly; the GPU acceptance sampler
        # unpacks and applies it in a single fused pass.
        return packed_bitmask
