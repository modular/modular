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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import llguidance
import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.interfaces import (
    GenerationStatus,
    LogProbabilities,
    RequestID,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.core.exceptions import InputError
from max.pipelines.lib.utils import upper_bounded_default
from max.profiler import traced
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from max.interfaces import PipelineTokenizer

logger = logging.getLogger("max.pipelines")


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

        # Build byte representation for each token (required by TokenizerWrapper)
        vocab_size = len(tokenizer.get_vocab())
        self._tokens: list[bytes] = []
        for i in range(vocab_size):
            token_str = tokenizer.convert_ids_to_tokens(i)
            if token_str is None:
                self._tokens.append(b"")
            else:
                self._tokens.append(token_str.encode("utf-8", errors="replace"))

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
    context_batch: list[TextGenerationContextType], max_seq_len: int
) -> dict[RequestID, TextGenerationOutput]:
    """Build response from updated contexts.

    Args:
        context_batch: The list of context objects
        max_seq_len: The maximum sequence length

    Returns:
        Dictionary mapping request IDs to TextGenerationOutput objects
    """
    res: dict[RequestID, TextGenerationOutput] = {}

    for context in context_batch:
        # Identify the Max Length
        context_max_length = upper_bounded_default(
            upper_bound=max_seq_len, default=context.max_length
        )

        # Break early if beyond max length
        current_length = context.tokens.processed_length + 1
        if current_length >= context_max_length:
            context.status = GenerationStatus.MAXIMUM_LENGTH

        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


@traced
def update_context_and_prepare_responses(
    generated_tokens_host: npt.NDArray[np.int32],
    flat_batch: list[TextGenerationContextType],
    num_steps: int,
    batch_log_probabilities: list[list[LogProbabilities | None]] | None = None,
    enable_log_probs: bool = False,
    overwrite_future: bool = False,
    fsm_already_advanced_steps: int = 0,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed
            as [batch, step].
        flat_batch: List of generation contexts, one per request, matching
            batch dimension.
        num_steps: Number of generation steps to process for each context.
        batch_log_probabilities: List of per-step log probability outputs (or
            None), each entry is a list per batch for that step.
        enable_log_probs: Whether to include log probability data in outputs.
        overwrite_future: Whether to overwrite future tokens in the context.
        fsm_already_advanced_steps: Number of steps for which the FSM was
            already advanced during multi-step execution. For these steps,
            only the token buffer is updated (FSM is skipped).

    Returns:
        A dictionary mapping request IDs to their respective generation outputs.
    """
    res: dict[RequestID, TextGenerationOutput] = {}
    for batch_index, context in enumerate(flat_batch):
        for step in range(num_steps):
            # Convert to a Python scalar to improve serialization performance.
            next_token = int(generated_tokens_host[batch_index, step])

            # Get Log probs if needed.
            log_probs: LogProbabilities | None = None
            if enable_log_probs:
                assert batch_log_probabilities is not None
                if step < len(batch_log_probabilities):
                    log_probs_for_step = batch_log_probabilities[step]
                    if log_probs_for_step and batch_index < len(
                        log_probs_for_step
                    ):
                        log_probs = log_probs_for_step[batch_index]

            if overwrite_future:
                # If generated_length is still 0, then there is no placeholder
                # future token. This is possible due to chunked prefill or preemption.
                if context.tokens.generated_length:
                    context.realize_future_token(
                        new_token=next_token, log_probabilities=log_probs
                    )
            else:
                # Update token buffer for all steps
                context.advance_token_buffer(
                    new_token=next_token, log_probabilities=log_probs
                )
                # Only advance FSM for steps that weren't already advanced
                # during multi-step execution
                if step >= fsm_already_advanced_steps:
                    context.advance_fsm(next_token)

            if context.is_done:
                break

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
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after speculative decoding.

    When both boundary ids are provided, also toggles
    ``ctx.in_reasoning_phase`` from the just-committed tokens, in commit
    order so a ``<think>...</think>`` pair within one accept set ends
    correctly.
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
                if ctx.matcher is not None:
                    ctx.advance_fsm(token)
            elif ctx.is_done:
                break
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

    return build_response(
        context_batch=context_batch,
        max_seq_len=max_seq_len,
    )


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

    Attributes:
        enabled: Whether structured output is enabled.
        vocab_size: Vocabulary size from the tokenizer, or None if disabled.
    """

    enabled: bool = False
    vocab_size: int | None = None
    _tokenizer_info: Any = field(default=None, repr=False)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        enable_structured_output: bool,
    ) -> StructuredOutputHelper:
        """Create a helper from a tokenizer.

        Args:
            tokenizer: A pipeline tokenizer with a HuggingFace delegate attribute.
            enable_structured_output: Whether structured output is enabled.

        Returns:
            A configured StructuredOutputHelper instance.
        """
        if not enable_structured_output:
            return cls(enabled=False)

        assert hasattr(tokenizer, "delegate")
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

        return cls(
            enabled=True,
            vocab_size=vocab_size,
            _tokenizer_info=tokenizer_info,
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

        Args:
            context: Request context to update.
            bitmask: Preallocated bitmask buffer; updated in-place.
            index: Position in the bitmask for this request.

        Raises:
            ValueError: If a JSON schema or grammar is provided but structured
                output is not enabled.
        """
        # Check for grammar first (e.g., Kimi tool call regex)
        if context.grammar and context.matcher is None:
            if not self.enabled:
                raise ValueError(
                    "grammar provided but structured output is not enabled."
                )

            try:
                matcher = LLMatcher(self._tokenizer_info, context.grammar)
                context.set_matcher(matcher)
            except Exception as e:
                raise InputError(
                    f"Grammar provided in request cannot be compiled. "
                    f"From llguidance: {e}"
                ) from e

        # Fall back to json_schema if no grammar
        elif context.json_schema and context.matcher is None:
            if not self.enabled:
                raise ValueError(
                    "json_schema provided but structured output is not enabled."
                )

            try:
                grammar = LLMatcher.grammar_from_json_schema(
                    context.json_schema,
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

        Args:
            context: Request context with a matcher.
            bitmask: Bitmask buffer to update in-place.
            index: Position in the bitmask for this request.
        """
        if context.matcher:
            llguidance.numpy.fill_next_token_bitmask(
                context.matcher, bitmask, index=index
            )

    @traced
    def compute_speculative_bitmasks(
        self,
        context_batch: list[TextGenerationContextType],
        draft_tokens: npt.NDArray[np.int64],
        num_positions: int,
    ) -> npt.NDArray[np.bool_]:
        """Compute speculative bitmasks for structured output in spec decode.

        For each draft position i, the bitmask at position i contains valid
        tokens given the FSM state after consuming draft[0:i-1]. The last
        position (num_positions - 1) is for the bonus token.

        This method speculatively advances the FSM through draft tokens to
        compute bitmasks, then rolls back to restore the original state.

        Args:
            context_batch: List of generation contexts.
            draft_tokens: Draft tokens to verify, shape [batch, K].
            num_positions: Number of bitmask positions (K + 1, including bonus).

        Returns:
            Boolean bitmask array of shape [batch_size, num_positions, vocab_size].
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set for speculative bitmasks")

        batch_size = len(context_batch)
        num_draft_tokens_to_verify = (
            draft_tokens.shape[1] if draft_tokens.size else 0
        )

        # Check if any context has structured output
        has_structured_output = any(
            ctx.json_schema is not None
            or ctx.matcher is not None
            or ctx.grammar is not None
            for ctx in context_batch
        )

        if not has_structured_output:
            # Fast path: all unconstrained, return all-True bitmask
            return np.ones(
                (batch_size, num_positions, self.vocab_size), dtype=np.bool_
            )

        # Allocate packed bitmask (int32) for llguidance
        packed_bitmask = llguidance.numpy.allocate_token_bitmask(
            batch_size * num_positions, self.vocab_size
        )
        packed_vocab_size = packed_bitmask.shape[1]
        packed_bitmask = packed_bitmask.reshape(
            batch_size, num_positions, packed_vocab_size
        )

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

        # Fill bitmasks for each context and each draft position.
        # The packed_bitmask is initialized to -1 (all bits set = all tokens valid),
        # so we only need to call fill_next_token_bitmask for constrained contexts.
        for ctx_idx, ctx in enumerate(context_batch):
            if not ctx.matcher:
                continue  # Unconstrained: leave at initial -1 (all tokens valid)

            # Position 0: valid tokens at current FSM state
            llguidance.numpy.fill_next_token_bitmask(
                ctx.matcher,
                packed_bitmask[ctx_idx, 0, :].reshape(1, -1),
                index=0,
            )

            # Positions 1..num_positions-1: advance FSM through draft tokens
            # Position num_positions-1 is for the bonus token (after all drafts)
            tokens_consumed = 0
            for i in range(num_draft_tokens_to_verify):
                draft_token = int(draft_tokens[ctx_idx, i])
                # Skip invalid/placeholder tokens (negative or out of vocab range)
                # These positions stay at -1 (unconstrained)
                if draft_token < 0 or draft_token >= self.vocab_size:
                    raise ValueError(
                        f"Invalid draft token not in vocabulary: {draft_token}."
                    )
                # Check if the FSM accepts this token. If not, don't advance
                # and leave position at -1 (unconstrained).
                if ctx.matcher.try_consume_tokens([draft_token]) == 1:
                    tokens_consumed += 1
                    llguidance.numpy.fill_next_token_bitmask(
                        ctx.matcher,
                        packed_bitmask[ctx_idx, i + 1, :].reshape(1, -1),
                        index=0,
                    )
                else:
                    # FSM rejected this token. Don't try consuming any of
                    # the remaining draft tokens. Leave remaining
                    # positions at -1 (unconstrained).
                    break

            # Rollback FSM to original state
            if tokens_consumed > 0:
                ctx.matcher.rollback(tokens_consumed)
        # Unpack packed int32 bitmask to bool using vectorized bitwise ops.
        bits = 2 ** np.arange(32, dtype=np.int32)
        # Shape: [batch, num_positions, packed_vocab, 32]
        unpacked = (packed_bitmask[..., np.newaxis] & bits) != 0
        # Reshape to [batch, num_positions, packed_vocab * 32] and slice
        unpacked = unpacked.reshape(batch_size, num_positions, -1)
        return unpacked[:, :, : self.vocab_size].astype(np.bool_)
