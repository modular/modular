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

"""Context objects and protocols for pipeline inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import llguidance
import numpy as np
import numpy.typing as npt
from max.pipelines.request import RequestID
from max.pipelines.request.open_responses import OutputImageContent

from .eos_tracking import EOSTracker
from .log_probabilities import LogProbabilities
from .outputs import GenerationOutput, TextGenerationOutput
from .sampling_params import BaseContext, SamplingParams
from .status import GenerationStatus
from .tokens import ImageMetadata, TokenBuffer

_CHUNK_SIZE = 128
FUTURE_TOKEN = -999

_logger = logging.getLogger("max.pipelines")


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


@dataclass
class TextGenerationResponseFormat:
    """Represents the response format specification for a text generation request."""

    type: str
    """The type of response format, for example, ``json_object`` or ``grammar``."""

    json_schema: dict[str, Any] = field(default_factory=dict)
    """A JSON schema dictionary that defines the structure and validation rules for the generated response."""

    grammar: str | None = None
    """Grammar for constrained decoding.

    When set with ``type="grammar"``, this takes precedence over ``json_schema``.
    Used for model-specific constrained decoding formats like Kimi's tool call grammar.
    """

    grammar_enforced: bool = False
    """Whether to actively enforce grammar via bitmask.

    When True from the start, enforce grammar from the first token.
    When False initially (for tool_choice=auto without response_format), the
    grammar is compiled but not enforced until a tool call start token is
    detected.
    """

    tools_forced: bool = False
    """Whether tool calling was forced (tool_choice=required or named function).

    Controls whether ``grammar_enforced`` is ``True`` from the first generated
    token. Independent of the ``--enable-structured-output`` flag (which only
    gates user-supplied schemas; see ``requires_structured_output_flag``).
    """

    requires_structured_output_flag: bool = False
    """Whether this request requires ``--enable-structured-output`` to be set.

    True when the constraint includes a user-supplied JSON schema (from
    ``response_format``). False for pure tool-call grammars derived from
    the model's tool parser, which work without the operator flag because
    the grammar is server-controlled, not user-controlled.
    """

    has_json_schema: bool = False
    """Whether this request includes a JSON schema response format."""


# ---------------------------------------------------------------------------
# Snapshot / state helpers
# ---------------------------------------------------------------------------


@dataclass
class GrammarEnforcementSnapshot:
    """Captured grammar-enforcement state for rollback.

    The speculative bitmask path walks the enforcement state through
    draft tokens to compute downstream slot constraints and then
    restores this snapshot so committed-token processing on the next
    batch replays the same transitions from a clean state. Lives next
    to :class:`TextGenerationContext` because the protocol exposes it
    via :meth:`TextGenerationContext.snapshot_grammar_state` /
    :meth:`TextGenerationContext.restore_grammar_state`; the concrete
    implementation in :class:`TextContext` constructs and consumes instances.
    """

    in_thinking_region: bool
    grammar_enforced: bool
    tool_calling_match_buffer: list[int]
    thinking_match_buffer: list[int]


@dataclass
class SpecDecodingState:
    """Per-request state for speculative decoding."""

    draft_tokens_to_verify: list[int] = field(default_factory=list)
    """The draft tokens to verify in the next batch"""

    maybe_accepted_draft_tokens: list[int] = field(default_factory=list)
    """The draft tokens that are being verified in the current batch

    We are unsure whether these tokens will be accepted or not. However, to ensure
    that we allocate enough KV, we conservatively assume that they will all be
    accepted.

    This should only be present when running with overlap scheduler."""


# ---------------------------------------------------------------------------
# TextGenerationContext protocol and TypeVar
# ---------------------------------------------------------------------------


@runtime_checkable
class TextGenerationContext(BaseContext, Protocol):
    """Protocol defining the interface for text generation contexts in token generation.

    A ``TextGenerationContext`` represents model inputs for text generation pipelines,
    managing the state of tokens throughout the generation process. It handles token
    arrays, generation status, sampling parameters, and various indices that track
    different stages of token processing.
    """

    @property
    def tokens(self) -> TokenBuffer:
        """The token buffer for the context."""
        ...

    @property
    def eos_tracker(self) -> EOSTracker:
        """Holds EOS-related settings for this sequence and performs EOS/stop checks.

        Returns:
            The ``EOSTracker`` for this sequence.
        """
        ...

    @property
    def max_length(self) -> int | None:
        """The maximum allowed length for this sequence.

        When set, generation will stop when this length is reached, regardless
        of other stopping criteria.

        Returns:
            The maximum sequence length limit, or ``None`` if no limit is set.
        """
        ...

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt.

        This method is used when a request is evicted, meaning that the context
        needed to be re-encoded in the following CE iteration.
        """
        ...

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Computes the maximum number of generation steps available.

        Args:
            max_seq_len: The maximum allowed sequence length for this context.

        Returns:
            The number of generation steps that can be executed before reaching
            the sequence length limit.
        """
        ...

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens that must be generated.

        Returns:
            The minimum number of new tokens to generate.
        """
        ...

    @property
    def log_probabilities(self) -> int:
        """The number of top tokens to return log probabilities for.

        Returns:
            The number of top tokens to include in log probability output.
            Returns 0 if log probabilities are disabled.
        """
        ...

    @property
    def log_probabilities_echo(self) -> bool:
        """Whether to include input tokens in the returned log probabilities.

        Returns:
            ``True`` if input tokens should be included in log probability output,
            ``False`` otherwise.
        """
        ...

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns the token indices that should be masked in the output logits.

        Args:
            num_steps: The number of generation steps to compute masks for.

        Returns:
            A list of NumPy arrays, where each array contains token indices
            that should be masked for the corresponding generation step.
        """
        ...

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance the token buffer without touching FSM state.

        Args:
            new_token: The token to append to the buffer.
            log_probabilities: Optional log probabilities for this token.
        """
        ...

    def advance_fsm(self, token: int) -> bool:
        """Advance the FSM matcher state by one token.

        Args:
            token: The token to consume in the FSM.

        Returns:
            True if the token was accepted by the matcher, False if no
            matcher is present.
        """
        ...

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance both token buffer and FSM state.

        Args:
            new_token: The token ID to add to the generation sequence.
            log_probabilities: Optional log probability data for the new token
                and alternatives.
        """
        ...

    def update_with_future_token(self) -> None:
        """Append a placeholder future token to the generated tokens.

        This is primarily used for overlap scheduling.
        """
        ...

    def realize_future_token(
        self, new_token: int, log_probabilities: LogProbabilities | None = None
    ) -> None:
        """Overwrite the placeholder future token with the actual token.

        This is primarily used for overlap scheduling.
        """
        ...

    @property
    def matcher(self) -> Any | None:
        """The grammar matcher for structured output generation, if configured.

        Returns:
            The grammar matcher instance, or ``None`` if no structured generation
            is configured for this context.
        """
        ...

    @property
    def json_schema(self) -> str | None:
        """The JSON schema for constrained decoding, if configured.

        Returns:
            The JSON schema string, or ``None`` if no schema constraint is active.
        """
        ...

    @property
    def grammar(self) -> str | None:
        """Grammar for constrained decoding, if configured."""
        return None

    def set_matcher(self, matcher: Any) -> None:
        """Set a grammar matcher for constrained decoding.

        Args:
            matcher: The grammar matcher instance to use for constraining output.
        """
        ...

    @property
    def sampling_params(self) -> SamplingParams:
        """The sampling parameters configured for this generation request.

        Returns:
            The :class:`SamplingParams` instance containing all sampling
            configuration for this context.
        """
        ...

    @property
    def is_initial_prompt(self) -> bool:
        """Whether this context contains only the initial prompt.

        Returns:
            ``True`` if no tokens have been generated yet, ``False`` if
            generation has begun.
        """
        ...

    def to_generation_output(self) -> TextGenerationOutput:
        """Converts this context to a :class:`TextGenerationOutput` object.

        Returns:
            The output object containing the results of the text generation.
        """
        ...

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        """Returns the speculative decoding state."""
        ...

    cached_prefix_length: int | None
    """Prompt tokens served from the KV prefix cache on first admission."""

    in_reasoning_phase: bool
    """Whether the latest committed tokens are inside a ``<think>...</think>``
    block."""

    grammar_enforced: bool
    """Whether grammar is currently being enforced via bitmask."""

    tools_forced: bool
    """Whether tool calling was forced (tool_choice=required or named function)."""

    requires_structured_output_flag: bool
    """Whether this request requires ``--enable-structured-output`` to be set."""

    def set_tool_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        """Set token sequences for conditional tool call enforcement.

        Args:
            start_token_ids: Token IDs marking tool call start.
            end_token_ids: Token IDs marking tool call end.
        """
        ...

    def set_thinking_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        """Configure thinking region for conditional grammar enforcement.

        Args:
            start_token_ids: Token IDs marking thinking start.
            end_token_ids: Token IDs marking thinking end (e.g., ``</think>``).
        """
        ...

    def update_enforcement_state(self, token: int) -> bool:
        """Advance the grammar-enforcement state machine by one token.

        Args:
            token: The newly committed token.

        Returns:
            True if the matcher should consume the token.
        """
        ...

    def snapshot_grammar_state(self) -> GrammarEnforcementSnapshot:
        """Capture enforcement state for a speculative rollback."""
        ...

    def restore_grammar_state(
        self, snapshot: GrammarEnforcementSnapshot
    ) -> None:
        """Restore state captured by ``snapshot_grammar_state``."""
        ...


TextGenerationContextType = TypeVar(
    "TextGenerationContextType", bound=TextGenerationContext
)
"""Type variable for text generation context types, constrained to TextGenerationContext."""


# ---------------------------------------------------------------------------
# VLM protocol and TypeVar
# ---------------------------------------------------------------------------


@runtime_checkable
class VLMTextGenerationContext(TextGenerationContext, Protocol):
    """Protocol defining the interface for VLM input contexts."""

    @property
    def image_idx(self) -> int:
        """Index of the next unencoded image in the prompt."""
        ...

    @property
    def images(self) -> list[ImageMetadata]:
        """The images in the context."""
        ...

    @property
    def next_images(self) -> list[ImageMetadata]:
        """The images that are not yet encoded."""
        ...

    @property
    def needs_vision_encoding(self) -> bool:
        """Whether vision encoding is needed for this context."""
        ...

    @property
    def image_token_indices(self) -> npt.NDArray[np.int32]:
        """Positions of image-placeholder tokens within this context's token buffer."""
        ...

    def compute_image_aligned_idx(self, idx: int) -> int:
        """Aligns an index downward to avoid splitting an image token span.

        Args:
            idx: The candidate index into the token sequence.

        Returns:
            The adjusted index, guaranteed not to split an image token span.
        """
        ...


VLMContextType = TypeVar("VLMContextType", bound=VLMTextGenerationContext)
"""Type variable for VLM context types, constrained to VLMTextGenerationContext."""


# ---------------------------------------------------------------------------
# Pixel generation protocol and TypeVar
# ---------------------------------------------------------------------------


@runtime_checkable
class PixelGenerationContext(BaseContext, Protocol):
    """Protocol defining the interface for pixel generation contexts.

    A ``PixelGenerationContext`` represents model inputs for pixel generation
    pipelines, managing the state and parameters needed for generating images
    or videos.
    """

    @property
    def tokens(self) -> TokenBuffer:
        """The token buffer for the context."""
        ...

    @property
    def latents(self) -> npt.NDArray[np.float32]:
        """The latents for the context."""
        ...

    @property
    def height(self) -> int:
        """Height of generated output in pixels."""
        ...

    @property
    def width(self) -> int:
        """Width of generated output in pixels."""
        ...

    @property
    def num_inference_steps(self) -> int:
        """Number of denoising steps."""
        ...

    @property
    def guidance_scale(self) -> float:
        """Classifier-free guidance scale (1.0 to disable CFG)."""
        ...

    @property
    def num_images_per_prompt(self) -> int:
        """Number of images to generate."""
        ...


PixelGenerationContextType = TypeVar(
    "PixelGenerationContextType", bound=PixelGenerationContext
)
"""Type variable for pixel generation context types, constrained to PixelGenerationContext."""


# ---------------------------------------------------------------------------
# Concrete helpers used by the context implementations below
# ---------------------------------------------------------------------------


@dataclass
class StructuredOutputRegionDelimiters:
    """Token ID sequences that define structured output boundaries.

    Used for conditional grammar enforcement: when the start sequence
    is detected, grammar enforcement activates. When the end sequence
    is detected, enforcement deactivates.
    """

    start_token_ids: list[int] | None = None
    """Token ID sequence marking the start of a structured output region."""

    end_token_ids: list[int] | None = None
    """Token ID sequence marking the end of a structured output region."""


@dataclass
class GrammarEnforcementState:
    """Manages grammar enforcement state for constrained decoding.

    Encapsulates the logic for tracking whether grammar is currently being
    enforced, detecting tool call and thinking region boundary token sequences,
    and managing the token buffer for multi-token sequence matching.

    The key transitions are: detecting ``</think>`` exits the thinking region,
    and detecting tool-call start/end tokens toggles enforcement on/off for
    ``tool_choice=auto``. For ``tool_choice=required`` or a JSON schema,
    enforcement is active from the first generated token (after any thinking
    region is exited).
    """

    grammar_enforced: bool = False
    """Whether grammar is currently being enforced via bitmask.

    For tool_choice=required or response_format: True from start.
    For tool_choice=auto without response_format: False initially,
    flipped to True when tool call start token is detected.
    """

    tools_forced: bool = False
    """Whether tool calling was forced (tool_choice=required or named).

    Controls whether grammar_enforced is True from the first generated token.
    Independent of the --enable-structured-output server flag (which only gates
    user-supplied schemas; see ``requires_structured_output_flag``).
    """

    requires_structured_output_flag: bool = False
    """Whether this request requires --enable-structured-output to be set.

    True when the constraint includes a user-supplied JSON schema. False for
    pure tool-call grammars derived from the model's tool parser.
    """

    has_json_schema: bool = False
    """Whether this request includes a JSON schema response format."""

    tool_region: StructuredOutputRegionDelimiters | None = None
    """Token sequences defining tool call boundaries, if conditional enforcement."""

    thinking_region_delimiters: StructuredOutputRegionDelimiters | None = None
    """Token sequences defining thinking boundaries (e.g., ``</think>``).

    When set, grammar enforcement is suspended inside thinking regions.
    The key insight is that when thinking is enabled, the chat template
    already emits ``<think>`` in the prompt, so we start in thinking region
    and only need to detect ``</think>`` to exit.
    """

    _in_thinking_region: bool = False
    """Whether currently inside a thinking region.

    TODO: Consider consolidating with ``in_thinking_phase`` in text generation pipeline.
    """

    _tool_calling_match_buffer: list[int] = field(default_factory=list)
    """Buffer for partial matching of multi-token start/end tags."""

    _thinking_match_buffer: list[int] = field(default_factory=list)
    """Buffer for partial matching of thinking end sequence."""

    @classmethod
    def from_response_format(
        cls, response_format: TextGenerationResponseFormat | None
    ) -> GrammarEnforcementState:
        """Creates a state from the given response format, or a default state."""
        if not response_format:
            return cls()
        return cls(
            grammar_enforced=response_format.grammar_enforced,
            tools_forced=response_format.tools_forced,
            requires_structured_output_flag=response_format.requires_structured_output_flag,
            has_json_schema=response_format.has_json_schema,
        )

    def update_enforcement_state(self, token: int) -> bool:
        """Update enforcement state based on sampled token.

        Checks if the token completes a start/end sequence and
        toggles grammar_enforced accordingly. Thinking region transitions
        take priority over tool region transitions.

        Args:
            token: The newly sampled token.

        Returns:
            True if the matcher should consume the token.
        """
        # Check thinking region transitions FIRST (higher priority).
        # Thinking-end delimiter is NOT grammar content — return False
        # so the caller skips the matcher even though enforcement resumed.
        if (
            self.thinking_region_delimiters is not None
            and self._in_thinking_region
        ):
            if (
                self.thinking_region_delimiters.end_token_ids is not None
                and self._check_sequence_match_with_buffer(
                    token,
                    self.thinking_region_delimiters.end_token_ids,
                    self._thinking_match_buffer,
                )
            ):
                self._in_thinking_region = False
                if self.tools_forced or self.has_json_schema:
                    self.grammar_enforced = True
            return False

        # Tool region logic (for tool_choice=auto). Skipped when tools_forced
        # is set: forced grammars enforce start-to-finish via the regex
        # itself, so auto-mode toggles must not flip grammar_enforced.
        # Both start and end tags ARE grammar content — return True so the
        # caller feeds the token to the matcher before enforcement flips.
        if self.tool_region is not None and not self.tools_forced:
            if not self.grammar_enforced:
                if (
                    self.tool_region.start_token_ids is not None
                    and self._check_sequence_match(
                        token, self.tool_region.start_token_ids
                    )
                ):
                    self.grammar_enforced = True
                    return True
            else:
                if (
                    self.tool_region.end_token_ids is not None
                    and self._check_sequence_match(
                        token, self.tool_region.end_token_ids
                    )
                ):
                    self.grammar_enforced = False
                    self._tool_calling_match_buffer.clear()
                    return True

        return self.grammar_enforced

    def snapshot(self) -> GrammarEnforcementSnapshot:
        """Capture state needed to roll back a speculative advance.

        The speculative bitmask path walks the enforcement state through
        draft tokens to compute downstream slot constraints, then
        unwinds so that committed-token processing on the next batch
        replays the same transitions from a clean state. The returned
        snapshot is opaque to callers; pass it to `restore`.
        """
        return GrammarEnforcementSnapshot(
            in_thinking_region=self._in_thinking_region,
            grammar_enforced=self.grammar_enforced,
            tool_calling_match_buffer=list(self._tool_calling_match_buffer),
            thinking_match_buffer=list(self._thinking_match_buffer),
        )

    def restore(self, snapshot: GrammarEnforcementSnapshot) -> None:
        """Restore state captured by :meth:`snapshot`."""
        self._in_thinking_region = snapshot.in_thinking_region
        self.grammar_enforced = snapshot.grammar_enforced
        self._tool_calling_match_buffer[:] = snapshot.tool_calling_match_buffer
        self._thinking_match_buffer[:] = snapshot.thinking_match_buffer

    def _check_sequence_match(self, token: int, target: list[int]) -> bool:
        """Check if token completes a target sequence using the default buffer."""
        return self._check_sequence_match_with_buffer(
            token, target, self._tool_calling_match_buffer
        )

    def _check_sequence_match_with_buffer(
        self, token: int, target: list[int], buffer: list[int]
    ) -> bool:
        """Check if token completes a target sequence using a specific buffer."""
        buffer.append(token)

        max_len = len(target)
        if len(buffer) > max_len:
            del buffer[: len(buffer) - max_len]

        if buffer[-len(target) :] == target:
            buffer.clear()
            return True
        return False


# ---------------------------------------------------------------------------
# Concrete context implementations
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TextContext:
    """A base class for model context, specifically for Text model variants.

    This class manages the state and processing of text generation, including
    token management, caching, and generation parameters.

    Configuration:
        request_id: A unique identifier for this sequence.
        max_length: Maximum allowed length of the generated sequence
        tokens: NumPy array containing the token IDs
        eos_tracker: holds EOS config and performs checks for EOS conditions
        log_probabilities: Whether to return token log probabilities
        log_probabilities_echo: Whether to return log probabilities for prompt tokens
        ignore_eos: Whether to ignore end of sequence tokens and continue generating
        matcher: Optional grammar matcher for constrained decoding
        json_schema: Optional JSON schema for structured output
        sampling_params: Parameters controlling the token sampling strategy
        min_tokens: Minimum number of new tokens to generate.
        target_endpoint: Optional target endpoint identifier for routing requests
        _status: Current generation status (active, finished, etc)
        _log_probabilities_data: Token log probabilities data
        _is_initial_prompt: Whether this is the initial prompt encoding
        _draft_offset: Offset for draft decoding
        _spec_decoding_state: Optional per-request speculative decoding state
        vocab_size: Optional vocabulary size for validating generated token IDs
    """

    max_length: int
    tokens: TokenBuffer
    request_id: RequestID = field(default_factory=RequestID)
    eos_tracker: EOSTracker = field(default_factory=EOSTracker)
    vocab_size: int | None = field(default=None)
    log_probabilities: int = field(default=0)
    log_probabilities_echo: bool = field(default=False)
    ignore_eos: bool = field(default=False)
    json_schema: str | None = field(default=None)
    grammar: str | None = field(default=None)
    """Grammar for constrained decoding (e.g., regex grammar).

    When set, this takes precedence over ``json_schema``. Used for model-specific
    constrained decoding formats like Kimi's tool call grammar.
    """

    grammar_state: GrammarEnforcementState = field(
        default_factory=GrammarEnforcementState
    )
    """Grammar enforcement state for constrained decoding."""

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    model_name: str = field(default="")
    _matcher: Any | None = field(default=None)
    status: GenerationStatus = field(default=GenerationStatus.ACTIVE)
    _log_probabilities_data: dict[int, LogProbabilities] = field(
        default_factory=dict
    )

    _is_initial_prompt: bool = field(default=True)
    _draft_offset: int = field(default=0)
    _spec_decoding_state: SpecDecodingState | None = field(default=None)

    in_reasoning_phase: bool = field(default=False)
    """Whether the latest committed tokens are inside a ``<think>...</think>``
    block. Toggled host-side after each commit when a reasoning parser is
    configured."""

    target_endpoint: str | None = field(default=None)

    external_block_metadata: Any = field(default=None)
    """Block metadata from the Orchestrator for distributed KV cache (dKV).

    When set, the DKVConnector reads this during lookup() to determine
    which blocks are available in the external BlockStore system.
    """

    dkv_hint_instance_name: str = field(default="")
    """Instance name from the Orchestrator's dkv_cache_hint identifying
    the dKV instance that owns the cached blocks. The DKVConnector
    compares this to its own instance name (learned via
    ExchangeMetadata) and skips the fetch when they match — those
    blocks are owned locally and surface through MAX's own prefix
    cache instead.
    """

    cached_prefix_length: int | None = field(default=None)
    """Number of prompt tokens served from the KV prefix cache.

    Set by the block manager when a request is admitted to a CE batch
    (0 if no matching prefix). ``BatchMetrics.create``
    consumes the value to emit a per-request cache hit rate observation, and
    uses ``_cache_metrics_emitted`` to guard against re-emitting on
    chunked-prefill follow-up calls.
    """

    _cache_metrics_emitted: bool = field(default=False)
    """Set to ``True`` after the first CE batch to prevent re-emitting cache hit metrics on chunked-prefill follow-up calls."""

    def __post_init__(self) -> None:
        """Initialize context state after deserialization.

        This method is called each time the model is deserialized from msgspec.
        """
        if self.min_tokens + self.tokens.prompt_length > self.max_length:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) + prompt_len ({self.tokens.prompt_length}) must be less than or equal to max_length ({self.max_length})"
            )

        if self.target_endpoint is not None:
            if not self.target_endpoint.startswith(("tcp://", "ipc://")):
                raise ValueError(
                    f"target_endpoint must be prefixed with 'tcp://' or 'ipc://': {self.target_endpoint}"
                )
            if (
                self.target_endpoint.startswith("tcp://")
                and ":" not in self.target_endpoint.split("://")[-1]
            ):
                raise ValueError(
                    f"target_endpoint must contain a port if using tcp: {self.target_endpoint}"
                )

    @property
    def is_done(self) -> bool:
        """Whether text generation has finished."""
        return self.status.is_done

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens to generate."""
        return self.sampling_params.min_new_tokens

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        """Gets or creates the per-request speculative decoding state."""
        if self._spec_decoding_state is None:
            self._spec_decoding_state = SpecDecodingState()
        return self._spec_decoding_state

    def apply_processing_offset(self, offset: int) -> None:
        """Applies a processing offset to the token buffer."""
        self.tokens.apply_processing_offset(offset)

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns per-step masks for logits that should be masked (e.g. EOS during ``min_tokens``).

        This is primarily used for the ``min_tokens`` setting, where we mask
        EOS tokens in the logits to avoid generating them before we reach
        ``min_tokens``.

        Returns:
            A list of arrays, one per step; each array has shape ``(N, 2)`` with
            (batch index, token ID) pairs for logits to mask.
        """
        ret_list: list[npt.NDArray[np.int32]] = []
        start_range = self.tokens.prompt_length
        end_range = self.tokens.prompt_length + self.min_tokens

        for i in range(
            self.tokens.current_position,
            self.tokens.current_position + num_steps,
        ):
            if i < start_range or i >= end_range:
                ret_list.append(np.zeros((0, 2), dtype=np.int32))
                continue

            new_list = []
            for eos_token_id in self.eos_tracker.eos_token_ids:
                new_list.append((0, eos_token_id))

            ret_list.append(np.asarray(new_list, dtype=np.int32))

        return ret_list

    def set_matcher(self, matcher: llguidance.LLMatcher) -> None:
        """Sets the grammar matcher for constrained decoding."""
        self._matcher = matcher

    @property
    def matcher(self) -> llguidance.LLMatcher | None:
        """The optional grammar matcher for constrained decoding."""
        return self._matcher

    @property
    def grammar_enforced(self) -> bool:
        """Whether grammar is currently being enforced."""
        return self.grammar_state.grammar_enforced

    @grammar_enforced.setter
    def grammar_enforced(self, value: bool) -> None:
        self.grammar_state.grammar_enforced = value

    @property
    def tools_forced(self) -> bool:
        """Whether tool calling was forced."""
        return self.grammar_state.tools_forced

    @tools_forced.setter
    def tools_forced(self, value: bool) -> None:
        self.grammar_state.tools_forced = value

    @property
    def requires_structured_output_flag(self) -> bool:
        """Whether this request requires --enable-structured-output."""
        return self.grammar_state.requires_structured_output_flag

    @requires_structured_output_flag.setter
    def requires_structured_output_flag(self, value: bool) -> None:
        self.grammar_state.requires_structured_output_flag = value

    def set_tool_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        """Set token sequences for conditional tool call enforcement.

        Args:
            start_token_ids: Token IDs marking tool call start.
            end_token_ids: Token IDs marking tool call end.
        """
        if start_token_ids is not None or end_token_ids is not None:
            self.grammar_state.tool_region = StructuredOutputRegionDelimiters(
                start_token_ids=start_token_ids,
                end_token_ids=end_token_ids,
            )

    def set_thinking_region(
        self,
        start_token_ids: list[int] | None,
        end_token_ids: list[int] | None,
    ) -> None:
        """Configure thinking region for conditional grammar enforcement.

        When a thinking region is configured and ``_in_thinking_region`` is True,
        grammar enforcement is suspended until the end token sequence is detected.
        This enables reasoning output during constrained decoding.

        Args:
            start_token_ids: Token IDs marking thinking start (can be None if we
                start inside thinking, which is the case when chat template
                already emits ``<think>``).
            end_token_ids: Token IDs marking thinking end (e.g., ``</think>``).
        """
        if end_token_ids is not None:
            self.grammar_state.thinking_region_delimiters = (
                StructuredOutputRegionDelimiters(
                    start_token_ids=start_token_ids,
                    end_token_ids=end_token_ids,
                )
            )

    def update_enforcement_state(self, token: int) -> bool:
        """Advance the grammar-enforcement state machine by one token.

        Forwards to :meth:`GrammarEnforcementState.update_enforcement_state`.

        Args:
            token: The newly committed token.

        Returns:
            True if the matcher should consume the token.
        """
        return self.grammar_state.update_enforcement_state(token)

    def snapshot_grammar_state(self) -> GrammarEnforcementSnapshot:
        """Forwards to `GrammarEnforcementState.snapshot`."""
        return self.grammar_state.snapshot()

    def restore_grammar_state(
        self, snapshot: GrammarEnforcementSnapshot
    ) -> None:
        """Forwards to `GrammarEnforcementState.restore`."""
        self.grammar_state.restore(snapshot)

    def to_generation_output(self) -> TextGenerationOutput:
        """Get completion tokens that are ready to be returned to the user.

        This method retrieves tokens that have been generated but not yet
        delivered to the user, along with their associated log probability data.

        Returns:
            TextGenerationOutput: The completion tokens and their associated
            log probabilities, if available.
        """
        # Return early, if we have no outstanding generated tokens
        if not self.tokens.has_outstanding_generated_tokens:
            return TextGenerationOutput(
                request_id=self.request_id,
                tokens=[],
                log_probabilities=None,
                final_status=self.status,
                num_cached_tokens=self.cached_prefix_length,
            )

        element_ids = range(
            self.tokens._completion_range.start,
            self.tokens._completion_range.end,
        )
        # Consume Generated Tokens
        if len(element_ids) > 0:
            generated_tokens = [
                int(x) for x in self.tokens.consume_recently_generated_tokens()
            ]
            if FUTURE_TOKEN in generated_tokens:
                raise ValueError(
                    "Attempted to create generation output while future token is not yet realized."
                )
            for token_id in generated_tokens:
                if token_id < 0:
                    raise RuntimeError(
                        f"Generated negative token_id={token_id} for request {self.request_id}"
                    )
                if self.vocab_size is not None and token_id >= self.vocab_size:
                    raise RuntimeError(
                        f"Generated out-of-vocabulary token_id={token_id} for request {self.request_id} (valid range: [0, {self.vocab_size}))"
                    )
        else:
            generated_tokens = []

        # Retrieve Log Probabilities
        log_probabilities: list[LogProbabilities] | None = None
        for token_idx in element_ids:
            if token_idx in self._log_probabilities_data:
                if log_probabilities is None:
                    log_probabilities = []

                log_probabilities.append(
                    self._log_probabilities_data.pop(token_idx)
                )

        return TextGenerationOutput(
            request_id=self.request_id,
            tokens=generated_tokens,
            log_probabilities=log_probabilities,
            final_status=self.status,
            num_cached_tokens=self.cached_prefix_length,
        )

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance the token buffer without touching FSM state.

        This method handles token buffer mutations including:
        - Chunked prefill advancement
        - Log probability storage
        - Token buffer advancement
        - EOS/max-length status updates

        It does NOT advance the FSM matcher. Use ``advance_fsm()`` separately
        if FSM advancement is needed, or use ``update()`` for the common case
        of advancing both together.

        Args:
            new_token: The token to append to the buffer.
            log_probabilities: Optional log probabilities for this token.
        """
        if self.tokens.actively_chunked:
            self.tokens.advance_chunk()
            return

        if log_probabilities:
            self._log_probabilities_data[self.tokens.current_position] = (
                log_probabilities
            )

        if self.tokens.all[-1] == FUTURE_TOKEN:
            raise ValueError("Cannot append a token after a future token.")

        self.tokens.advance_with_token(new_token)

        if self.eos_tracker.is_eos_from_tokens(self.tokens.generated):
            self.status = GenerationStatus.END_OF_SEQUENCE
        elif self.tokens.current_position >= self.max_length:
            self.status = GenerationStatus.MAXIMUM_LENGTH

        self._is_initial_prompt = False

    def advance_fsm(self, token: int) -> bool:
        """Advance the FSM matcher state by one token.

        This method:
        1. Updates enforcement state based on tool call boundaries (if conditional)
        2. Advances the FSM if grammar is currently enforced

        It does NOT modify the token buffer. Use ``advance_token_buffer()``
        separately if token buffer advancement is needed, or use ``update()``
        for the common case of advancing both together.

        Matcher rejection is not expected at this point (assuming the
        bitmask was applied correctly). But if the matcher does reject
        a token, enforcement is disabled for the rest of the request.
        Continuing to enforce against a desynced matcher would produce
        schema-shaped nonsense (every downstream bitmask would be filtered
        against a stale grammar position with no relation to what was emitted).
        Instead we let the request finish unconstrained.

        Args:
            token: The token to consume in the FSM.

        Returns:
            True if the token was handled — either consumed by the FSM,
            recognized as a state-transition delimiter (e.g. a
            thinking-end token), or skipped because enforcement is
            inactive. False only when no matcher is present.
        """
        if self.matcher is None:
            return False

        # EOS tokens are not part of the grammar — they signal the end of
        # generation, not schema content.  Skip the matcher so it stays in
        # a clean terminal state rather than logging a spurious rejection.
        if token in self.eos_tracker.eos_token_ids:
            self.grammar_state.grammar_enforced = False
        elif (
            self.grammar_state.update_enforcement_state(token)
            and self.matcher.try_consume_tokens([token]) != 1
        ):
            _logger.error(
                "Matcher rejected token %d (request %s); disabling "
                "enforcement for the rest of the request. "
                "matcher_errors=%s matcher_warnings=%s",
                token,
                self.request_id,
                self.matcher.get_error(),
                self.matcher.get_grammar_warnings(),
            )
            self.grammar_state.grammar_enforced = False

        return True

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance both token buffer and FSM state.

        This is the standard single-step update that most callers should use.
        It combines ``advance_token_buffer()`` and ``advance_fsm()`` for the
        common case where both need to be advanced together.

        For multi-step execution where FSM is advanced separately (e.g., to
        compute bitmasks between steps), use the individual methods directly.

        Args:
            new_token: The token to append and consume.
            log_probabilities: Optional log probabilities for this token.
        """
        self.advance_token_buffer(new_token, log_probabilities)
        self.advance_fsm(new_token)

    def update_with_future_token(self) -> None:
        """Append a placeholder future token to the generated tokens.

        This is primarily used for overlap scheduling. For structured output
        contexts (those with a matcher), only the token buffer is advanced.
        The FSM will be advanced later when the future token is realized
        with the actual generated token.
        """
        if self.tokens.all[-1] == FUTURE_TOKEN:
            raise ValueError("Cannot have multiple future tokens.")

        if self.matcher is not None:
            # For structured output, only advance the token buffer.
            # The FSM cannot accept placeholder tokens and will be
            # advanced with the real token in sync_and_process_outputs().
            self.advance_token_buffer(FUTURE_TOKEN)
        else:
            self.update(new_token=FUTURE_TOKEN)

    def realize_future_token(
        self, new_token: int, log_probabilities: LogProbabilities | None = None
    ) -> None:
        """Overwrite the placeholder future token with the actual token.

        This is primarily used for overlap scheduling.
        """
        if self.tokens.generated_length == 0:
            raise ValueError(
                "Cannot realize a future token when there are no generated tokens."
            )

        if self.tokens.all[-1] != FUTURE_TOKEN:
            raise ValueError(
                "Attempted to realize a non-future token. Found token: ",
                self.tokens.all[-1],
            )

        # Overwrite the log probabilities data
        if log_probabilities:
            self._log_probabilities_data[self.tokens.current_position - 1] = (
                log_probabilities
            )

        self.tokens.overwrite_last_token(new_token)

        if self.eos_tracker.is_eos_from_tokens(self.tokens.generated):
            self.status = GenerationStatus.END_OF_SEQUENCE

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        delete_last_generated_token = self.tokens.all[-1] == FUTURE_TOKEN
        self.tokens.reset_as_new_prompt(
            delete_last_generated_token=delete_last_generated_token
        )
        self._is_initial_prompt = True
        self._spec_decoding_state = None

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Computes the maximum number of steps without exceeding ``max_seq_len``.

        Takes the current context length into account.
        """
        return max_seq_len - (len(self.tokens) - self.tokens.active_length)

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        return self._is_initial_prompt


@dataclass(kw_only=True)
class TextAndVisionContext(TextContext):
    """A base class for model context, specifically for Vision model variants.

    For example::

      - <vision_start_token_id> = 97
      - <vision_token_id> = 98
      - <vision_end_token_id> = 99

    Token array::

      -       idx: [  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ]
      - token_ids: [ 51 52 53 54 97 98 98 98 98 99 55 56 57 58 97 98 98 98 98 99 59 60 61 62 ]
                                    ^-- img0 --^                  ^-- img1 --^
                                                       ^ start_idx=11 (image_idx=1)

    Then we would have::

      - ImageMetadata(start_idx=5, end_idx=9, ...)  # img0
      - ImageMetadata(start_idx=15, end_idx=19, ...)  # img1

    These image ranges should be non-overlapping.

    The image_idx is determined based on the value of start_idx. It is the idx of
    the first image that is not yet encoded. For example in the above diagram
    when start_idx=11, this implies that image_idx=1.

    When chunk prefill is **not** active, we restrict current_position from being in the
    middle of an image.  This is verified in `_validate_state` which is called before and
    after mutating methods like `_bump_token_indices`.  During chunked prefill the
    restriction is relaxed because the vision encoder cache ensures images are encoded
    once and reused across chunks.
    """

    vision_token_ids: list[int]
    """The value of the <vision_token_id> special token. The reason this is a list
    is primarily due to Pixtral which also has a image_break_token_id."""

    images: list[ImageMetadata] = field(default_factory=list)
    """Metadata about each image in the prompt. """

    extra_model_args: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    """Extra model arguments for the vision model. These are model specific arguments."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.images) > 0:
            for prev_img, next_img in zip(
                self.images[:-1], self.images[1:], strict=True
            ):
                if next_img.start_idx < prev_img.start_idx:
                    raise ValueError("Images must be sorted")
                if next_img.start_idx <= prev_img.end_idx:
                    raise ValueError("Images must be non-overlapping")

        for img in self.images:
            if len(self.tokens) < img.end_idx:
                raise ValueError(
                    "Images must be before the end of the token array"
                )

            # Instead of checking all tokens in the image (which can be expensive),
            # we only check the first and last tokens.
            if (
                self.tokens[img.start_idx] not in self.vision_token_ids
                or self.tokens[img.end_idx - 1] not in self.vision_token_ids
            ):
                raise ValueError(
                    f"Images must be filled with <vision_token_id> ({self.vision_token_ids})"
                )

        self._validate_state()

    @property
    def image_idx(self) -> int:
        """Index of the next unencoded image in the prompt."""
        for i, img in enumerate(self.images):
            if self.tokens.processed_length < img.end_idx:
                return i
        return len(self.images)

    @property
    def next_images(self) -> list[ImageMetadata]:
        """Returns the images that are not yet encoded."""
        image_idx = self.image_idx
        if len(self.images) == 0 or self.image_idx == len(self.images):
            return []
        return self.images[image_idx:]

    @property
    def needs_vision_encoding(self) -> bool:
        """Returns whether vision encoding is needed for this context."""
        return self.image_idx < len(self.images)

    @property
    def image_token_indices(self) -> npt.NDArray[np.int32]:
        """Positions of image-placeholder tokens in the full token sequence.

        Derived from ``images`` metadata.  Subclasses that precompute indices
        at tokenization time (e.g. KimiK2.5, Qwen2.5VL) may override this
        with a stored field for efficiency.
        """
        if not self.images:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(
            [
                np.arange(img.start_idx, img.end_idx, dtype=np.int32)
                for img in self.images
            ]
        )

    def compute_image_aligned_idx(self, idx: int) -> int:
        """Possibly aligns a index value downward if it lies in the middle of an image."""
        for img in self.images:
            if img.start_idx <= idx < img.end_idx:
                return img.start_idx
        return idx

    def _find_bisected_image(self, idx: int) -> ImageMetadata | None:
        """Returns an image if the given index lies in the middle of an image.

        This means that there are image tokens in both [0:idx) and [idx:end).

        As such, this does NOT include the start or end indices.
        """
        for img in self.images:
            if img.start_idx < idx < img.end_idx:
                return img
        return None

    def _validate_state(self) -> None:
        """Validates the state of the context."""
        # During chunked prefill, current_position may bisect an image
        # because the vision encoder cache handles re-encoding.
        if not self.tokens.actively_chunked and (
            img := self._find_bisected_image(self.tokens.current_position)
        ):
            raise ValueError(
                f"It is invalid for the current_position ({self.tokens.current_position}) to bisect an image ({img})."
            )

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Updates the context with a new token and validates vision state."""
        super().update(new_token=new_token, log_probabilities=log_probabilities)
        self._validate_state()


@dataclass(kw_only=True)
class PixelContext:
    """A model-ready context for image/video generation requests.

    Per the design doc, this class contains only numeric data that the model
    will execute against. User-facing strings (prompt, negative_prompt) are
    consumed during tokenization and do not appear here.

    All preprocessing is performed by PixelGenerationTokenizer.new_context():
    - Prompt tokenization -> tokens field
    - Negative prompt tokenization -> negative_tokens field
    - Timestep schedule computation -> timesteps field
    - Initial noise generation -> latents field

    Configuration:
        tokens: Tokenized prompt IDs (TokenBuffer).
        request_id: A unique identifier for this generation request.
        negative_tokens: Tokenized negative prompt IDs (TokenBuffer).
        timesteps: Precomputed timestep schedule for denoising.
        latents: Precomputed initial noise (latents).
        height: Height of the generated image/video in pixels.
        width: Width of the generated image/video in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for classifier-free guidance.
        num_images_per_prompt: Number of images/videos to generate per prompt.
        input_image: Optional HWC uint8 numpy array for image-to-image generation.
        input_images: Optional list of input images for image-to-image generation.
        model_name: Name of the model being used.
    """

    # Tokenized prompts
    tokens: TokenBuffer
    """Primary encoder tokens."""

    # Request identification
    request_id: RequestID = field(default_factory=RequestID)

    model_name: str = field(default="")

    mask: npt.NDArray[np.bool_] | None = field(default=None)
    """Mask for text encoder's attention."""

    tokens_2: TokenBuffer | None = field(default=None)
    """Secondary encoder tokens. None for single-encoder models."""

    negative_tokens: TokenBuffer | None = field(default=None)
    """Negative tokens for primary encoder."""

    negative_mask: npt.NDArray[np.bool_] | None = field(default=None)
    """Mask for the negative text encoder path."""

    negative_tokens_2: TokenBuffer | None = field(default=None)
    """Negative tokens for secondary encoder. None for single-encoder models."""

    explicit_negative_prompt: bool = field(default=False)
    """Whether the request explicitly supplied a negative prompt."""

    # Precomputed tensors
    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed timesteps schedule for denoising."""

    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed sigmas schedule for denoising."""

    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed initial noise (latents) for generation."""

    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed latent image IDs for generation."""

    text_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Precomputed text position IDs, shape ``(B, seq_len, 4)`` int64."""

    negative_text_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Precomputed text position IDs for the negative prompt."""

    height: int = field(default=1024)
    width: int = field(default=1024)
    num_inference_steps: int = field(default=50)
    guidance_scale: float = field(default=3.5)
    true_cfg_scale: float = field(default=1.0)
    strength: float = field(default=0.6)
    cfg_normalization: bool = field(default=False)
    cfg_truncation: float = field(default=1.0)
    num_warmup_steps: int = field(default=0)
    num_images_per_prompt: int = field(default=1)
    input_image: npt.NDArray[np.uint8] | None = field(default=None)
    """Input image as numpy array (H, W, C) in uint8 format for image-to-image generation."""
    input_images: list[npt.NDArray[np.uint8]] | None = field(default=None)
    """Input images as list of numpy arrays (H, W, C) in uint8 format for image-to-image generation."""
    prompt_images: list[npt.NDArray[np.uint8]] | None = field(default=None)
    """Optional prompt-conditioning images prepared by the tokenizer."""
    vae_condition_images: list[npt.NDArray[np.uint8]] | None = field(
        default=None
    )
    """Optional VAE-conditioning images prepared by the tokenizer.

    Qwen image edit keeps prompt-conditioning images and VAE-conditioning
    images separate because the multimodal prompt encoder and the VAE latent
    conditioning path use different resize targets.
    """
    output_format: str = field(default="jpeg")
    """Image encoding format for the output (e.g., 'jpeg', 'png', 'webp')."""
    status: GenerationStatus = field(default=GenerationStatus.ACTIVE)

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        """Compute number of available steps for scheduler compatibility.

        For image and video generation, this returns the number of inference steps.
        """
        return self.num_inference_steps

    def reset(self) -> None:
        """Resets the context's state."""
        self.status = GenerationStatus.ACTIVE

    def update(self, latents: npt.NDArray[Any]) -> None:
        """Update the context with newly generated latents/image data."""
        self.latents = latents

    def to_generation_output(self) -> GenerationOutput:
        """Convert this context to a GenerationOutput object."""
        return GenerationOutput(
            request_id=self.request_id,
            final_status=self.status,
            output=[
                OutputImageContent.from_numpy(
                    self.latents.astype(np.uint8), format="png"
                )
            ],
        )


if TYPE_CHECKING:
    # Verify that concrete classes implement their respective protocols
    def _verify_text_context_protocol() -> TextGenerationContext:
        return TextContext(
            request_id=RequestID(),
            max_length=5,
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            eos_tracker=EOSTracker(),
        )

    def _verify_vlm_context_protocol() -> VLMTextGenerationContext:
        return TextAndVisionContext(
            request_id=RequestID(),
            max_length=5,
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            eos_tracker=EOSTracker(),
            vision_token_ids=[],
            images=[],
        )

    def _verify_pixel_context_protocol() -> PixelGenerationContext:
        return PixelContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        )
