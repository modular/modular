# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Protocol, TypeVar, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt

from .log_probabilities import LogProbabilities
from .logit_processors_type import LogitsProcessor
from .request import RequestID
from .status import GenerationStatus

logger = logging.getLogger("max.pipelines")


@dataclass
class SamplingParamsInput:
    """Input dataclass for creating SamplingParams instances.

    All fields are optional, allowing partial specification with None values
    indicating "use default". This enables static type checking while maintaining
    the flexibility to specify only the parameters you want to override.
    """

    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    temperature: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    min_new_tokens: int | None = None
    ignore_eos: bool | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    detokenize: bool | None = None
    seed: int | None = None
    logits_processors: Sequence[LogitsProcessor] | None = None


@dataclass(frozen=True)
class SamplingParamsGenerationConfigDefaults:
    """Default sampling parameter values extracted from a model's GenerationConfig.

    This class encapsulates sampling parameter defaults that come from a HuggingFace
    model's GenerationConfig. These defaults have middle priority when creating
    SamplingParams instances:

    Priority order (highest to lowest):
    1. User-provided values (SamplingParamsInput)
    2. Model's GenerationConfig values (this class)
    3. SamplingParams class defaults

    All fields default to None, indicating that the model's GenerationConfig does not
    explicitly set that parameter. When None, SamplingParams will fall back to its
    own class defaults.

    Example:
        >>> # Extract from model config
        >>> gen_config = model_config.generation_config
        >>> defaults = SamplingParamsGenerationConfigDefaults(
        ...     temperature=0.7,
        ...     top_k=50,
        ...     max_new_tokens=512
        ... )
        >>> # Use with SamplingParams
        >>> params = SamplingParams.from_input_and_generation_config(
        ...     SamplingParamsInput(),
        ...     sampling_params_defaults=defaults
        ... )
    """

    temperature: float | None = None
    """Temperature value from the model's GenerationConfig, if explicitly set."""

    top_p: float | None = None
    """Top-p (nucleus sampling) value from the model's GenerationConfig, if explicitly set."""

    top_k: int | None = None
    """Top-k sampling value from the model's GenerationConfig, if explicitly set."""

    repetition_penalty: float | None = None
    """Repetition penalty value from the model's GenerationConfig, if explicitly set."""

    max_new_tokens: int | None = None
    """Maximum number of new tokens from the model's GenerationConfig, if explicitly set."""

    min_new_tokens: int | None = None
    """Minimum number of new tokens from the model's GenerationConfig, if explicitly set."""

    do_sample: bool | None = None
    """If False, use greedy sampling."""

    @cached_property
    def values_to_update(self) -> dict[str, float | int]:
        """Extract non-None field values as a dictionary.

        Returns:
            A dictionary mapping field names to their values, excluding any fields
            that are None. This dictionary can be used to update SamplingParams
            default values.

        Example:
            >>> defaults = SamplingParamsGenerationConfigDefaults(
            ...     temperature=0.7,
            ...     top_k=50
            ... )
            >>> defaults.values_to_update
            {'temperature': 0.7, 'top_k': 50}
        """
        values = {}
        for _field in fields(self):
            field_value = getattr(self, _field.name)
            if field_value is not None:
                values[_field.name] = field_value

        return values


@dataclass(frozen=False)
class SamplingParams:
    """Request specific sampling parameters that are only known at run time."""

    top_k: int = -1
    """Limits the sampling to the K most probable tokens. This defaults to -1 (which evaluates the top 255 tokens).
    For greedy sampling, set to 1."""

    top_p: float = 1
    """Only use the tokens whose cumulative probability is within the top_p threshold. This applies to the top_k tokens."""

    min_p: float = 0.0
    """Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this."""

    temperature: float = 1
    """Controls the randomness of the model's output; higher values produce more diverse responses.
    For greedy sampling, set to temperature to 0."""

    frequency_penalty: float = 0.0
    """The frequency penalty to apply to the model's output. A positive value will penalize new tokens
    based on their frequency in the generated text: tokens will receive a penalty proportional to the
    count of appearances."""

    presence_penalty: float = 0.0
    """The presence penalty to apply to the model's output. A positive value will penalize new tokens
    that have already appeared in the generated text at least once by applying a constant penalty."""

    repetition_penalty: float = 1.0
    """The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens
    that have already appeared in the generated text at least once by dividing the logits by the
    repetition penalty."""

    max_new_tokens: int | None = None
    """The maximum number of new tokens to generate in the response.

    When set to an integer value, generation will stop after this many tokens.
    When None (default), the model may generate tokens until it reaches its
    internal limits or other stopping criteria are met.
    """

    min_new_tokens: int = 0
    """The minimum number of tokens to generate in the response."""

    ignore_eos: bool = False
    """If True, the response will ignore the EOS token, and continue to
    generate until the max tokens or a stop string is hit."""

    stop: list[str] | None = None
    """A list of detokenized sequences that can be used as stop criteria when generating a new sequence."""

    stop_token_ids: list[int] | None = None
    """A list of token ids that are used as stopping criteria when generating a new sequence."""

    detokenize: bool = True
    """Whether to detokenize the output tokens into text."""

    seed: int = field(default_factory=lambda: secrets.randbits(32))
    """The seed to use for the random number generator. Defaults to a cryptographically secure random value."""

    logits_processors: Sequence[LogitsProcessor] | None = None
    """Callables to post-process the model logits.
    See :obj:`~max.interfaces.logit_processors_type.LogitsProcessor` for examples."""

    @classmethod
    def from_input_and_generation_config(
        cls,
        input_params: SamplingParamsInput,
        sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
    ) -> SamplingParams:
        """Create SamplingParams with defaults from HuggingFace's GenerationConfig.

        This method creates a SamplingParams instance by combining three sources of values,
        in priority order (highest to lowest):
        1. User-provided values in input_params (non-None)
        2. Model's GenerationConfig values (only if explicitly set in the model's config)
        3. SamplingParams class defaults

        Args:
            input_params: Dataclass containing user-specified parameter values.
                Values of None will be replaced with model defaults or class defaults.
            sampling_params_defaults: SamplingParamsGenerationConfigDefaults containing
                default sampling parameters extracted from the model's GenerationConfig.

        Returns:
            A new SamplingParams instance with model-aware defaults.

        Example:
            >>> sampling_defaults = model_config.sampling_params_defaults
            >>> params = SamplingParams.from_input_and_generation_config(
            ...     SamplingParamsInput(temperature=0.7),  # User override
            ...     sampling_params_defaults=sampling_defaults
            ... )
        """
        # Start with model's generation config values (only if explicitly set)
        defaults: dict[str, Any] = sampling_params_defaults.values_to_update

        # Handle special mappings from GenerationConfig
        if "do_sample" in defaults and defaults["do_sample"] is not None:
            # If do_sample is False, set greedy defaults (unless user overrides)
            if not defaults["do_sample"]:
                defaults["temperature"] = 0
                defaults["top_k"] = 0

            # This isnt included in SamplingParams, therefore we should remove it.
            del defaults["do_sample"]

        # Overlay user-provided values (highest priority)
        for _field in fields(input_params):
            value = getattr(input_params, _field.name)
            if value is not None:
                defaults[_field.name] = value

        return cls(**defaults)

    def log_sampling_info(self) -> None:
        """Log comprehensive sampling parameters information.

        Displays all sampling parameters in a consistent visual format similar to
        pipeline configuration logging.
        """
        logger.info("Sampling Config")
        logger.info("=" * 60)

        # Core sampling parameters
        logger.info(f"    top_k                  : {self.top_k}")
        logger.info(f"    top_p                  : {self.top_p}")
        logger.info(f"    min_p                  : {self.min_p}")
        logger.info(f"    temperature            : {self.temperature}")

        # Penalty parameters
        logger.info(f"    frequency_penalty      : {self.frequency_penalty}")
        logger.info(f"    presence_penalty       : {self.presence_penalty}")
        logger.info(f"    repetition_penalty     : {self.repetition_penalty}")

        # Generation control parameters
        logger.info(f"    max_new_tokens         : {self.max_new_tokens}")
        logger.info(f"    min_new_tokens         : {self.min_new_tokens}")
        logger.info(f"    ignore_eos             : {self.ignore_eos}")
        logger.info(f"    detokenize             : {self.detokenize}")

        # Stopping criteria
        if self.stop:
            stop_str = ", ".join(f'"{s}"' for s in self.stop)
            logger.info(f"    stop_strings           : [{stop_str}]")
        else:
            logger.info("    stop_strings           : None")

        if self.stop_token_ids:
            stop_ids_str = ", ".join(str(id) for id in self.stop_token_ids)
            logger.info(f"    stop_token_ids         : [{stop_ids_str}]")
        else:
            logger.info("    stop_token_ids         : None")
        logger.info("")

    def __post_init__(self):
        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")

        if self.min_p != 0.0 and self.top_k != 1:
            raise ValueError(
                "We currently do not handle explicit min_p and top_k at the same time."
            )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0.")

        # TODO(E2EOPT-315) -- this is a temporary band-aid, we will add support for top_k = -1 in the future.
        if self.top_k in [0, -1]:
            self.top_k = 255
        elif self.top_k < -1 or self.top_k > 255:
            raise ValueError(
                f"top_k must be -1 or greater than 0 and less than or equal to 255, was {self.top_k}."
            )

        if self.temperature == 0:
            # Set top_k to 1 to ensure greedy sampling.
            self.top_k = 1


@runtime_checkable
class BaseContext(Protocol):
    """Core interface for request lifecycle management across all of MAX, including serving, scheduling, and pipelines.

    This protocol is intended to provide a unified, minimal contract for request state and status handling throughout the MAX stack.
    Over time, `BaseContext` is expected to supersede and replace `InputContext` as the canonical context interface, as we refactor and standardize context handling across the codebase.
    """

    @property
    def request_id(self) -> RequestID:
        """Unique identifier for the request."""
        ...

    @property
    def status(self) -> GenerationStatus:
        """Current generation status of the request."""
        ...

    @status.setter
    def status(self, status: GenerationStatus) -> None:
        """Update the generation status of the request."""
        ...

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


BaseContextType = TypeVar("BaseContextType", bound=BaseContext)


@runtime_checkable
class InputContext(BaseContext, Protocol):
    """Protocol defining the interface for model input contexts in token generation.

    An ``InputContext`` represents model inputs for ``TokenGenerator`` instances, managing
    the state of tokens throughout the generation process. It handles token arrays,
    generation status, sampling parameters, and various indices that track different
    stages of token processing.

    The context maintains a token array with the following layout::

        .                      +---------- full prompt ----------+   CHUNK_SIZE*N v
        . +--------------------+---------------+-----------------+----------------+
        . |     completed      |  next_tokens  |                 |  preallocated  |
        . +--------------------+---------------+-----------------+----------------+
        .            start_idx ^    active_idx ^         end_idx ^

    Token Array Regions:
        - completed: Tokens that have already been processed and encoded.
        - next_tokens: Tokens that will be processed in the next iteration.
          This may be a subset of the full prompt due to chunked prefill.
        - preallocated: Token slots that have been preallocated. The token array
          resizes to multiples of ``CHUNK_SIZE`` to accommodate new tokens.

    Key Indices:
        - ``start_idx``: Marks the beginning of completed tokens
        - ``active_idx``: Marks the start of next_tokens within the array
        - ``end_idx``: Marks the end of all active tokens (one past the last token)
    """

    @property
    def request_id(self) -> RequestID:
        """The unique identifier for this generation request.

        Returns:
            A ``RequestID`` that uniquely identifies this request across the system.
        """
        ...

    @property
    def status(self) -> GenerationStatus:
        """The current generation status of this context.

        Returns:
            The ``GenerationStatus`` indicating the current state of generation
            (e.g., encoding, generating, completed, or error).
        """
        ...

    @status.setter
    def status(self, status: GenerationStatus) -> None:
        """Update the current generation status of this context.

        This method transitions the context to a new generation state, such as
        moving from encoding to generating or marking completion.

        Args:
            status: The new ``GenerationStatus`` to assign to this context.
        """
        ...

    @property
    def is_done(self) -> bool:
        """Whether the generation process for this context has completed.

        Returns:
            ``True`` if generation has finished successfully or been terminated,
            ``False`` if generation is still in progress.
        """
        ...

    @property
    def eos_token_ids(self) -> set[int]:
        """The set of end-of-sequence token IDs that can terminate generation.

        Returns:
            A set of token IDs that, when generated, will signal the end of the
            sequence and terminate the generation process.
        """
        ...

    @property
    def active_idx(self) -> int:
        """The index marking the start of ``next_tokens`` within the token array.

        This index separates completed tokens from tokens that will be processed
        in the next iteration during chunked prefill or generation.

        Returns:
            The zero-based index where ``next_tokens`` begin in the token array.
        """
        ...

    @property
    def start_idx(self) -> int:
        """The index marking the start of completed tokens in the token array.

        Completed tokens are those that have already been processed and encoded
        by the model in previous iterations.

        Returns:
            The zero-based index where completed tokens begin in the token array.
        """
        ...

    @property
    def end_idx(self) -> int:
        """The index marking the end of all active tokens in the token array.

        This is an exclusive end index (one past the last active token), following
        Python's standard slicing conventions.

        Returns:
            The zero-based index one position past the last active token.
        """
        ...

    @property
    def current_length(self) -> int:
        """The current total length of the sequence.

        This includes both completed tokens and tokens currently being processed,
        representing the total number of tokens in the active sequence.

        Returns:
            The total number of tokens including completed and active tokens.
        """
        ...

    @property
    def max_length(self) -> int | None:
        """The maximum allowed length for this sequence.

        When set, generation will stop when this length is reached, regardless
        of other stopping criteria.

        Returns:
            The maximum sequence length limit, or None if no limit is set.
        """
        ...

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens that must be generated.

        Generation will continue until at least this many new tokens have been
        produced, even if other stopping criteria are met (e.g., EOS tokens).

        Returns:
            The minimum number of new tokens to generate.
        """
        ...

    @property
    def log_probabilities(self) -> int:
        """The number of top tokens to return log probabilities for.

        When greater than 0, the system returns log probabilities for the top N
        most likely tokens at each generation step.

        Returns:
            The number of top tokens to include in log probability output.
            Returns 0 if log probabilities are disabled.
        """
        ...

    @property
    def log_probabilities_echo(self) -> bool:
        """Whether to include input tokens in the returned log probabilities.

        When ``True``, log probabilities will be computed and returned for input
        (prompt) tokens in addition to generated tokens.

        Returns:
            ``True`` if input tokens should be included in log probability output,
            ``False`` otherwise.
        """
        ...

    @property
    def active_length(self) -> int:
        """The number of tokens being processed in the current iteration.

        During context encoding (prompt processing), this equals the prompt size
        or chunk size for chunked prefill. During token generation, this is
        typically 1 (one new token per iteration).

        Returns:
            The number of tokens being processed in this iteration.
        """
        ...

    @property
    def next_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """The tokens to be processed in the next model iteration.

        This array contains the tokens that will be fed to the model in the
        upcoming forward pass. The length should match ``active_length``.

        Returns:
            A 1D NumPy array of int32 token IDs with length equal to ``active_length``.
        """
        ...

    @property
    def tokens(self) -> npt.NDArray[np.integer[Any]]:
        """The complete token array including preallocated slots.

        This includes all tokens (completed, active, and preallocated empty slots).
        For most use cases, prefer ``all_tokens`` to get only the active tokens.

        Returns:
            A 1D NumPy array of int32 values containing all tokens including padding.
        """
        ...

    @property
    def all_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """All active tokens in the context (prompt and generated).

        This property returns only the meaningful tokens, excluding any
        preallocated but unused slots in the token array.

        Returns:
            A 1D NumPy array of int32 values containing all prompt and generated tokens.
        """
        return self.tokens[: self.end_idx]

    @property
    def prompt_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """The original prompt tokens for this context.

        These are the input tokens that were provided to start the generation
        process, before any tokens were generated by the model.

        Returns:
            A 1D NumPy array of int32 values containing the original prompt token IDs.
        """
        ...

    @property
    def generated_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """All tokens generated by the model for this context.

        This excludes the original prompt tokens and includes only tokens
        that have been produced during the generation process.

        Returns:
            A 1D NumPy array of int32 values containing generated token IDs.
        """
        ...

    @property
    def last_generated_token(self) -> int:
        """Returns the most recently generated token. If no tokens have been generated, raises an error.

        Returns:
            int: The most recently generated token.
        """
        ...

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Get token indices that should be masked in the output logits.

        This method is primarily used to implement the ``min_tokens`` constraint,
        where certain tokens (typically EOS tokens) are masked to prevent early
        termination before the minimum token count is reached.

        Args:
            num_steps: The number of generation steps to compute masks for.

        Returns:
            A list of NumPy arrays, where each array contains token indices
            that should be masked (set to negative infinity) in the logits
            for the corresponding generation step.
        """
        ...

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Update the context with a newly generated token.

        This method adds a generated token to the context, updating the token
        array and associated metadata. It also stores log probability information
        if provided.

        Args:
            new_token: The token ID to add to the generation sequence.
            log_probabilities: Optional log probability data for the new token
                and alternatives. Used for analysis and debugging.
        """
        ...

    def jump_ahead(self, new_token: int) -> None:
        """Jump ahead in generation by adding a token and updating indices.

        This method is used in speculative decoding scenarios to quickly
        advance the generation state when draft tokens are accepted.

        Args:
            new_token: The token ID to add when jumping ahead in the sequence.
        """
        ...

    def bump_token_indices(
        self,
        start_idx: int = 0,
        active_idx: int = 0,
        end_idx: int = 0,
    ) -> None:
        """Increment token indices by the specified amounts.

        This method provides fine-grained control over token index management,
        allowing incremental updates to track token processing progress.

        Args:
            start_idx: Amount to increment the ``start_idx`` by.
            active_idx: Amount to increment the ``active_idx`` by.
            end_idx: Amount to increment the ``end_idx`` by.
        """
        ...

    def set_token_indices(
        self,
        start_idx: int | None = None,
        active_idx: int | None = None,
        end_idx: int | None = None,
    ) -> None:
        """Set token indices to specific absolute values.

        This method provides direct control over token index positioning,
        allowing precise management of the token array state.

        Args:
            start_idx: New absolute value for ``start_idx``, if provided.
            active_idx: New absolute value for ``active_idx``, if provided.
            end_idx: New absolute value for ``end_idx``, if provided.
        """
        ...

    @property
    def matcher(self) -> Any | None:
        """The grammar matcher for structured output generation, if configured.

        The matcher enforces structural constraints (like JSON schema) during
        generation to ensure valid formatted output.

        Returns:
            The grammar matcher instance, or None if no structured generation
            is configured for this context.

        Note:
            The matcher type depends on the structured generation backend used
            (e.g., outlines, guidance, etc.). In the future, this should be
            replaced with a Protocol for better type safety.
        """
        ...

    @property
    def json_schema(self) -> str | None:
        """The JSON schema for constrained decoding, if configured.

        When set, this schema constrains token generation to produce valid JSON
        output that conforms to the specified structure.

        Returns:
            The JSON schema string, or None if no schema constraint is active.
        """
        ...

    def set_matcher(self, matcher: Any) -> None:
        """Set a grammar matcher for constrained decoding.

        This method configures structured output generation by installing a
        grammar matcher that enforces format constraints during token generation.

        Args:
            matcher: The grammar matcher instance to use for constraining output.
                The specific type depends on the structured generation backend.
        """
        ...

    def reset(self) -> None:
        """Reset the context state by consolidating all tokens into a new prompt.

        This method is typically used when a request is evicted from cache,
        requiring the context to be re-encoded in a subsequent context encoding
        iteration. All generated tokens become part of the new prompt.
        """
        ...

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Compute the maximum number of generation steps available.

        This method calculates how many tokens can be generated without
        exceeding the specified maximum sequence length limit.

        Args:
            max_seq_len: The maximum allowed sequence length for this context.

        Returns:
            The number of generation steps that can be executed before reaching
            the sequence length limit.
        """
        ...

    @property
    def needs_ce(self) -> bool:
        """Returns whether this context needs context encoding (CE).

        CE mode indicates that the context has additional prompt tokens to encode.

        Returns:
            bool: True if the context needs CE, False otherwise.
        """
        ...

    @property
    def is_initial_prompt(self) -> bool:
        """Whether this context contains only the initial prompt.

        This property indicates if the context has not yet been updated with
        any generated tokens and still contains only the original input.

        Returns:
            ``True`` if no tokens have been generated yet, ``False`` if generation
            has begun and tokens have been added.
        """
        ...

    @property
    def sampling_params(self) -> SamplingParams:
        """The sampling parameters configured for this generation request.

        These parameters control how tokens are selected during generation,
        including temperature, top-k/top-p filtering, and stopping criteria.

        Returns:
            The ``SamplingParams`` instance containing all sampling configuration
            for this context.
        """
        ...


class ImageMetadata(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """Metadata about an image in the prompt.

    Each image corresponds to a range in the text token array [start_idx, end_idx).
    """

    start_idx: int
    """Index of the first <vision_token_id> special token for the image"""

    end_idx: int
    """One after the index of the last <vision_token_id> special token for the image"""

    pixel_values: npt.NDArray[np.floating[Any]]
    """Pixel values for the image"""

    def __post_init__(self) -> None:
        if self.start_idx < 0:
            raise ValueError("Images must have a valid start index")
        if self.end_idx <= self.start_idx:
            raise ValueError(
                "Images must have a valid start and end index containing at least one <vision_token_id>"
            )

    def __repr__(self):
        return f"ImageMetadata(start_idx={self.start_idx}, end_idx={self.end_idx}, pixel_values={self.pixel_values.shape})"


@runtime_checkable
class VLMInputContext(InputContext, Protocol):
    """Protocol defining the interface for VLM input contexts."""

    @property
    def image_idx(self) -> int:
        """Index of the next unencoded image in the prompt."""
        ...

    @property
    def images(self) -> list[ImageMetadata]:
        """Returns the images in the context."""
        ...

    @property
    def next_images(self) -> list[ImageMetadata]:
        """Returns the images that are not yet encoded."""
        ...

    @property
    def needs_vision_encoding(self) -> bool:
        """Returns whether vision encoding is needed for this context."""
        ...

    def compute_image_aligned_idx(self, idx: int) -> int:
        """Possibly aligns a index value downward if it lies in the middle of an image."""
        ...
