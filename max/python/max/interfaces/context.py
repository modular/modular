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
from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, TypeVar, runtime_checkable

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

    # Do not cache this property, callers mutate the returned dict.
    @property
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
    """Limits the sampling to the K most probable tokens. This defaults to -1 (to sample all tokens), for greedy sampling set to 1."""

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
                defaults["top_k"] = 1

            # This isn't included in SamplingParams, therefore we should remove it.
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
        # Normalize None values to field defaults. These can arrive when a
        # user explicitly sends top_p=null or top_k=null in a request.
        # Treating None as "use the default" keeps the fast gumbel-sampling
        # path reachable and avoids NaN propagation through numpy.
        _defaults = self.__dataclass_fields__
        if self.top_k is None:
            self.top_k = _defaults["top_k"].default
        if self.top_p is None:
            self.top_p = _defaults["top_p"].default

        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")

        if self.min_p != 0.0 and self.top_k != 1:
            raise ValueError(
                "We currently do not handle explicit min_p and top_k at the same time."
            )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0.")

        if self.top_k == 0:
            self.top_k = -1
        if self.top_k < -1 or self.top_k > 255:
            raise ValueError(
                f"top_k must be -1 or greater than 0 and less than or equal to 255, was {self.top_k}."
            )

        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], was {self.top_p}.")

        # If temperature is 0, set top_k to 1.
        if self.temperature == 0:
            logger.debug("Temperature is 0, overriding top_k to 1.")
            self.top_k = 1


@runtime_checkable
class BaseContext(Protocol):
    """Core interface for request lifecycle management across all of MAX, including serving, scheduling, and pipelines.

    This protocol is intended to provide a unified, minimal contract for request state and status handling throughout the MAX stack.
    Each pipeline variant (e.g., text generation, embeddings, image generation) is expected to extend this interface by creating
    their own modality-specific context classes that implement this protocol and add additional functionality relevant to their
    particular use case.

    The minimal interface ensures that all context types can be handled uniformly by the scheduling and serving infrastructure,
    while allowing pipeline-specific implementations to add their own state management, input validation, and result handling.
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
"""
Type variable for generic programming with BaseContext implementations.

This TypeVar is bound to BaseContext, meaning it can represent any type that implements
the BaseContext protocol. It enables type-safe generic functions and classes that work
with any BaseContext subtype while preserving the specific type information through
the type system.

Example usage:
    def process_context(context: BaseContextType) -> BaseContextType:
        # Function that accepts any BaseContext implementation
        # and returns the same type
        ...
"""
