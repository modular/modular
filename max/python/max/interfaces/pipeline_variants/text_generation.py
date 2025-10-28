# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces.context import BaseContext, SamplingParams
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import Request, RequestID
from max.interfaces.status import GenerationStatus


class TextGenerationRequestFunction(TypedDict):
    """
    Represents a function definition for a text generation request.
    """

    name: str
    """The name of the function to be invoked."""

    description: str | None
    """A human-readable description of the function's purpose."""

    parameters: dict[str, Any]
    """A dictionary describing the function's parameters, typically following a JSON schema."""


class TextGenerationRequestTool(TypedDict):
    """
    Represents a tool definition for a text generation request.
    """

    type: str
    """The type of the tool, typically indicating the tool's category or usage."""

    function: TextGenerationRequestFunction
    """The function definition associated with the tool, including its name, description, and parameters."""


class TextGenerationResponseFormat(TypedDict):
    """
    Represents the response format specification for a text generation request.
    """

    type: str
    """The type of response format, e.g., "json_object"."""

    json_schema: dict[str, Any]
    """A JSON schema dictionary that defines the structure and validation rules for the generated response."""


class TextGenerationRequestMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]
    """
    The role of the message sender, indicating whether the message is from the system, user, or assistant.
    """

    content: str | list[dict[str, Any]]
    """
    Content can be a simple string or a list of message parts of different modalities.

    For example:

    .. code-block:: json

        {
          "role": "user",
          "content": "What's the weather like in Boston today?"
        }

    Or:

    .. code-block:: json

        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What's in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                  "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
              }
            }
          ]
        }
    """


@dataclass(frozen=True)
class TextGenerationRequest(Request):
    model_name: str = field()
    """
    The name of the model to be used for generating tokens. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: str | Sequence[int] | None = None
    """
    The prompt to be processed by the model. This field supports legacy
    completion APIs and can accept either a string or a sequence of integers
    representing token IDs. If not provided, the model may generate output
    based on the messages field.
    """
    messages: list[TextGenerationRequestMessage] | None = None
    """
    A list of messages for chat-based interactions. This is used in chat
    completion APIs, where each message represents a turn in the conversation.
    If provided, the model will generate responses based on these messages.
    """
    images: list[bytes] | None = None
    """
    A list of image byte arrays that can be included as part of the request.
    This field is optional and may be used for multimodal inputs where images
    are relevant to the prompt or task.
    """
    tools: list[TextGenerationRequestTool] | None = None
    """
    A list of tools that can be invoked during the generation process. This
    allows the model to utilize external functionalities or APIs to enhance its
    responses.
    """
    response_format: TextGenerationResponseFormat | None = None
    """
    Specifies the desired format for the model's output. When set, it enables
    structured generation, which adheres to the json_schema provided.
    """
    timestamp_ns: int = 0
    """
    The time (in nanoseconds) when the request was received by the server. This
    can be useful for performance monitoring and logging purposes.
    """
    request_path: str = "/"
    """
    The endpoint path for the request. This is typically used for routing and
    logging requests within the server infrastructure.
    """
    logprobs: int = 0
    """
    The number of top log probabilities to return for each generated token. A value
    of 0 means that log probabilities will not be returned. Useful for analyzing
    model confidence in its predictions.
    """
    echo: bool = False
    """
    If set to True, the response will include the original prompt along with the
    generated output. This can be useful for debugging or when you want to see how
    the input relates to the output.
    """
    stop: str | list[str] | None = None
    """
    Optional list of stop expressions (see: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)
    """
    chat_template_options: dict[str, Any] | None = None
    """
    Optional dictionary of options to pass when applying the chat template.
    """

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    """Token sampling configuration parameters for the request."""

    target_endpoint: str | None = None
    """
    Optional target endpoint identifier for routing the request to a specific
    service or model instance. This should be used in disaggregate serving
    scenarios, when you want to dynamically route to a specific instance.
    If not specified, the request will be routed to the default endpoint.
    """


def _check_text_generation_output_implements_pipeline_output(
    x: TextGenerationOutput,
) -> PipelineOutput:
    return x


class TextGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents the output of a text generation operation, combining token IDs,
    final generation status, request ID, and optional log probabilities for each token.
    """

    request_id: RequestID
    """The unique identifier for the generation request."""

    tokens: list[int]
    """List of generated token IDs."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    log_probabilities: list[LogProbabilities] | None = None
    """Optional list of log probabilities for each token."""

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the text generation process is complete.

        Returns:
            bool: True if the generation is done, False otherwise.
        """
        return self.final_status.is_done


@runtime_checkable
class TextGenerationContext(BaseContext, Protocol):
    """Protocol defining the interface for text generation contexts in token generation.

    A ``TextGenerationContext`` represents model inputs for text generation pipelines, managing
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
            The maximum sequence length limit, or ``None`` if no limit is set.
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
    def next_tokens(self) -> npt.NDArray[Any]:
        """The tokens to be processed in the next model iteration.

        This array contains the tokens that will be fed to the model in the
        upcoming forward pass. The length should match ``active_length``.

        Returns:
            A 1D NumPy array of int32 token IDs with length equal to ``active_length``.
        """
        ...

    @property
    def tokens(self) -> npt.NDArray[Any]:
        """The complete token array including preallocated slots.

        This includes all tokens (completed, active, and preallocated empty slots).
        For most use cases, prefer ``all_tokens`` to get only the active tokens.

        Returns:
            A 1D NumPy array of int32 values containing all tokens including padding.
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

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt.
        This method is used when a request is evicted, meaning that the context
        needed to be re-encoded in the following CE iteration."""
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
    def all_tokens(self) -> npt.NDArray[Any]:
        """All active tokens in the context (prompt and generated).

        This property returns only the meaningful tokens, excluding any
        preallocated but unused slots in the token array.

        Returns:
            A 1D NumPy array of int32 values containing all prompt and generated tokens.
        """
        return self.tokens[: self.end_idx]

    @property
    def prompt_tokens(self) -> npt.NDArray[Any]:
        """The original prompt tokens for this context.

        These are the input tokens that were provided to start the generation
        process, before any tokens were generated by the model.

        Returns:
            A 1D NumPy array of int32 values containing the original prompt token IDs.
        """
        ...

    @property
    def generated_tokens(self) -> npt.NDArray[Any]:
        """All tokens generated by the model for this context.

        This excludes the original prompt tokens and includes only tokens
        that have been produced during the generation process.

        Returns:
            A 1D NumPy array of int32 values containing generated token IDs.
        """
        ...

    def get_last_generated_token(self) -> int:
        """The most recently generated token.

        This property returns the token ID of the most recent token that was
        generated by the model during the generation process. If no tokens
        have been generated yet, this method will raise an error.

        This is not a @property method since it can raise.

        Returns:
            The token ID of the most recently generated token.

        Raises:
            ValueError: If no tokens have been generated yet.
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
        """Update the context with a newly generated token, and update status.

        This method adds a generated token to the context, updating the token
        array, associated metadata, and log probabilities (if provided).
        It is also responsible for updating the context's generation status and
        determining if the generation sequence is complete, either due to reaching
        an end-of-sequence condition or meeting stopping criteria.

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
            The JSON schema string, or ``None`` if no schema constraint is active.
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

    @property
    def needs_ce(self) -> bool:
        """Returns whether this context needs context encoding (CE).

        CE mode indicates that the context has additional prompt tokens to encode.

        Returns:
            bool: True if the context needs CE, False otherwise.
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

    def to_generation_output(self) -> TextGenerationOutput:
        """
        Convert this context to a TextGenerationOutput object.

        This property provides a standardized way to extract the final output
        of the text generation process from the context, including generated
        text, tokens, and any associated metadata.

        Returns:
            TextGenerationOutput: The output object containing the results of
            the text generation for this context.
        """
        ...


TextGenerationContextType = TypeVar(
    "TextGenerationContextType", bound=TextGenerationContext
)
"""Type variable for text generation context types, constrained to BaseContext.

This allows generic typing of text generation pipeline components to accept any
context type that implements the BaseContext protocol.
"""


class BatchType(Enum):
    """Type of batch."""

    CE = "CE"
    """Context encoding batch."""
    TG = "TG"
    """Token generation batch."""


@dataclass(eq=True)
class TextGenerationInputs(PipelineInputs, Generic[TextGenerationContextType]):
    """
    Input parameters for text generation pipeline operations.

    This class encapsulates the batch of contexts and number of steps required
    for token generation in a single input object, replacing the previous
    pattern of passing batch and num_steps as separate parameters.
    """

    batches: list[dict[RequestID, TextGenerationContextType]]
    """Variable list of batches, with each batch being a dictionary mapping
    request IDs to context objects.

    There can be multiple batches when using data parallelism, in which each
    batch is mapped to a different device.
    """

    num_steps: int
    """Number of steps to run for."""

    input_tokens: int = -1
    """Number of input tokens."""

    batch_type: BatchType = BatchType.TG
    """Type of batch."""

    def __post_init__(self) -> None:
        self.input_tokens = sum(
            ctx.active_length for ctx in self.batch.values()
        )
        self.batch_type = BatchType.TG
        for req in self.batch.values():
            if req.needs_ce:
                self.batch_type = BatchType.CE
                break

    @property
    def batch(self) -> dict[RequestID, TextGenerationContextType]:
        """Returns merged batches."""
        return {k: v for batch in self.batches for k, v in batch.items()}

    def __bool__(self) -> bool:
        return len(self.batch) > 0

    def __repr__(self) -> str:
        return f"TextGenerationInputs(batch_size={len(self.batch)}, num_steps={self.num_steps})"

    @property
    def enable_echo(self) -> bool:
        """Return True if any context in the batch has echo enabled."""
        return any(self.batch_echo)

    @property
    def enable_log_probs(self) -> bool:
        """Return True if any context in the batch requests log probabilities."""
        return any(self.batch_top_log_probs)

    @cached_property
    def batch_top_log_probs(self) -> list[int]:
        """List of requested top log probabilities per context in the batch."""
        return [ctx.log_probabilities for ctx in self.batch.values()]

    @cached_property
    def batch_echo(self) -> list[bool]:
        """List indicating whether echo is enabled for each context in the batch."""
        return [ctx.log_probabilities_echo for ctx in self.batch.values()]


def hash_image(pixel_values: npt.NDArray[np.floating[Any]]) -> int:
    """Compute the hash of an image."""
    return hash(pixel_values.data.tobytes())


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

    image_hash: int = -1
    """Hash of the image, for use in prefix caching"""

    def __post_init__(self) -> None:
        if self.start_idx < 0:
            raise ValueError("Images must have a valid start index")
        if self.end_idx <= self.start_idx:
            raise ValueError(
                "Images must have a valid start and end index containing at least one <vision_token_id>"
            )

        # Compute the hash of the image in post init, overriding the default value of -1
        self.image_hash = hash_image(self.pixel_values)

    def __repr__(self):
        return f"ImageMetadata(start_idx={self.start_idx}, end_idx={self.end_idx}, pixel_values={self.pixel_values.shape})"


@runtime_checkable
class VLMTextGenerationContext(TextGenerationContext, Protocol):
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
