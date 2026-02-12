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

import numpy as np
import numpy.typing as npt
from max.interfaces.context import BaseContext, SamplingParams
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import Request, RequestID
from max.interfaces.status import GenerationStatus
from max.interfaces.tokens import TokenBuffer
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextGenerationRequestFunction(TypedDict):
    """Represents a function definition for a text generation request."""

    name: str
    """The name of the function to be invoked."""

    description: str | None
    """A human-readable description of the function's purpose."""

    parameters: dict[str, Any]
    """A dictionary describing the function's parameters, typically following a JSON schema."""


class TextGenerationRequestTool(TypedDict):
    """Represents a tool definition for a text generation request."""

    type: str
    """The type of the tool, typically indicating the tool's category or usage."""

    function: TextGenerationRequestFunction
    """The function definition associated with the tool, including its name, description, and parameters."""


class TextGenerationResponseFormat(TypedDict):
    """Represents the response format specification for a text generation request."""

    type: str
    """The type of response format, e.g., "json_object"."""

    json_schema: dict[str, Any]
    """A JSON schema dictionary that defines the structure and validation rules for the generated response."""


class ContentPart(BaseModel):
    type: Literal["text", "image"]


class MessageContentPart(BaseModel):
    type: str = Field(..., description="Content type identifier")
    model_config = ConfigDict(frozen=True)


class TextContentPart(MessageContentPart):
    type: Literal["text"] = Field(
        default="text", description="Content type identifier"
    )
    text: str = Field(..., description="Text text content")


class ImageContentPart(MessageContentPart):
    type: Literal["image"] = Field(
        default="image", description="Content type identifier"
    )


MessageContent = TextContentPart | ImageContentPart

MessageRole = Literal["system", "user", "assistant", "tool", "function"]


class TextGenerationRequestMessage(BaseModel):
    role: MessageRole = Field(
        ..., description="Text role of the message sender"
    )

    content: str | list[MessageContent]
    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_format(cls, v: Any) -> str | list[MessageContent]:
        """Normalizes message content to a string or list of content parts."""
        if isinstance(v, str):
            return v

        if not isinstance(v, list):
            raise ValueError(
                f"Invalid content format: {type(v).__name__}. "
                "Expected str or list of content parts."
            )

        normalized: list[MessageContent] = []
        for item in v:
            if isinstance(item, (TextContentPart, ImageContentPart)):
                normalized.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid content part type: {type(item).__name__}. "
                    "Expected dict or MessageContentPart instance."
                )

            if "type" not in item:
                raise ValueError(
                    f"Malformed message content part: missing 'type' field. Got: {item}"
                )

            content_type = item["type"]

            if content_type == "text":
                text_value = item.get("text") or item.get("content", "")
                normalized.append(TextContentPart(text=text_value))
            elif content_type == "image":
                normalized.append(ImageContentPart())
            elif content_type == "image_url":
                raise ValueError(
                    "image_url content type not supported in internal format. "
                    "Images must be provided as bytes in TextGenerationRequest.images "
                    "with image placeholders (type='image') in message content."
                )
            else:
                raise ValueError(
                    f"Unsupported message content type: '{content_type}'"
                )

        return normalized

    def flatten_content(self) -> dict[str, str]:
        """Flattens message content to a single role/content dict for text-only messages."""
        if isinstance(self.content, str):
            return {"role": str(self.role), "content": self.content}

        content_str = ""
        for content in self.content:
            if isinstance(content, TextContentPart):
                if content_str != "":
                    content_str += "\n"

                content_str += content.text

            else:
                raise ValueError("only text content can be flattened.")

        return {"role": str(self.role), "content": content_str}

    @cached_property
    def number_of_images(self) -> int:
        """Returns the number of ImageContentPart instances in the message content."""
        if isinstance(self.content, str):
            return 0
        return sum(
            1 for item in self.content if isinstance(item, ImageContentPart)
        )


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
    messages: list[TextGenerationRequestMessage] = field(default_factory=list)
    """
    A list of messages for chat-based interactions. This is used in chat
    completion APIs, where each message represents a turn in the conversation.
    If provided, the model will generate responses based on these messages.
    """
    images: list[bytes] = field(default_factory=list)
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

    def __post_init__(self) -> None:
        """Validates mutual exclusivity, image-messaging constraints, and message-image consistency after object initialization."""
        # Convert dict messages to TextGenerationRequestMessage objects
        if self.messages is not None:
            converted_messages: list[TextGenerationRequestMessage] = []
            for msg in self.messages:
                if isinstance(msg, dict):
                    converted_messages.append(
                        TextGenerationRequestMessage(**msg)
                    )
                elif isinstance(msg, TextGenerationRequestMessage):
                    converted_messages.append(msg)
                else:
                    raise TypeError(f"Invalid message type: {type(msg)}")
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, "messages", converted_messages)

        if self.prompt and self.messages:
            raise ValueError(
                "both prompt and messages cannot be provided to TextGenerationRequest"
            )

        if self.images and isinstance(self.prompt, str):
            raise ValueError(
                "string prompts cannot be provided, when images are provided, use messages"
            )

        if self.images and self.number_of_images != len(self.images):
            raise ValueError(
                f"number of images provided in TextGenerationRequest do not match messages:\n{self.messages}"
            )

    @cached_property
    def number_of_images(self) -> int:
        """Returns the total number of image-type contents across all provided messages.

        Returns:
            int: Total count of image-type contents found in messages.
        """
        return (
            sum(message.number_of_images for message in self.messages)
            if self.messages
            else 0
        )


def _check_text_generation_output_implements_pipeline_output(
    x: TextGenerationOutput,
) -> PipelineOutput:
    return x


@dataclass(kw_only=True)
class TextGenerationOutput:
    """Represents the output of a text generation operation.

    Combines token IDs, final generation status, request ID, and optional log
    probabilities for each token.
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
        """Indicates whether the text generation process is complete.

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
    """

    @property
    def tokens(self) -> TokenBuffer:
        """The token buffer for the context."""
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
        """Convert this context to a TextGenerationOutput object.

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
"""Type variable for text generation context types, constrained to TextGenerationContext.

This allows generic typing of text generation pipeline components to accept any
context type that implements the TextGenerationContext protocol.
"""


class BatchType(Enum):
    """Type of batch."""

    CE = "CE"
    """Context encoding batch."""
    TG = "TG"
    """Token generation batch."""


@dataclass(eq=True)
class TextGenerationInputs(PipelineInputs, Generic[TextGenerationContextType]):
    """Input parameters for text generation pipeline operations.

    This class encapsulates the batch of contexts and number of steps required
    for token generation in a single input object, replacing the previous
    pattern of passing batch and num_steps as separate parameters.
    """

    batches: list[list[TextGenerationContextType]]
    """Variable list of batches, with each batch being a list of contexts.

    There can be multiple batches when using data parallelism, in which each
    batch is mapped to a different device replica.
    """

    num_steps: int
    """Number of steps to run for."""

    input_tokens: int = -1
    """Number of input tokens."""

    batch_type: BatchType = BatchType.TG
    """Type of batch."""

    def __post_init__(self) -> None:
        self.input_tokens = sum(
            ctx.tokens.active_length for ctx in self.flat_batch
        )
        self.context_tokens = sum(
            ctx.tokens.processed_length for ctx in self.flat_batch
        )
        self.batch_type = BatchType.TG
        for context in self.flat_batch:
            if context.tokens.generated_length == 0:
                self.batch_type = BatchType.CE
                break

    @property
    def flat_batch(self) -> list[TextGenerationContextType]:
        """Flattened list of contexts across all replicas."""
        return [context for batch in self.batches for context in batch]

    def __bool__(self) -> bool:
        return len(self.flat_batch) > 0

    def __repr__(self) -> str:
        return (
            "TextGenerationInputs("
            f"batch_size={len(self.flat_batch)}, "
            f"num_steps={self.num_steps}"
            ")"
        )

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
        return [ctx.log_probabilities for ctx in self.flat_batch]

    @cached_property
    def batch_echo(self) -> list[bool]:
        """List indicating whether echo is enabled for each context in the batch."""
        return [ctx.log_probabilities_echo for ctx in self.flat_batch]


@dataclass(kw_only=True)
class ImageMetadata:
    """Metadata about an image in the prompt.

    Each image corresponds to a range in the text token array [start_idx, end_idx).
    """

    start_idx: int
    """Index of the first <vision_token_id> special token for the image"""

    end_idx: int
    """One after the index of the last <vision_token_id> special token for the image"""

    pixel_values: npt.NDArray[Any]
    """Pixel values for the image.

    Can be various dtypes depending on the vision model:

    - float32: Original precision
    - uint16: BFloat16 bits stored as uint16 (workaround for NumPy's lack of
      native bfloat16 support). Reinterpreted as bfloat16 on GPU.
    """

    image_hash: int | None = None
    """Hash of the image, for use in prefix caching"""

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
