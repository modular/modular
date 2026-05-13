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

"""Defines request, context, and output types for text generation pipelines, including multimodal and VLM support."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import chain
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
from max.interfaces.eos_tracking import EOSTracker
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import RequestID
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
    """The type of response format, for example, ``json_object`` or ``grammar``."""

    json_schema: dict[str, Any]
    """A JSON schema dictionary that defines the structure and validation rules for the generated response."""

    grammar: str | None
    """Grammar for constrained decoding.

    When set with ``type="grammar"``, this takes precedence over ``json_schema``.
    Used for model-specific constrained decoding formats like Kimi's tool call grammar.
    """


class ContentPart(BaseModel):
    """A single part of a multi-modal message content."""

    type: Literal["text", "image"]


class MessageContentPart(BaseModel):
    """A typed, immutable part of a message's content."""

    type: str = Field(..., description="Content type identifier")
    model_config = ConfigDict(frozen=True)


class TextContentPart(MessageContentPart):
    """A plain-text content part of a message."""

    type: Literal["text"] = Field(
        default="text", description="Content type identifier"
    )
    text: str = Field(..., description="Text text content")


class ImageContentPart(MessageContentPart):
    """An image content part of a message."""

    type: Literal["image"] = Field(
        default="image", description="Content type identifier"
    )


class VideoContentPart(MessageContentPart):
    """A video content part of a message."""

    type: Literal["video"] = Field(
        default="video", description="Content type identifier"
    )


MessageContent = TextContentPart | ImageContentPart | VideoContentPart

MessageRole = Literal["system", "user", "assistant", "tool", "function"]


class TextGenerationRequestMessage(BaseModel):
    """A single message in a text generation request conversation."""

    # Follows the openai spec

    role: MessageRole = Field(
        ..., description="Text role of the message sender"
    )

    content: str | list[MessageContent] = Field(
        default="",
        description=(
            "Message text/multimodal content. Defaults to the empty string for "
            "assistant messages that only carry ``tool_calls``."
        ),
    )

    # Tool call fields from:
    # https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/chat/chat_completion_message_function_tool_call_param.py
    # The KimiK2.5 chat template uses these fields:
    # https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/chat_template.jinja
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool calls emitted by an assistant turn in the conversation "
            "history. Each entry follows the OpenAI shape "
            "``{id, type, function: {name, arguments}}``. Passed through "
            "verbatim to the chat template so multi-turn tool-use prompts "
            "render correctly."
        ),
    )

    tool_call_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the assistant ``tool_calls`` entry that this tool "
            "message is responding to. Required by chat templates that emit a "
            "tool-response header referencing the originating call."
        ),
    )

    reasoning_content: str | None = Field(
        default=None,
        description=(
            "Reasoning/thinking content produced alongside an assistant turn."
        ),
    )

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_format(cls, v: Any) -> str | list[MessageContent]:
        """Normalizes message content to a string or list of content parts."""
        if v is None:
            # OpenAI permits ``content=None`` for assistant messages whose
            # payload is carried entirely by ``tool_calls``; collapse to the
            # empty string so the chat template still has something to render.
            return ""
        if isinstance(v, str):
            return v

        if not isinstance(v, list):
            raise ValueError(
                f"Invalid content format: {type(v).__name__}. "
                "Expected str or list of content parts."
            )

        normalized: list[MessageContent] = []
        for item in v:
            if isinstance(
                item, (TextContentPart, ImageContentPart, VideoContentPart)
            ):
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
            elif content_type == "video":
                normalized.append(VideoContentPart())
            elif content_type == "image_url":
                raise ValueError(
                    "image_url content type not supported in internal format. "
                    "Images must be provided as bytes in TextGenerationRequest.images "
                    "with image placeholders (type='image') in message content."
                )
            elif content_type == "video_url":
                raise ValueError(
                    "video_url content type not supported in internal format. "
                    "Videos must be provided as bytes in TextGenerationRequest.videos "
                    "with video placeholders (type='video') in message content."
                )
            else:
                raise ValueError(
                    f"Unsupported message content type: '{content_type}'"
                )

        return normalized

    def flatten_content(self) -> dict[str, Any]:
        """Flattens message content to a role/content dict for text-only messages.

        Preserves OpenAI-style tool-calling metadata (``tool_calls``,
        ``tool_call_id``, ``reasoning_content``) when set so chat templates
        that consume conversation history with tool use receive a faithful
        representation of each turn.
        """
        if isinstance(self.content, str):
            content_str = self.content
        else:
            parts: list[str] = []
            for content in self.content:
                if isinstance(content, TextContentPart):
                    parts.append(content.text)
                else:
                    raise ValueError("only text content can be flattened.")
            content_str = "\n".join(parts)

        flattened: dict[str, Any] = {
            "role": str(self.role),
            "content": content_str,
        }
        if self.tool_calls is not None:
            flattened["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            flattened["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            flattened["reasoning_content"] = self.reasoning_content
        return flattened

    @cached_property
    def number_of_images(self) -> int:
        """Returns the number of ImageContentPart instances in the message content."""
        if isinstance(self.content, str):
            return 0
        return sum(
            1 for item in self.content if isinstance(item, ImageContentPart)
        )

    @cached_property
    def number_of_videos(self) -> int:
        """Returns the number of VideoContentPart instances in the message content."""
        if isinstance(self.content, str):
            return 0
        return sum(
            1 for item in self.content if isinstance(item, VideoContentPart)
        )


@dataclass(frozen=True)
class TextGenerationRequest:
    """An immutable request for text token generation from a pipeline."""

    request_id: RequestID = field()
    """A unique identifier for the request."""

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
    videos: list[bytes] = field(default_factory=list)
    """
    A list of video byte arrays that can be included as part of the request.
    Each video is decoded into frames during preprocessing.
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
    If set to ``True``, the response will include the original prompt along with
    the generated output. This can be useful for debugging or when you want to
    see how the input relates to the output.
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

    dkv_cache_hint: dict[str, Any] | None = None
    """Cache hint from the Orchestrator for distributed KV cache.

    When present, the serving layer converts this into
    ``TextContext.external_block_metadata`` so the DKVConnector can
    fetch cached blocks before the forward pass.
    """

    def __str__(self) -> str:
        return str(self.request_id)

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

        if self.videos and isinstance(self.prompt, str):
            raise ValueError(
                "string prompts cannot be provided, when videos are provided, use messages"
            )

        if self.images and self.number_of_images != len(self.images):
            raise ValueError(
                f"number of images provided in TextGenerationRequest do not match messages:\n{self.messages}"
            )

        if self.videos and self.number_of_videos != len(self.videos):
            raise ValueError(
                f"number of videos provided in TextGenerationRequest do not match messages:\n{self.messages}"
            )

    @cached_property
    def number_of_images(self) -> int:
        """Returns the total number of image-type contents across all provided messages.

        Returns:
            Total count of image-type contents found in messages.
        """
        return (
            sum(message.number_of_images for message in self.messages)
            if self.messages
            else 0
        )

    @cached_property
    def number_of_videos(self) -> int:
        """Returns the total number of video-type contents across all provided messages.

        Returns:
            Total count of video-type contents found in messages.
        """
        return (
            sum(message.number_of_videos for message in self.messages)
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

    num_cached_tokens: int | None = None
    """Number of prompt tokens served from the KV prefix cache."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the text generation process is complete.

        Returns:
            ``True`` if the generation is done, ``False`` otherwise.
        """
        return self.final_status.is_done

    @classmethod
    def merge(cls, outputs: list[TextGenerationOutput]) -> TextGenerationOutput:
        """Combine many TextGenerationOutput chunks into a single TextGenerationOutput."""
        if len(outputs) == 0:
            raise ValueError("Cannot combine empty list of chunks")
        if len(outputs) == 1:
            return outputs[0]

        if all(output.log_probabilities is not None for output in outputs):
            log_probabilities = list(
                chain.from_iterable(
                    output.log_probabilities or [] for output in outputs
                )
            )
        elif all(output.log_probabilities is None for output in outputs):
            log_probabilities = None
        else:
            raise ValueError(
                "Cannot combine TextGenerationOutput chunks with mixed None and non-None log_probabilities"
            )

        return cls(
            request_id=outputs[0].request_id,
            tokens=list(
                chain.from_iterable(output.tokens for output in outputs)
            ),
            log_probabilities=log_probabilities,
            final_status=outputs[-1].final_status,
            num_cached_tokens=outputs[0].num_cached_tokens,
        )


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
        produced, even if other stopping criteria are met (for example, EOS tokens).

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
        """Returns the token indices that should be masked in the output logits.

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

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance the token buffer without touching FSM state.

        This method handles token buffer mutations including log probability
        storage, token buffer advancement, and EOS/max-length status updates.
        It does NOT advance the FSM matcher.

        Use ``advance_fsm()`` separately if FSM advancement is needed, or use
        ``update()`` for the common case of advancing both together.

        Args:
            new_token: The token to append to the buffer.
            log_probabilities: Optional log probabilities for this token.
        """
        ...

    def advance_fsm(self, token: int) -> bool:
        """Advance the FSM matcher state by one token.

        This method advances only the FSM state for constrained decoding.
        It does NOT modify the token buffer. Use ``advance_token_buffer()``
        separately if token buffer advancement is needed, or use ``update()``
        for the common case of advancing both together.

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

        This is the standard single-step update that most callers should use.
        It combines ``advance_token_buffer()`` and ``advance_fsm()`` for the
        common case where both need to be advanced together.

        For multi-step execution where FSM is advanced separately (e.g., to
        compute bitmasks between steps), use the individual methods directly.

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

    @property
    def matcher(self) -> Any | None:
        """The grammar matcher for structured output generation, if configured.

        The matcher enforces structural constraints (like JSON schema) during
        generation to ensure valid formatted output.

        Returns:
            The grammar matcher instance, or ``None`` if no structured generation
            is configured for this context.

        Note:
            The matcher type depends on the structured generation backend used
            (for example, outlines, guidance, etc.). In the future, this should be
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

    @property
    def grammar(self) -> str | None:
        """Grammar for constrained decoding, if configured."""
        return None

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
            The :class:`~max.interfaces.context.SamplingParams` instance containing all sampling configuration
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
        """Converts this context to a :class:`TextGenerationOutput` object.

        Provides a standardized way to extract the final output of the text
        generation process from the context, including generated text, tokens,
        and any associated metadata.

        Returns:
            The output object containing the results of the text generation
            for this context.
        """
        ...

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        """Returns the speculative decoding state."""
        ...

    cached_prefix_length: int | None
    """Prompt tokens served from the KV prefix cache on first admission.

    Set by the block manager when a request is admitted to a CE batch (0
    if the cache had no matching prefix). ``BatchMetrics.create`` consumes
    the value to emit a per-request cache hit rate observation, then
    resets it to ``None`` so chunked-prefill follow-up calls do not
    re-emit.
    """

    in_reasoning_phase: bool
    """Whether the latest committed tokens are inside a ``<think>...</think>``
    block. Toggled host-side in the spec-decode commit step when a reasoning
    parser is configured. Consumed by thinking-mode temperature scaling and
    relaxed acceptance to gate per-row behavior."""


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
        """``True`` if any context in the batch has echo enabled."""
        return any(self.batch_echo)

    @property
    def enable_log_probs(self) -> bool:
        """``True`` if any context in the batch requests log probabilities."""
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
        """Positions of image-placeholder tokens within this context's token buffer.

        Offsets are relative to the start of the full token sequence (not the
        active window).  Used by ``compute_multimodal_merge_indices`` to build
        batch-level scatter indices that account for ``processed_length``.
        """
        ...

    def compute_image_aligned_idx(self, idx: int) -> int:
        """Aligns an index downward to avoid splitting an image token span.

        If ``idx`` falls within the token range occupied by an image, this
        method returns the ``start_idx`` of that image so that the split point
        does not cut through image tokens. If ``idx`` does not land inside any
        image span, it is returned unchanged.

        Args:
            idx: The candidate index into the token sequence.

        Returns:
            The adjusted index, guaranteed not to split an image token span.
        """
        ...


VLMContextType = TypeVar("VLMContextType", bound=VLMTextGenerationContext)
"""Type variable for VLM context types, constrained to VLMTextGenerationContext.

This allows generic typing of VLM pipeline components to accept any
context type that implements the VLMTextGenerationContext protocol.
"""
