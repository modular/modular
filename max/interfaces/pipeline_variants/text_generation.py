# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, TypedDict, TypeVar, Union

import msgspec
from max.interfaces.context import BaseContext, SamplingParams
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.pipeline import PipelineInputs
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

    content: Union[str, list[dict[str, Any]]]
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
    model_name: str
    """
    The name of the model to be used for generating tokens. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: Union[str, Sequence[int], None] = None
    """
    The prompt to be processed by the model. This field supports legacy
    completion APIs and can accept either a string or a sequence of integers
    representing token IDs. If not provided, the model may generate output
    based on the messages field.
    """
    messages: Optional[list[TextGenerationRequestMessage]] = None
    """
    A list of messages for chat-based interactions. This is used in chat
    completion APIs, where each message represents a turn in the conversation.
    If provided, the model will generate responses based on these messages.
    """
    images: Optional[list[bytes]] = None
    """
    A list of image byte arrays that can be included as part of the request.
    This field is optional and may be used for multimodal inputs where images
    are relevant to the prompt or task.
    """
    tools: Optional[list[TextGenerationRequestTool]] = None
    """
    A list of tools that can be invoked during the generation process. This
    allows the model to utilize external functionalities or APIs to enhance its
    responses.
    """
    response_format: Optional[TextGenerationResponseFormat] = None
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
    stop: Optional[Union[str, list[str]]] = None
    """
    Optional list of stop expressions (see: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)
    """
    chat_template_options: Optional[dict[str, Any]] = None
    """
    Optional dictionary of options to pass when applying the chat template.
    """

    sampling_params: SamplingParams = SamplingParams()
    """Token sampling configuration parameters for the request."""

    target_endpoint: Optional[str] = None
    """
    Optional target endpoint identifier for routing the request to a specific
    service or model instance. This should be used in disaggregate serving 
    scenarios, when you want to dynamically route to a specific instance.
    If not specified, the request will be routed to the default endpoint.
    """


class TextGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents the output of a text generation operation, combining token IDs,
    final generation status, request ID, and optional log probabilities for each token.
    """

    request_id: str
    """The unique identifier for the generation request."""

    tokens: list[int]
    """List of generated token IDs."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    log_probabilities: Optional[list[LogProbabilities]] = None
    """Optional list of log probabilities for each token."""

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the text generation process is complete.

        Returns:
            bool: True if the generation is done, False otherwise.
        """
        return self.final_status.is_done


# NOTE: TextGenerationContextType only enforces the BaseContext protocol, which provides the minimal contract for context objects.
# It is NOT sufficient for actual text generation functionality, as real text generation requires additional methods and properties
# (such as token management, status updates, and sampling parameters) that are not guaranteed by BaseContext alone.
# This is a temporary solution to allow for gradual migration of the codebase to the new context interface.


TextGenerationContextType = TypeVar(
    "TextGenerationContextType",
    bound=BaseContext,
)
"""Type variable for text generation context types, constrained to BaseContext.

This allows generic typing of text generation pipeline components to accept any
context type that implements the BaseContext protocol.
"""


@dataclass(frozen=True)
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
    """Number of tokens to generate."""

    @property
    def batch(self) -> dict[RequestID, TextGenerationContextType]:
        """Returns merged batches."""
        return {k: v for batch in self.batches for k, v in batch.items()}
