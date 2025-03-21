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

"""Interfaces for text generation pipeline behaviors."""

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from .response import TextGenerationResponse


class TokenGeneratorRequestFunction(TypedDict):
    name: str
    description: str
    parameters: dict


class TokenGeneratorRequestTool(TypedDict):
    type: str
    function: TokenGeneratorRequestFunction


class TokenGeneratorResponseFormat(TypedDict):
    type: str
    json_schema: dict


class TokenGeneratorRequestMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Union[str, list[dict[str, Any]]]
    """Content can be simple string or a list of message parts of different
    modalities.

    For example:

    .. code-block:: json

        {
          "role": "user",
          "content": "What'\''s the weather like in Boston today?"
        }

    Or:

    .. code-block:: json

        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What'\''s in this image?"
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
class TokenGeneratorRequest:
    id: str
    """
    A unique identifier for the request. This ID can be used to trace and log
    the request throughout its lifecycle, facilitating debugging and tracking.
    """
    index: int
    """
    The sequence order of this request within a batch. This is useful for
    maintaining the order of requests when processing multiple requests
    simultaneously, ensuring that responses can be matched back to their
    corresponding requests accurately.
    """
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
    messages: Optional[list[TokenGeneratorRequestMessage]] = None
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
    tools: Optional[list[TokenGeneratorRequestTool]] = None
    """
    A list of tools that can be invoked during the generation process. This
    allows the model to utilize external functionalities or APIs to enhance its
    responses.
    """
    response_format: Optional[TokenGeneratorResponseFormat] = None
    """
    Specifies the desired format for the model's output. When set, it enables
    structured generation, which adheres to the json_schema provided.
    """
    max_new_tokens: Optional[int] = None
    """
    The maximum number of new tokens to generate in the response. If not set,
    the model may generate tokens until it reaches its internal limits or based
    on other stopping criteria.
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
    stop: Optional[Union[str, List[str]]] = None
    """
    Optional list of stop expressions (see: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)
    """
    ignore_eos: bool = False
    """
    If set to True, the response will ignore the EOS token, and continue to generate until the Max tokens or a
    stop string is hit.
    """

    def __str__(self) -> str:
        txt = f"Id: {self.id}"
        if self.max_new_tokens:
            txt += f", MaxNewTokens: {self.max_new_tokens}"
        return txt


TokenGeneratorContext = TypeVar("TokenGeneratorContext")
TokenGeneratorBatchKey = TypeVar("TokenGeneratorBatchKey")

TokenizerEncoded = TypeVar("TokenizerEncoded")


@runtime_checkable
class PipelineTokenizer(
    Generic[TokenGeneratorContext, TokenizerEncoded], Protocol
):
    """Interface for LLM tokenizers."""

    @property
    def eos(self) -> int:
        """The end of sequence token for this tokenizer."""
        ...

    @property
    def expects_content_wrapping(self) -> bool:
        """If true, this tokenizer expects messages to have a `content` property.

        Text messages are formatted as:

        .. code-block:: json

            { "type": "text", "content": "text content" }

        instead of the OpenAI spec:

        .. code-block:: json

            { "type": "text", "text": "text content" }

        NOTE: Multimodal messages omit the `content` property.
        Both :obj:`image_urls` and :obj:`image` content parts are converted to:

        .. code-block:: json

            { "type": "image" }

        Their content is provided as byte arrays through the top-level property
        on the request object, i.e., :obj:`TokenGeneratorRequest.images`.
        """
        ...

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> TokenGeneratorContext:
        """Creates a new context from a request object. This is sent to the
        worker process once and then cached locally.

        Args:
            request (TokenGeneratorRequest): Incoming request.

        Returns:
            TokenGeneratorContext: Initialized context.
        """
        ...

    async def encode(
        self,
        prompt: str,
        add_special_tokens: bool,
    ) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt (str): Un-encoded prompt text.

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: TokenizerEncoded,
        **kwargs,
    ) -> str:
        """Decodes response tokens to text.

        Args:
            context (TokenGeneratorContext): Current generation context.
            encoded (TokenizerEncoded): Encoded response tokens.

        Returns:
            str: Un-encoded response text.
        """
        ...


@runtime_checkable
class TokenGenerator(Generic[TokenGeneratorContext], Protocol):
    """Interface for LLM token-generator models."""

    def next_token(
        self, batch: dict[str, TokenGeneratorContext], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Computes the next token response for a single batch.

        Args:
            batch (dict[str, TokenGeneratorContext]): Batch of contexts.
            num_steps int: Number of tokens to generate.

        Returns:
            list[dict[str, TextResponse]]: List of encoded responses (indexed by req. ID)
        """
        ...

    def release(self, context: TokenGeneratorContext) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.
        """
        ...
