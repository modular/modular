# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from .request import RequestType

# TODO: Bound this to TextGenerationContext, after we've audited the class.
UnboundContextType = TypeVar("UnboundContextType", covariant=True)
TokenizerEncoded = TypeVar("TokenizerEncoded")


@runtime_checkable
class PipelineTokenizer(
    Protocol[UnboundContextType, TokenizerEncoded, RequestType]
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
        on the request object, i.e., :obj:`RequestType.images`.
        """
        ...

    async def new_context(self, request: RequestType) -> UnboundContextType:
        """Creates a new context from a request object. This is sent to the
        worker process once and then cached locally.

        Args:
            request (RequestType): Incoming request.

        Returns:
            UnboundContextType: Initialized context.
        """
        ...

    async def encode(
        self, prompt: str, add_special_tokens: bool
    ) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt (str): Un-encoded prompt text.

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(self, encoded: TokenizerEncoded, **kwargs) -> str:
        """Decodes response tokens to text.

        Args:
            encoded (TokenizerEncoded): Encoded response tokens.

        Returns:
            str: Un-encoded response text.
        """
        ...
