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

from __future__ import annotations

from typing import (
    Generic,
    Protocol,
    runtime_checkable,
)

from typing_extensions import TypeVar

TokenGeneratorContext = TypeVar("TokenGeneratorContext")
TokenGeneratorBatchKey = TypeVar("TokenGeneratorBatchKey")

TokenizerEncoded = TypeVar("TokenizerEncoded")
PipelineTokenizerRequest = TypeVar(
    "PipelineTokenizerRequest", contravariant=True
)


@runtime_checkable
class PipelineTokenizer(
    Generic[TokenGeneratorContext, TokenizerEncoded, PipelineTokenizerRequest],
    Protocol,
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
        on the request object, i.e., :obj:`PipelineTokenizerRequest.images`.
        """
        ...

    async def new_context(
        self, request: PipelineTokenizerRequest
    ) -> TokenGeneratorContext:
        """Creates a new context from a request object. This is sent to the
        worker process once and then cached locally.

        Args:
            request (PipelineTokenizerRequest): Incoming request.

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
