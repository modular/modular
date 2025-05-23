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

import enum
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .response import TextGenerationStatus


class AudioFormat(enum.Enum):
    """The format of the audio to be generated."""

    WAV = "wav"


@dataclass(frozen=True)
class AudioGenerationRequest:
    id: str
    """
    A unique identifier for the request. This ID can be used to trace and log
    the request throughout its lifecycle, facilitating debugging and tracking.
    """
    input: str
    """
    The text to generate audio for. The maximum length is 4096 characters.
    """
    index: int
    """
    The sequence order of this request within a batch. This is useful for
    maintaining the order of requests when processing multiple requests
    simultaneously, ensuring that responses can be matched back to their
    corresponding requests accurately.
    """
    model: str
    """
    The name of the model to be used for generating audio chunks. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    voice: str
    """
    The voice to use for audio generation.
    """
    instructions: str = ""
    """
    Control the voice of your generated audio with additional instructions.
    Currently unused.
    """
    response_format: AudioFormat = AudioFormat.WAV
    """
    The format to audio in. Currently only supports wav.
    """
    speed: float = 1.0
    """
    The speed of the generated audio. Select a value from 0.25 to 4.0.
    Defaults to 1.0.
    """


AudioGeneratorContext = TypeVar("AudioGeneratorContext")

TokenizerEncoded = TypeVar("TokenizerEncoded")

DecoderOutput = TypeVar("DecoderOutput")


@runtime_checkable
class PipelineAudioTokenizer(
    Generic[AudioGeneratorContext, TokenizerEncoded], Protocol
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
        self, request: AudioGenerationRequest
    ) -> AudioGeneratorContext:
        """Creates a new context from a request object. This is sent to the
        worker process once and then cached locally.

        Args:
            request (AudioGenerationRequest): Incoming request.

        Returns:
            AudioGeneratorContext: Initialized context.
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
            add_special_tokens (bool): Whether to add special tokens to the
                prompt.

        Returns:
            TokenizerEncoded: Encoded tokens.

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(
        self,
        context: AudioGeneratorContext,
        encoded: TokenizerEncoded,
        **kwargs,
    ) -> str:
        """Decodes response tokens to text.

        Args:
            context (AudioGeneratorContext): Current generation context.
            encoded (TokenizerEncoded): Encoded response tokens.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: Un-encoded response text.
        """
        ...


@runtime_checkable
class AudioGenerator(Generic[AudioGeneratorContext, DecoderOutput], Protocol):
    """Interface for audio generation models."""

    def next_chunk(
        self, batch: dict[str, AudioGeneratorContext], num_tokens: int
    ) -> dict[str, TextGenerationStatus]:
        """Computes the next audio chunk for a single batch.

        The new speech tokens are saved to the context.

        Args:
            batch (dict[str, AudioGeneratorContext]): Batch of contexts.
            num_tokens (int): Number of speech tokens to generate.

        Returns:
            dict[str, TextGenerationStatus]: Dictionary mapping request IDs to
                speech token generation status.
        """
        ...

    def decode(
        self, batch: dict[str, AudioGeneratorContext], num_tokens: int
    ) -> dict[str, DecoderOutput]:
        """Decodes speech tokens to audio bytes.

        Args:
            batch (dict[str, AudioGeneratorContext]): Batch of audio generation contexts.
            num_tokens (int): Number of speech tokens to decode.

        Returns:
            dict[str, DecoderOutput]: Dictionary mapping request IDs to WAV audio data.
        """
        ...

    def release(self, context: AudioGeneratorContext) -> None:
        """Releases resources associated with this context.

        Args:
            context (AudioGeneratorContext): Finished context.
        """
        ...
