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

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

from .response import AudioGenerationResponse
from .text_generation import SamplingParams


@dataclass(frozen=True)
class AudioGenerationRequest:
    id: str
    """A unique identifier for the request. This ID can be used to trace and log
    the request throughout its lifecycle, facilitating debugging and tracking.
    """

    index: int
    """The sequence order of this request within a batch. This is useful for
    maintaining the order of requests when processing multiple requests
    simultaneously, ensuring that responses can be matched back to their
    corresponding requests accurately.
    """

    model: str
    """The name of the model to be used for generating audio chunks. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """

    input: Optional[str] = None
    """The text to generate audio for. The maximum length is 4096 characters.
    """

    audio_prompt_tokens: list[int] = field(default_factory=list)
    """The prompt speech IDs to use for audio generation."""

    audio_prompt_transcription: str = ""
    """The audio prompt transcription to use for audio generation."""

    sampling_params: SamplingParams = SamplingParams()
    """Request sampling configuration options."""

    _assistant_message_override: str | None = None
    """(ONLY FOR BENCHMARKING PURPOSES) An assistant message that replaces the
    speech token pattern."""

    prompt: Optional[list[int] | str] = field(default=None)
    """Optionally provide a preprocessed list of token ids or a prompt string to pass as input directly into the model.
    This replaces automatically generating TokenGeneratorRequestMessages given the input, audio prompt tokens,
    audio prompt transcription fields."""

    streaming: bool = True
    """Whether to stream the audio generation."""

    def __post_init__(self) -> None:
        if self.prompt is None and self.input is None:
            raise RuntimeError("either token_ids or input must be provided.")


AudioGeneratorContext = TypeVar("AudioGeneratorContext")

TokenizerEncoded = TypeVar("TokenizerEncoded")

DecoderOutput = TypeVar("DecoderOutput")


@dataclass(frozen=True)
class AudioGeneratorOutput:
    audio_data: torch.Tensor
    metadata: dict[str, Any]
    is_done: bool


@runtime_checkable
class AudioGenerator(Generic[AudioGeneratorContext], Protocol):
    """Interface for audio generation models."""

    def next_chunk(
        self, batch: dict[str, AudioGeneratorContext]
    ) -> dict[str, AudioGenerationResponse]:
        """Computes the next audio chunk for a single batch.

        The new speech tokens are saved to the context. The most recently
        generated audio is return through the `AudioGenerationResponse`.

        Args:
            batch (dict[str, AudioGeneratorContext]): Batch of contexts.

        Returns:
            dict[str, AudioGenerationResponse]: Dictionary mapping request IDs to
                audio generation responses.
        """
        ...

    def release(self, context: AudioGeneratorContext) -> None:
        """Releases resources associated with this context.

        Args:
            context (AudioGeneratorContext): Finished context.
        """
        ...

    @property
    def decoder_sample_rate(self) -> int:
        """The sample rate of the decoder."""
        ...

    @property
    def prev_num_steps(self) -> int:
        """The number of speech tokens that were generated during the processing
        of the previous batch.
        """
        ...
