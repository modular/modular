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
    TypeVar,
    runtime_checkable,
)

from max.interfaces import AudioGenerationResponse

AudioGeneratorContext = TypeVar("AudioGeneratorContext")

TokenizerEncoded = TypeVar("TokenizerEncoded")

DecoderOutput = TypeVar("DecoderOutput")


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
