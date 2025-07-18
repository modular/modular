# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Audio generation interface definitions for Modular's MAX API.

This module provides data structures and interfaces for handling audio generation
responses, including status tracking and audio data encapsulation.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import msgspec
import numpy as np
from max.interfaces.status import GenerationStatus


class AudioGenerationMetadata(
    msgspec.Struct, tag=True, omit_defaults=True, kw_only=True
):
    """
    Represents metadata associated with audio generation.

    This class will eventually replace the metadata dictionary used throughout
    the AudioGenerationOutput object, providing a structured and type-safe
    alternative for audio generation metadata.

    Configuration:
        sample_rate: The sample rate of the generated audio in Hz.
        duration: The duration of the generated audio in seconds.
        chunk_id: Identifier for the audio chunk (useful for streaming).
        timestamp: Timestamp when the audio was generated.
        final_chunk: Whether this is the final chunk in a streaming sequence.
        model_name: Name of the model used for generation.
        request_id: Unique identifier for the generation request.
        tokens_generated: Number of tokens generated for this audio.
        processing_time: Time taken to process this audio chunk in seconds.
        echo: Echo of the input prompt or identifier for verification.
    """

    sample_rate: Optional[int] = msgspec.field(default=None)
    duration: Optional[float] = msgspec.field(default=None)
    chunk_id: Optional[int] = msgspec.field(default=None)
    timestamp: Optional[str] = msgspec.field(default=None)
    final_chunk: Optional[bool] = msgspec.field(default=None)
    model_name: Optional[str] = msgspec.field(default=None)
    request_id: Optional[str] = msgspec.field(default=None)
    tokens_generated: Optional[int] = msgspec.field(default=None)
    processing_time: Optional[float] = msgspec.field(default=None)
    echo: Optional[str] = msgspec.field(default=None)

    def to_dict(self) -> dict[str, Union[int, float, str, bool]]:
        """
        Convert the metadata to a dictionary format.

        Returns:
            dict[str, any]: Dictionary representation of the metadata.
        """
        result = {}
        for attr in self.__annotations__:
            if value := getattr(self, attr, None):
                result[attr] = value
        return result

    def __eq__(self, other: Any) -> bool:
        """
        Support equality comparison with both AudioGenerationMetadata objects and dictionaries.

        This allows tests to compare metadata objects with plain dictionaries.
        """
        if isinstance(other, AudioGenerationMetadata):
            return super().__eq__(other)
        elif isinstance(other, dict):
            return self.to_dict() == other
        return False


class AudioGenerationResponse(msgspec.Struct, tag=True, omit_defaults=True):
    """Represents a response from the audio generation API.

    This class encapsulates the result of an audio generation request, including
    the final status, generated audio data, and optional buffered speech tokens.
    """

    final_status: GenerationStatus
    """The final status of the generation process."""
    audio: Optional[np.ndarray] = msgspec.field(default=None)
    """The generated audio data, if available."""
    buffer_speech_tokens: Optional[np.ndarray] = msgspec.field(default=None)
    """Buffered speech tokens, if available."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the audio generation process is complete.

        Returns:
            :class:`bool`: ``True`` if generation is done, ``False`` otherwise.
        """
        return self.final_status.is_done

    @property
    def has_audio_data(self) -> bool:
        """Checks if audio data is present in the response.

        Returns:
            :class:`bool`: ``True`` if audio data is available, ``False`` otherwise.
        """
        return self.audio is not None

    @property
    def audio_data(self) -> np.ndarray:
        """Returns the audio data if available.

        Returns:
            :class:`np.ndarray`: The generated audio data.

        Raises:
            :class:`AssertionError`: If audio data is not available.
        """
        assert self.audio is not None
        return self.audio


class AudioGeneratorOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents the output of an audio generation step.
    """

    audio_data: np.ndarray = msgspec.field()
    """The generated audio data as a NumPy array."""

    metadata: AudioGenerationMetadata = msgspec.field()
    """Metadata associated with the audio generation, such as chunk information, prompt details, or other relevant context."""

    is_done: bool = msgspec.field()
    """Indicates whether the audio generation is complete (True) or if more chunks are expected (False)."""

    buffer_speech_tokens: np.ndarray | None = msgspec.field(default=None)
    """An optional field containing the last N speech tokens generated by the model. This can be used to buffer speech tokens for a follow-up request, enabling seamless continuation of audio generation."""
