# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Audio generation interface definitions for Modular's MAX API.

This module provides data structures and interfaces for handling audio generation
responses, including status tracking and audio data encapsulation.
"""

from typing import Optional

import msgspec
import numpy as np
from max.interfaces.status import GenerationStatus


class AudioGenerationResponse(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents a response from the audio generation API.

    Configuration:
        final_status: The final status of the generation process.
        audio: The generated audio data, if available.
        buffer_speech_tokens: Buffered speech tokens, if available.
    """

    final_status: GenerationStatus
    audio: Optional[np.ndarray] = msgspec.field(default=None)
    buffer_speech_tokens: Optional[np.ndarray] = msgspec.field(default=None)

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the audio generation process is complete.

        Returns:
            bool: True if generation is done, False otherwise.
        """
        return self.final_status.is_done

    @property
    def has_audio_data(self) -> bool:
        """
        Checks if audio data is present in the response.

        Returns:
            bool: True if audio data is available, False otherwise.
        """
        return self.audio is not None

    @property
    def audio_data(self) -> np.ndarray:
        """
        Returns the audio data if available.

        Returns:
            np.ndarray: The generated audio data.

        Raises:
            AssertionError: If audio data is not available.
        """
        assert self.audio is not None
        return self.audio
