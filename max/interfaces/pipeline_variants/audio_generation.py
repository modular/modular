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
