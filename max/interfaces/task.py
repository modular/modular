# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Pipeline Tasks."""

from enum import Enum


class PipelineTask(str, Enum):
    """
    Enum representing the types of pipeline tasks supported.
    """

    TEXT_GENERATION = "text_generation"
    """Task for generating text."""
    EMBEDDINGS_GENERATION = "embeddings_generation"
    """Task for generating embeddings."""
    AUDIO_GENERATION = "audio_generation"
    """Task for generating audio."""
    SPEECH_TOKEN_GENERATION = "speech_token_generation"
    """Task for generating speech tokens."""
