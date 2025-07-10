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

    Attributes:
        TEXT_GENERATION: Task for generating text.
        EMBEDDINGS_GENERATION: Task for generating embeddings.
        AUDIO_GENERATION: Task for generating audio.
        SPEECH_TOKEN_GENERATION: Task for generating speech tokens.
    """

    TEXT_GENERATION = "text_generation"
    EMBEDDINGS_GENERATION = "embeddings_generation"
    AUDIO_GENERATION = "audio_generation"
    SPEECH_TOKEN_GENERATION = "speech_token_generation"
