# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Pipeline Tasks Module.

This module defines the set of supported pipeline tasks for the MAX API, encapsulated
in the `PipelineTask` enumeration. Pipeline tasks represent the high-level operations
that can be performed by a pipeline, such as text generation, embeddings generation,
audio generation, speech token generation, and pixel generation.

Each task type is associated with a specific input/output contract and is used to
route requests to the appropriate pipeline implementation. The `PipelineTask` enum
is used throughout the MAX API to ensure type safety and consistency when specifying
or querying the type of task a pipeline supports.

Typical usage includes:
    - Registering supported architectures and pipelines for a given task.
    - Determining the output type for a pipeline task.
    - Routing inference requests to the correct pipeline based on the task.

Available tasks:
    - TEXT_GENERATION: Generate text sequences from input prompts.
    - EMBEDDINGS_GENERATION: Generate vector embeddings for input data.
    - AUDIO_GENERATION: Generate audio outputs from input data.
    - SPEECH_TOKEN_GENERATION: Generate speech tokens for speech-related tasks.
    - PIXEL_GENERATION: Generate/Edit images/videos from input data.

See the `PipelineTask` enum for further details on each task type.
"""

from enum import Enum
from typing import Any

from .request import RequestID
from .scheduler import SchedulerResult


class PipelineTask(str, Enum):
    """Enum representing the types of pipeline tasks supported."""

    TEXT_GENERATION = "text_generation"
    """Task for generating text."""
    EMBEDDINGS_GENERATION = "embeddings_generation"
    """Task for generating embeddings."""
    AUDIO_GENERATION = "audio_generation"
    """Task for generating audio."""
    SPEECH_TOKEN_GENERATION = "speech_token_generation"
    """Task for generating speech tokens."""
    PIXEL_GENERATION = "pixel_generation"
    """Task for generating pixels."""

    @property
    def output_type(
        self,
    ) -> type[dict[RequestID, SchedulerResult[Any]]]:
        """Get the output type for the pipeline task.

        Returns:
            type: The output type for the pipeline task.
        """
        from .generation import GenerationOutput
        from .pipeline_variants import (
            AudioGenerationOutput,
            EmbeddingsGenerationOutput,
            TextGenerationOutput,
        )
        from .scheduler import SchedulerResult

        if self in [
            PipelineTask.TEXT_GENERATION,
            PipelineTask.SPEECH_TOKEN_GENERATION,
        ]:
            return dict[RequestID, SchedulerResult[TextGenerationOutput]]
        elif self == PipelineTask.EMBEDDINGS_GENERATION:
            return dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
        elif self == PipelineTask.AUDIO_GENERATION:
            return dict[RequestID, SchedulerResult[AudioGenerationOutput]]
        elif self == PipelineTask.PIXEL_GENERATION:
            return dict[RequestID, SchedulerResult[GenerationOutput]]
        else:
            raise ValueError(
                f"PipelineTask ({self}) does not have an output_type defined."
            )
