# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from .context import InputContext, SamplingParams
from .log_probabilities import LogProbabilities
from .pipeline import PipelineOutput
from .pipeline_variants import (
    AudioGenerationMetadata,
    AudioGenerationResponse,
    AudioGeneratorOutput,
    EmbeddingsOutput,
    TextGenerationOutput,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TextGenerationResponseFormat,
    TokenGenerator,
)
from .scheduler import SchedulerResult, SchedulerStatus
from .status import GenerationStatus
from .task import PipelineTask

__all__ = [
    "AudioGenerationMetadata",
    "AudioGenerationResponse",
    "AudioGeneratorOutput",
    "EmbeddingsOutput",
    "EngineStatus",
    "GenerationStatus",
    "GenerationStatus",
    "InputContext",
    "LogProbabilities",
    "LogProbabilities",
    "PipelineOutput",
    "PipelineTask",
    "PipelineTask",
    "SamplingParams",
    "SamplingParams",
    "SchedulerResult",
    "SchedulerResult",
    "SchedulerStatus",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponse",
    "TextGenerationResponseFormat",
    "TextGenerationResponseFormat",
    "TextResponse",
    "TokenGenerator",
    "TokenGenerator",
]
