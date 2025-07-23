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
    AudioGenerationRequest,
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
from .request import Request, RequestID
from .scheduler import SchedulerResult, SchedulerStatus
from .status import GenerationStatus
from .task import PipelineTask
from .tokenizer import PipelineTokenizer

__all__ = [
    "AudioGenerationMetadata",
    "AudioGenerationRequest",
    "AudioGenerationResponse",
    "AudioGeneratorOutput",
    "EmbeddingsOutput",
    "EngineStatus",
    "GenerationStatus",
    "InputContext",
    "LogProbabilities",
    "PipelineOutput",
    "PipelineTask",
    "PipelineTokenizer",
    "Request",
    "RequestID",
    "SamplingParams",
    "SchedulerResult",
    "SchedulerStatus",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponse",
    "TextGenerationResponseFormat",
    "TextResponse",
    "TokenGenerator",
]
