# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from typing import Callable, Union

from .context import (
    BaseContext,
    InputContext,
    SamplingParams,
)
from .log_probabilities import LogProbabilities
from .pipeline import PipelineOutput
from .pipeline_variants import (
    AudioGenerationMetadata,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioGenerator,
    AudioGeneratorOutput,
    EmbeddingsGenerator,
    EmbeddingsOutput,
    TextGenerationContextType,
    TextGenerationInputs,
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
from .utils import (
    SharedMemoryArray,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)

PipelinesFactory = Callable[
    [], Union[TokenGenerator, EmbeddingsGenerator, AudioGenerator]
]

__all__ = [
    "AudioGenerationMetadata",
    "AudioGenerationRequest",
    "AudioGenerationResponse",
    "AudioGenerator",
    "AudioGeneratorOutput",
    "BaseContext",
    "EmbeddingsGenerator",
    "EmbeddingsOutput",
    "GenerationStatus",
    "InputContext",
    "LogProbabilities",
    "PipelineOutput",
    "PipelineTask",
    "PipelineTokenizer",
    "PipelineTokenizer",
    "PipelinesFactory",
    "Request",
    "RequestID",
    "SamplingParams",
    "SchedulerResult",
    "SchedulerStatus",
    "SharedMemoryArray",
    "TextGenerationContextType",
    "TextGenerationInputs",
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
