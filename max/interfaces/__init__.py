# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from typing import Callable, Union

from .context import BaseContext, BaseContextType, InputContext, SamplingParams
from .log_probabilities import LogProbabilities
from .lora import LoRAOperation, LoRARequest, LoRAResponse, LoRAStatus, LoRAType
from .pipeline import Pipeline, PipelineOutputsDict
from .pipeline_variants import (
    AudioGenerationMetadata,
    AudioGenerationRequest,
    AudioGenerator,
    AudioGeneratorContext,
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
)
from .queue import MAXQueue
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
    [], Union[EmbeddingsGenerator, AudioGenerator, Pipeline]
]

__all__ = [
    "AudioGenerationMetadata",
    "AudioGenerationRequest",
    "AudioGenerator",
    "AudioGeneratorContext",
    "AudioGeneratorOutput",
    "BaseContext",
    "BaseContextType",
    "EmbeddingsGenerator",
    "EmbeddingsOutput",
    "GenerationStatus",
    "InputContext",
    "LoRAOperation",
    "LoRARequest",
    "LoRAResponse",
    "LoRAStatus",
    "LoRAType",
    "LogProbabilities",
    "MAXQueue",
    "Pipeline",
    "PipelineOutputsDict",
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
    "TextGenerationContextType",
    "TextGenerationInputs",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponse",
    "TextGenerationResponseFormat",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
