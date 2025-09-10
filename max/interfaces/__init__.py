# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from typing import Any, Callable, Union

from .context import (
    BaseContext,
    BaseContextType,
    InputContext,
    SamplingParams,
    SamplingParamsInput,
)
from .log_probabilities import LogProbabilities
from .logit_processors_type import (
    BatchLogitsProcessor,
    BatchProcessorInputs,
    LogitsProcessor,
    ProcessorInputs,
)
from .lora import LoRAOperation, LoRARequest, LoRAResponse, LoRAStatus, LoRAType
from .pipeline import (
    Pipeline,
    PipelineOutput,
    PipelineOutputsDict,
    PipelineOutputType,
)
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
from .queue import MAXPullQueue, MAXPushQueue, drain_queue, get_blocking
from .request import Request, RequestID
from .scheduler import Scheduler, SchedulerResult
from .status import GenerationStatus
from .task import PipelineTask
from .tokenizer import PipelineTokenizer
from .utils import (
    SharedMemoryArray,
    msgpack_eq,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)

PipelinesFactory = Callable[
    # TODO(GENAI-245): Use of Any here is not safe.
    [], Union[EmbeddingsGenerator[Any], AudioGenerator[Any], Pipeline[Any, Any]]
]

__all__ = [
    "AudioGenerationMetadata",
    "AudioGenerationRequest",
    "AudioGenerator",
    "AudioGeneratorContext",
    "AudioGeneratorOutput",
    "BaseContext",
    "BaseContextType",
    "BatchLogitsProcessor",
    "BatchProcessorInputs",
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
    "LogitsProcessor",
    "MAXPullQueue",
    "MAXPushQueue",
    "Pipeline",
    "PipelineOutput",
    "PipelineOutputType",
    "PipelineOutputsDict",
    "PipelineTask",
    "PipelineTokenizer",
    "PipelineTokenizer",
    "PipelinesFactory",
    "ProcessorInputs",
    "Request",
    "RequestID",
    "SamplingParams",
    "SamplingParamsInput",
    "Scheduler",
    "SchedulerResult",
    "SharedMemoryArray",
    "TextGenerationContextType",
    "TextGenerationContextType",
    "TextGenerationInputs",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponseFormat",
    "drain_queue",
    "get_blocking",
    "msgpack_eq",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
