# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from collections.abc import Callable

from .context import (
    BaseContext,
    BaseContextType,
    ImageMetadata,
    InputContext,
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
    VLMInputContext,
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
    PipelineInputs,
    PipelineInputsType,
    PipelineOutput,
    PipelineOutputsDict,
    PipelineOutputType,
)
from .pipeline_variants import (
    AudioGenerationContextType,
    AudioGenerationInputs,
    AudioGenerationMetadata,
    AudioGenerationOutput,
    AudioGenerationRequest,
    BatchType,
    EmbeddingsGenerationContextType,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    TextGenerationContext,
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
from .request import Request, RequestID, RequestType
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
    [], Pipeline[PipelineInputsType, PipelineOutputType]
]
"""
Type alias for factory functions that create pipeline instances.

Factory functions should return a Pipeline with properly typed inputs and outputs
that are bound to the PipelineInputs and PipelineOutput base classes respectively.
This ensures type safety while maintaining flexibility for different pipeline implementations.

Example:
    def create_text_pipeline() -> Pipeline[TextGenerationInputs, TextGenerationOutput]:
        return MyTextGenerationPipeline()

    factory: PipelinesFactory = create_text_pipeline
"""

__all__ = [
    "AudioGenerationContextType",
    "AudioGenerationInputs",
    "AudioGenerationMetadata",
    "AudioGenerationOutput",
    "AudioGenerationRequest",
    "BaseContext",
    "BaseContextType",
    "BatchLogitsProcessor",
    "BatchProcessorInputs",
    "BatchType",
    "EmbeddingsGenerationContextType",
    "EmbeddingsGenerationInputs",
    "EmbeddingsGenerationOutput",
    "GenerationStatus",
    "ImageMetadata",
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
    "PipelineInputs",
    "PipelineInputsType",
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
    "RequestType",
    "SamplingParams",
    "SamplingParamsGenerationConfigDefaults",
    "SamplingParamsInput",
    "Scheduler",
    "SchedulerResult",
    "SharedMemoryArray",
    "TextGenerationContext",
    "TextGenerationContextType",
    "TextGenerationInputs",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponseFormat",
    "VLMInputContext",
    "drain_queue",
    "get_blocking",
    "msgpack_eq",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
