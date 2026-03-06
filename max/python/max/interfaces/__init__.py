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
"""Universal interfaces between all aspects of the MAX Inference Stack."""

from collections.abc import Callable

from .context import (
    BaseContext,
    BaseContextType,
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
)
from .generation import GenerationOutput
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
    EmbeddingsContext,
    EmbeddingsGenerationContextType,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    ImageContentPart,
    ImageMetadata,
    MessageContent,
    PixelGenerationContext,
    PixelGenerationContextType,
    PixelGenerationInputs,
    PixelGenerationOutput,
    TextContentPart,
    TextGenerationContext,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TextGenerationResponseFormat,
    VLMTextGenerationContext,
)
from .queue import MAXPullQueue, MAXPushQueue, drain_queue, get_blocking
from .request import (
    DUMMY_REQUEST_ID,
    OpenResponsesRequest,
    Request,
    RequestID,
    RequestType,
)
from .scheduler import Scheduler, SchedulerResult
from .status import GenerationStatus
from .task import PipelineTask
from .tokenizer import PipelineTokenizer
from .tokens import TokenBuffer, TokenSlice
from .utils import (
    SharedMemoryArray,
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
    "DUMMY_REQUEST_ID",
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
    "EmbeddingsContext",
    "EmbeddingsGenerationContextType",
    "EmbeddingsGenerationInputs",
    "EmbeddingsGenerationOutput",
    "GenerationOutput",
    "GenerationStatus",
    "ImageContentPart",
    "ImageMetadata",
    "LoRAOperation",
    "LoRARequest",
    "LoRAResponse",
    "LoRAStatus",
    "LoRAType",
    "LogProbabilities",
    "LogitsProcessor",
    "MAXPullQueue",
    "MAXPushQueue",
    "MessageContent",
    "OpenResponsesRequest",
    "Pipeline",
    "PipelineInputs",
    "PipelineInputsType",
    "PipelineOutput",
    "PipelineOutputType",
    "PipelineOutputsDict",
    "PipelineTask",
    "PipelineTokenizer",
    "PipelinesFactory",
    "PixelGenerationContext",
    "PixelGenerationContextType",
    "PixelGenerationInputs",
    "PixelGenerationOutput",
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
    "TextContentPart",
    "TextGenerationContext",
    "TextGenerationContextType",
    "TextGenerationInputs",
    "TextGenerationOutput",
    "TextGenerationPipelineInterface",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponseFormat",
    "TokenBuffer",
    "TokenSlice",
    "VLMTextGenerationContext",
    "drain_queue",
    "get_blocking",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
