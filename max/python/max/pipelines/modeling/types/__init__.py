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

from max.pipelines.request import (
    DUMMY_REQUEST_ID,
    OpenResponsesRequest,
    Request,
    RequestID,
    RequestType,
)

from .context import (
    BaseContext,
    BaseContextType,
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
)
from .eos_tracking import EOSTracker
from .generation import GenerationOutput
from .log_probabilities import LogProbabilities
from .logit_processors_type import (
    BatchLogitsProcessor,
    BatchProcessorInputs,
    LogitsProcessor,
    ProcessorInputs,
)
from .lora import (
    LORA_REQUEST_ENDPOINT,
    LORA_RESPONSE_ENDPOINT,
    LoRAOperation,
    LoRARequest,
    LoRAResponse,
    LoRAStatus,
    LoRAType,
)
from .pipeline import (
    Pipeline,
    PipelineInputs,
    PipelineInputsType,
    PipelineOutput,
    PipelineOutputsDict,
    PipelineOutputType,
)
from .pipeline_variants import (
    BatchType,
    EmbeddingsContext,
    EmbeddingsGenerationContextType,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    GrammarEnforcementSnapshot,
    ImageContentPart,
    ImageMetadata,
    MessageContent,
    PixelGenerationContext,
    PixelGenerationContextType,
    PixelGenerationInputs,
    SpecDecodingState,
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
    VideoContentPart,
    VLMTextGenerationContext,
)
from .reasoning import ParsedReasoningDelta, ReasoningParser, ReasoningSpan
from .status import GenerationStatus
from .task import InputModality, PipelineTask
from .tokenizer import PipelineTokenizer, TokenizerEncoded, UnboundContextType
from .tokens import Range, TokenBuffer, TokenSlice
from .tool_parsing import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    ToolParser,
)
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
    "LORA_REQUEST_ENDPOINT",
    "LORA_RESPONSE_ENDPOINT",
    "BaseContext",
    "BaseContextType",
    "BatchLogitsProcessor",
    "BatchProcessorInputs",
    "BatchType",
    "EOSTracker",
    "EmbeddingsContext",
    "EmbeddingsGenerationContextType",
    "EmbeddingsGenerationInputs",
    "EmbeddingsGenerationOutput",
    "GenerationOutput",
    "GenerationStatus",
    "GrammarEnforcementSnapshot",
    "ImageContentPart",
    "ImageMetadata",
    "InputModality",
    "LoRAOperation",
    "LoRARequest",
    "LoRAResponse",
    "LoRAStatus",
    "LoRAType",
    "LogProbabilities",
    "LogitsProcessor",
    "MessageContent",
    "OpenResponsesRequest",
    "ParsedReasoningDelta",
    "ParsedToolCall",
    "ParsedToolCallDelta",
    "ParsedToolResponse",
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
    "ProcessorInputs",
    "Range",
    "ReasoningParser",
    "ReasoningSpan",
    "Request",
    "RequestID",
    "RequestType",
    "SamplingParams",
    "SamplingParamsGenerationConfigDefaults",
    "SamplingParamsInput",
    "SharedMemoryArray",
    "SpecDecodingState",
    "TextContentPart",
    "TextGenerationContext",
    "TextGenerationContextType",
    "TextGenerationInputs",
    "TextGenerationOutput",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TextGenerationResponseFormat",
    "TokenBuffer",
    "TokenSlice",
    "TokenizerEncoded",
    "ToolParser",
    "UnboundContextType",
    "VLMTextGenerationContext",
    "VideoContentPart",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
]
