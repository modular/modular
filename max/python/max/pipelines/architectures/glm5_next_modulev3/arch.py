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
"""GLM-5.3-Flash on ModuleV3.

Selected over the graph-API architecture by ``prefer_module_v3``; both read
the same checkpoint and the same cache tree.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.glm5_1.reasoning import (  # noqa: F401  registers "glm45"
    GlmReasoningParser,
)
from max.pipelines.architectures.glm5_1.tokenizer import GlmTokenizer
from max.pipelines.architectures.glm5_1.tool_parser import (  # noqa: F401  registers "glm45"
    GlmToolParser,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3_modulev3.batch_processor import (
    DeepseekV3ModuleV3BatchProcessor,
)
from .memory_planner import Glm5NextMemoryPlanner
from .model import Glm5NextModel
from .model_config import Glm5NextConfig
from .weight_adapters import convert_glm5_next_state_dict

__all__ = ["glm5_next_modulev3_arch"]

glm5_next_modulev3_arch = SupportedArchitecture(
    name="Glm5NextForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["zai-org/GLM-5.3-Flash"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Glm5NextConfig.DEFAULT_ENCODING,
    supported_encodings=Glm5NextConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Glm5NextModel,
    batching=DeepseekV3ModuleV3BatchProcessor,
    tokenizer=GlmTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_glm5_next_state_dict,
    },
    config=Glm5NextConfig,
    memory_planner=Glm5NextMemoryPlanner,
    multi_gpu_supported=True,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    default_structured_output_any_whitespace=True,
    supports_device_graph_capture=True,
)
