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

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3_2_modulev3 import weight_adapters
from ..deepseekV3_modulev3.batch_processor import (
    DeepseekV3ModuleV3BatchProcessor,
)
from ..deepseekV3_modulev3.memory_planner import (
    DeepseekV3ModuleV3MemoryPlanner,
)
from ..glm5_1.reasoning import (
    GlmReasoningParser,  # noqa: F401  registers "glm45"
)
from ..glm5_1.tokenizer import GlmTokenizer
from ..glm5_1.tool_parser import GlmToolParser  # noqa: F401  registers "glm45"
from .model import Glm5_1Model
from .model_config import Glm5_1Config

glm5_1_modulev3_arch = SupportedArchitecture(
    name="GlmMoeDsaForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.1-FP8",
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.2-FP8",
        "zai-org/GLM-5",
        "zai-org/GLM-5.3",
    ],
    default_encoding=Glm5_1Config.DEFAULT_ENCODING,
    supported_encodings=Glm5_1Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=Glm5_1Model,
    batching=DeepseekV3ModuleV3BatchProcessor,
    tokenizer=GlmTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=Glm5_1Config,
    memory_planner=DeepseekV3ModuleV3MemoryPlanner,
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    default_structured_output_any_whitespace=True,
)
