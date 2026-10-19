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

from . import weight_adapters
from .batch_processor import HYV4BatchProcessor
from .memory_planner import HYV4MemoryPlanner
from .model import HYV4Model
from .model_config import HYV4Config
from .reasoning import HYV4ReasoningParser  # noqa: F401  registers "hy_v4"
from .tokenizer import HYV4Tokenizer
from .tool_parser import HYV4ToolParser  # noqa: F401  registers "hy_v4"

hy_v4_arch = SupportedArchitecture(
    name="HYV4ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["tencent/Hy4-preview-FP8"],
    default_encoding=HYV4Config.DEFAULT_ENCODING,
    supported_encodings=HYV4Config.SUPPORTED_ENCODINGS,
    pipeline_model=HYV4Model,
    tokenizer=HYV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    requires_max_batch_context_length=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=HYV4Config,
    batching=HYV4BatchProcessor,
    memory_planner=HYV4MemoryPlanner,
    tool_parser="hy_v4",
    reasoning_parser="hy_v4",
)
