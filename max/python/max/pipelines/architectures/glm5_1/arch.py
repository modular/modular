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
from max.pipelines.architectures.deepseekV3_2 import weight_adapters
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import Glm5_1Model
from .model_config import Glm5_1Config

glm5_1_arch = SupportedArchitecture(
    name="GlmMoeDsaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.1-FP8",
        "zai-org/GLM-5",
    ],
    default_encoding="float8_e4m3fn",
    supported_encodings={
        "float4_e2m1fnx2",
        "float8_e4m3fn",
        "bfloat16",
    },
    multi_gpu_supported=True,
    pipeline_model=Glm5_1Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=Glm5_1Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
