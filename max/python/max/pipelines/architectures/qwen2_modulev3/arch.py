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
"""Registers Qwen2 on ModuleV3 with the pipeline registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model_config import Qwen2Config
from .qwen2 import Qwen2

qwen2_modulev3_arch = SupportedArchitecture(
    name="Qwen2ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/QwQ-32B",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen2Config.DEFAULT_ENCODING,
    supported_encodings=Qwen2Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Qwen2},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Qwen2Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
