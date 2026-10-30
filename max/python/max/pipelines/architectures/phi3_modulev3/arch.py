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
"""Registers Phi3 on ModuleV3 with the registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..llama3.weight_adapters import convert_gguf_state_dict
from ..llama3_modulev3.llama3 import Llama3Decoder
from ..llama3_modulev3.model_config import Llama3DecoderConfig
from .tokenizer import Phi3Tokenizer
from .weight_adapters import convert_safetensor_state_dict

phi3_modulev3_arch = SupportedArchitecture(
    name="Phi3ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["microsoft/phi-4", "microsoft/Phi-3.5-mini-instruct"],
    default_weights_format=WeightsFormat.gguf,
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    modules={GraphRole.DECODER: Llama3Decoder},
    tokenizer=Phi3Tokenizer,
    context_type=TextContext,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
        WeightsFormat.gguf: convert_gguf_state_dict,
    },
    config=Llama3DecoderConfig,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
