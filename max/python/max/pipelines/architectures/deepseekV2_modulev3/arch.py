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
"""Supported architecture entry for the ModuleV3 DeepSeek-V2."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import (
    GraphRole,
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .deepseekV2 import DeepseekV2Decoder
from .model_config import DeepseekV2Config

deepseekV2_modulev3_arch = SupportedArchitecture(
    name="DeepseekV2ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
    ],
    default_encoding=DeepseekV2Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV2Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=False,
    modules={GraphRole.DECODER: DeepseekV2Decoder},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    requires_max_batch_context_length=True,
    config=DeepseekV2Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
