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
"""SupportedArchitecture registration for OpenELM."""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import OpenELMModel
from .model_config import OpenELMConfig
from .weight_adapters import convert_safetensor_state_dict

openelm_arch = SupportedArchitecture(
    name="OpenELMForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "apple/OpenELM-270M-Instruct",
        "apple/OpenELM-450M-Instruct",
        "apple/OpenELM-1_1B-Instruct",
        "apple/OpenELM-3B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=OpenELMConfig.DEFAULT_ENCODING,
    supported_encodings=OpenELMConfig.SUPPORTED_ENCODINGS,
    pipeline_model=OpenELMModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    config=OpenELMConfig,
    batching=Llama3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
)
