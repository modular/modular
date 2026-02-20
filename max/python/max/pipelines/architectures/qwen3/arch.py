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
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)

from ..llama3_legacy import weight_adapters
from .model import Qwen3Model
from .model_config import Qwen3Config
from .weight_adapters import convert_qwen3_moe_state_dict

qwen3_arch = SupportedArchitecture(
    name="Qwen3ForCausalLM_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "float32": ["paged"],
    },
    pipeline_model=Qwen3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Qwen3Config,
    multi_gpu_supported=True,
)

# Qwen3MoE architecture - uses the same model and config as Qwen3,
# but with MoE-specific weight adapter to handle expert weight stacking
qwen3_moe_arch = SupportedArchitecture(
    name="Qwen3MoeForCausalLM_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-30B-A3B-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "float32": ["paged"],
    },
    pipeline_model=Qwen3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_moe_state_dict,
    },
    config=Qwen3Config,
    multi_gpu_supported=True,
)
