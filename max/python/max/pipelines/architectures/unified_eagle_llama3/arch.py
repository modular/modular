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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer

from . import weight_adapters
from .model import UnifiedEagleLlama3Model
from .model_config import UnifiedEagleLlama3Config

unified_eagle_llama3_arch = SupportedArchitecture(
    name="UnifiedEagleLlama3ForCausalLM",
    example_repo_ids=[
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    pipeline_model=UnifiedEagleLlama3Model,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    rope_type="normal",
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=UnifiedEagleLlama3Config,
)
