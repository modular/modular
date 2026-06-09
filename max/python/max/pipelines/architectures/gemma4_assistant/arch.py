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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..gemma4.model import Gemma3_MultiModalModel
from ..gemma4.weight_adapters import convert_safetensor_language_state_dict
from .model_config import Gemma4AssistantConfig

gemma4_assistant_arch = SupportedArchitecture(
    name="Gemma4AssistantForCausalLM",
    example_repo_ids=["google/gemma-4-31B-it-assistant"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=Gemma3_MultiModalModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_language_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    multi_gpu_supported=True,
    config=Gemma4AssistantConfig,
)
