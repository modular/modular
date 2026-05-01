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

from .model import LFM2Model
from .model_config import LFM2Config
from .weight_adapters import convert_lfm2_safetensor_state_dict

lfm2_arch = SupportedArchitecture(
    name="Lfm2ForCausalLM",
    default_encoding="float32",
    task=PipelineTask.TEXT_GENERATION,
    supported_encodings={"float32", "bfloat16"},
    example_repo_ids=["LiquidAI/LFM2.5-350M-Base"],
    pipeline_model=LFM2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="neox",
    default_weights_format=WeightsFormat.safetensors,
    required_arguments={
        "allow_safetensors_weights_fp32_bf6_bidirectional_cast": True,
        "trust_remote_code": True,
    },
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_lfm2_safetensor_state_dict,
    },
    config=LFM2Config,
)
