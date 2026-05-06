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
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.lib import SupportedArchitecture

from .model import Qwen3_5Model
from .model_config import Qwen3_5Config
from .tokenizer import Qwen3_5Tokenizer
from .weight_adapters import convert_qwen3_5_state_dict

qwen3_5_arch = SupportedArchitecture(
    name="Qwen3_5ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3.5-27B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    pipeline_model=Qwen3_5Model,
    tokenizer=Qwen3_5Tokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_5_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,  # TODO: Remove when Deltanet supports prefix caching
    },
    config=Qwen3_5Config,
    multi_gpu_supported=False,
)
