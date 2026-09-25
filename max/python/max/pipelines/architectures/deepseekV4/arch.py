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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import DeepseekV4Model
from .model_config import DeepseekV4Config
from .tokenizer import DeepseekV4Tokenizer

deepseekV4_arch = SupportedArchitecture(
    name="DeepseekV4ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V4-Flash-0731",
    ],
    default_encoding=DeepseekV4Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV4Config.SUPPORTED_ENCODINGS,
    # Single-device only for the bringup: the reference implementation's
    # sharding paths all sit behind ``world_size > 1`` and none of them are
    # ported yet.
    multi_gpu_supported=False,
    pipeline_model=DeepseekV4Model,
    tokenizer=DeepseekV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=DeepseekV4Config,
    reasoning_parser="deepseekv4",
    tool_parser="deepseekv4",
)
