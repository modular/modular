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

from . import weight_adapters
from .model import DeepseekV3_2Model
from .model_config import DeepseekV3_2Config

deepseekV3_2_arch = SupportedArchitecture(
    name="DeepseekV3_2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V3.2-Exp",
    ],
    default_encoding="float8_e4m3fn",
    supported_encodings={
        "float8_e4m3fn": ["paged"],
    },
    multi_gpu_supported=True,
    pipeline_model=DeepseekV3_2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3_2Config,
)
