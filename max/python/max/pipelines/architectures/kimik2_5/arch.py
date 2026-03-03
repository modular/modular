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
from .model import KimiK2_5Model
from .model_config import KimiK2_5Config

kimik2_5_arch = SupportedArchitecture(
    name="KimiK25ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-VL-A3B-Instruct",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    },
    multi_gpu_supported=True,
    pipeline_model=KimiK2_5Model,  # type: ignore[type-abstract]
    tokenizer=TextTokenizer,  # KimiK2_5VLTokenizer,
    context_type=TextContext,  # KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5Config,
)
