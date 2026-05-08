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
from max.interfaces import InputModality, PipelineTask
from max.pipelines.lib import SupportedArchitecture
from transformers import AutoConfig, PretrainedConfig

from . import weight_adapters
from .context import KimiK2_5TextAndVisionContext
from .model import KimiK2_5Model
from .model_config import KimiK2_5Config, KimiK2_5TextConfig
from .tokenizer import KimiK2_5VLTokenizer
from .unified_eagle_pipeline_model import Eagle3KimiK25Model


class _KimiK2Config(PretrainedConfig):
    """Minimal config for the ``kimi_k2`` model type.

    The Eagle3 draft checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``)
    declares ``model_type: "kimi_k2"`` which is not natively registered
    in transformers, and ships no ``auto_map``.  Registering this stub
    lets ``AutoConfig.from_pretrained`` succeed without a manual JSON
    fallback.
    """

    model_type = "kimi_k2"


AutoConfig.register("kimi_k2", _KimiK2Config, exist_ok=True)

kimik2_5_arch = SupportedArchitecture(
    name="KimiK25ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Kimi-K2.5-NVFP4",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    },
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    pipeline_model=KimiK2_5Model,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5Config,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
)

kimivl_arch = SupportedArchitecture(
    name="KimiVLForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "moonshotai/Kimi-VL-A3B-Instruct",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    },
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    pipeline_model=KimiK2_5Model,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimivl_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5Config,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
)

eagle3_kimik25_arch = SupportedArchitecture(
    name="Eagle3DeepseekV2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/Kimi-K2.5-NVFP4"],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    },
    multi_gpu_supported=True,
    pipeline_model=Eagle3KimiK25Model,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5TextConfig,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
)
