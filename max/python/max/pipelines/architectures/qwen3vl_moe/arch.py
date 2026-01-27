# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding

from .context import Qwen3VLTextAndVisionContext
from .model import Qwen3VLModel
from .tokenizer import Qwen3VLTokenizer
from .weight_adapters import convert_qwen3vl_model_state_dict

qwen3vl_moe_arch = SupportedArchitecture(
    name="Qwen3VLMoeForConditionalGeneration_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3vl_model_state_dict,
    },
    pipeline_model=Qwen3VLModel,
    tokenizer=Qwen3VLTokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    required_arguments={
        "enable_chunked_prefill": False,
    },
)

# Register the same architecture under Qwen's non-MoE name for models like Qwen3-VL-4B-Instruct
# repo https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
qwen3vl_arch = SupportedArchitecture(
    name="Qwen3VLForConditionalGeneration_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3vl_model_state_dict,
    },
    pipeline_model=Qwen3VLModel,
    tokenizer=Qwen3VLTokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    required_arguments={
        "enable_chunked_prefill": False,
    },
)
