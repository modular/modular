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
from max.pipelines.config_enums import RopeType, SupportedEncoding
from max.pipelines.interfaces import PipelineTask
from max.pipelines.kv_cache import KVCacheStrategy
from max.pipelines.registry import SupportedArchitecture
from max.pipelines.tokenizer import TextTokenizer

from . import weight_adapters
from .model import Llama4Model

llama4_arch = SupportedArchitecture(
    name="Llama4ForConditionalGeneration",
    example_repo_ids=[
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Llama4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    # NOTE: Llama 4 interleaves RoPE and NoPE (no positional encodings).
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
