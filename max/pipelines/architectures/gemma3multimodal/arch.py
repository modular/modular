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
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import Gemma3_MultiModalModel

gemma3_multimodal_arch = SupportedArchitecture(
    name="Gemma3ForConditionalGeneration",
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-12b-it",
        "google/gemma-3-12b-pt",
        "google/gemma-3-4b-it",
        "google/gemma-3-4b-pt",
        "google/gemma-3-12b-it",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-it",
        "google/gemma-3-27b-pt",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Gemma3_MultiModalModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
