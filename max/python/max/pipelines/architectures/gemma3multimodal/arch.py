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

import os

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextAndVisionTokenizer,
    TextTokenizer,
)

ENABLE_NEW_IMPL = os.environ.get("MAX_ENABLE_GEMMA3_VISION", "0").lower() in (
    "1",
    "true",
)

if ENABLE_NEW_IMPL:
    from .model import Gemma3_MultiModalModel
else:
    from . import weight_adapters_legacy as weight_adapters
    from .model_legacy import Gemma3_MultiModalModelLegacy


example_repo_ids = [
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
]

if ENABLE_NEW_IMPL:
    gemma3_multimodal_arch = SupportedArchitecture(
        name="Gemma3ForConditionalGeneration",
        example_repo_ids=example_repo_ids,
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
            SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
        },
        pipeline_model=Gemma3_MultiModalModel,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextAndVisionTokenizer,
        default_weights_format=WeightsFormat.safetensors,
        multi_gpu_supported=True,
        rope_type=RopeType.normal,
        required_arguments={
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
    )
else:
    gemma3_multimodal_arch = SupportedArchitecture(
        name="Gemma3ForConditionalGeneration",
        example_repo_ids=example_repo_ids,
        default_encoding=SupportedEncoding.bfloat16,
        supported_encodings={
            SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
            SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
        },
        pipeline_model=Gemma3_MultiModalModelLegacy,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        default_weights_format=WeightsFormat.safetensors,
        multi_gpu_supported=True,
        rope_type=RopeType.normal,
        weight_adapters={
            WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        },
    )
