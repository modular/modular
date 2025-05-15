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
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import PipelineTask
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import Gemma3Model

gemma3_arch = SupportedArchitecture(
    name="Gemma3ForCausalLM",
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # TODO(MODELS-487): >=4B models have a slightly different architecture
        # and config and use a different rotary embedding. These will likely
        # need a separate SupportedArchitecture registration.
        # "google/gemma-3-4b-it",
        # "google/gemma-3-4b-pt",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-12b-pt",
        # "google/gemma-3-27b-it",
        # "google/gemma-3-27b-pt",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Gemma3Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
