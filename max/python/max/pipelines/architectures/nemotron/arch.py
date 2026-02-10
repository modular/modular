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
"""Architecture registration for Nemotron (NemotronForCausalLM)."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import NemotronModel
from .model_config import NemotronConfig

nemotron_arch = SupportedArchitecture(
    name="NemotronForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Minitron-4B-Instruct",
        "nvidia/Nemotron-Mini-4B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=NemotronModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=NemotronConfig,
)
