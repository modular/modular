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

"""Register the Granite ``ModuleV3`` text-generation architecture."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..llama3_modulev3 import weight_adapters
from ..llama3_modulev3.llama3 import Llama3Decoder
from ..llama3_modulev3.model_config import Llama3DecoderConfig

granite_modulev3_arch = SupportedArchitecture(
    name="GraniteForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "ibm-granite/granite-3.1-8b-instruct",
        "ibm-granite/granite-3.1-8b-base",
    ],
    default_weights_format=WeightsFormat.gguf,
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    modules={GraphRole.DECODER: Llama3Decoder},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Llama3DecoderConfig,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
