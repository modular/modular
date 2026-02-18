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
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from ..llama3_legacy import weight_adapters
from ..llama3_legacy.model import Llama3Model
from ..llama3_legacy.model_config import Llama3Config

granite_arch = SupportedArchitecture(
    name="GraniteForCausalLM_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "ibm-granite/granite-3.1-8b-instruct",
        "ibm-granite/granite-3.1-8b-base",
    ],
    default_weights_format=WeightsFormat.gguf,
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: ["paged"],
        SupportedEncoding.bfloat16: ["paged"],
    },
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Llama3Config,
)
