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
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import Gemma3Model
from .model_config import Gemma3Config

gemma3_arch = SupportedArchitecture(
    name="Gemma3ForCausalLM_Legacy",
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # We have a different architecture for >=4B models. See gemma3multimodal
        # for more information.
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: ["paged"],
    },
    pipeline_model=Gemma3Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Gemma3Config,
)
