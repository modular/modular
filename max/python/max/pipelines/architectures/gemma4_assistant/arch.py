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
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..gemma4.tokenizer import Gemma4Tokenizer
from .model import Gemma4AssistantDraftArchConfig
from .model_config import Gemma4AssistantConfig
from .weight_adapters import convert_safetensor_state_dict

# Gemma 4's assistant drafts for an MTP target. It has no
# pipeline model of its own.
gemma4_assistant_arch = SupportedArchitecture(
    name="Gemma4AssistantForCausalLM",
    example_repo_ids=[
        "google/gemma-4-31B-it-assistant",
        "google/gemma-4-26B-A4B-it-assistant",
    ],
    default_encoding=Gemma4AssistantConfig.DEFAULT_ENCODING,
    supported_encodings=Gemma4AssistantConfig.SUPPORTED_ENCODINGS,
    pipeline_model=None,
    tokenizer=Gemma4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    multi_gpu_supported=True,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    config=Gemma4AssistantDraftArchConfig,
    # The assistant reads the host's row offsets, which only the distributed
    # signature declares. Its pairing encodes images eagerly, so it isn't
    # captured.
    supports_device_graph_capture=False,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
)
