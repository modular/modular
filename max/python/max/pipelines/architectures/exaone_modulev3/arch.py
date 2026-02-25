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
    TextTokenizer,
)
from transformers import AutoConfig, PretrainedConfig

from ..llama3_modulev3 import weight_adapters
from ..llama3_modulev3.model import Llama3Model
from ..llama3_modulev3.model_config import Llama3Config
from .weight_adapters import convert_exaone_safetensor_state_dict


class ExaoneConfig(PretrainedConfig):
    """Local config class for EXAONE 3.5 models.

    The ``exaone`` model type is not natively registered in transformers, and the
    remote ``configuration_exaone.py`` shipped in EXAONE 3.5 HuggingFace repos is
    incompatible with the pinned transformers version.  Registering this minimal
    subclass lets ``AutoConfig.from_pretrained`` load the repo's ``config.json``
    without requiring ``trust_remote_code``.
    """

    model_type = "exaone"

    @property
    def num_hidden_layers(self) -> int:
        """Aliases ``num_layers`` to the standard ``num_hidden_layers`` name."""
        return self.num_layers


AutoConfig.register("exaone", ExaoneConfig)

exaone_modulev3_arch = SupportedArchitecture(
    name="ExaoneForCausalLM_ModuleV3",
    default_encoding="float32",
    task=PipelineTask.TEXT_GENERATION,
    supported_encodings={
        "float32": ["paged"],
        "bfloat16": ["paged"],
    },
    example_repo_ids=[
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    ],
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="neox",
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_exaone_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Llama3Config,
)
