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
"""Registers Laguna (``LagunaForCausalLM``) on ModuleV3.

Selected with ``--prefer-module-v3``. Shares the graph architecture's config,
tokenizer, and reasoning and tool parsers; only the model is reimplemented.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.tokenizer import ReasoningTextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import LagunaModuleV3BatchProcessor
from .model import LagunaModel

laguna_modulev3_arch = SupportedArchitecture(
    name="LagunaForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "poolside/Laguna-M.1-NVFP4",
        "poolside/Laguna-S-2.1-NVFP4",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=LagunaConfig.DEFAULT_ENCODING,
    supported_encodings=LagunaConfig.SUPPORTED_ENCODINGS,
    pipeline_model=LagunaModel,
    tokenizer=ReasoningTextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=LagunaConfig,
    batching=LagunaModuleV3BatchProcessor,
    multi_gpu_supported=False,
    reasoning_parser="laguna",
    tool_parser="laguna",
)
