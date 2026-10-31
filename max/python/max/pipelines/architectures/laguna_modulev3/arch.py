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

Selected with ``--prefer-module-v3``. Shares the graph architecture's
tokenizer and reasoning and tool parsers, and extends its config; only the
model is reimplemented, and registered as the ``DECODER`` module.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.laguna.arch import laguna_format_spec
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.lib.tokenizer import ReasoningTextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .laguna import Laguna
from .model_config import LagunaDecoderConfig

laguna_modulev3_arch = SupportedArchitecture(
    name="LagunaForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "poolside/Laguna-M.1-NVFP4",
        "poolside/Laguna-S-2.1-NVFP4",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=LagunaDecoderConfig.DEFAULT_ENCODING,
    supported_encodings=LagunaDecoderConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Laguna},
    tokenizer=ReasoningTextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=LagunaDecoderConfig,
    multi_gpu_supported=False,
    reasoning_parser="laguna",
    tool_parser="laguna",
    format_spec=laguna_format_spec,
)
