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
"""DeepSeek-V4-Flash on ModuleV3.

Selected over the graph-API architecture by ``prefer_module_v3``; both read
the same checkpoint and the same cache tree.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .memory_planner import DeepseekV4MemoryPlanner
from .model import DeepseekV4Model
from .model_config import DeepseekV4Config
from .reasoning import (  # noqa: F401  registers "deepseekv4_modulev3"
    DeepseekV4ReasoningParser,
)
from .tokenizer import DeepseekV4Tokenizer
from .tool_parser import (  # noqa: F401  registers "deepseekv4_modulev3"
    DeepseekV4ToolParser,
)

__all__ = ["deepseekV4_modulev3_arch"]

deepseekV4_modulev3_arch = SupportedArchitecture(
    name="DeepseekV4ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["deepseek-ai/DeepSeek-V4-Flash-0731"],
    default_encoding=DeepseekV4Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV4Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    # A pipeline model, not a registered DECODER: Mach serving pins this
    # graph's name and input order. Moving to registration is a follow-up.
    pipeline_model=DeepseekV4Model,
    tokenizer=DeepseekV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=DeepseekV4Config,
    memory_planner=DeepseekV4MemoryPlanner,
    reasoning_parser="deepseekv4_modulev3",
    tool_parser="deepseekv4_modulev3",
)
