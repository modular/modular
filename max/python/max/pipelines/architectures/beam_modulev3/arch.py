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
"""Registers Beam (``BeamForCausalLM``) on ModuleV3."""

from pathlib import Path

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from .beam import Beam
from .memory_planner import BeamMemoryPlanner
from .model_config import BeamConfig
from .reasoning import BeamReasoningParser  # noqa: F401  registers "beam"
from .tokenizer import BeamTokenizer
from .tool_parser import BeamToolParser  # noqa: F401  registers "beam"
from .weight_adapters import beam_weight_rules

beam_modulev3_arch = SupportedArchitecture(
    name="BeamForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["reflection/Beam-1.0-501B-A23B-NVFP4"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=BeamConfig.DEFAULT_ENCODING,
    supported_encodings=BeamConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Beam},
    tokenizer=BeamTokenizer,
    reasoning_parser="beam",
    tool_parser="beam",
    context_type=TextContext,
    weight_rules=beam_weight_rules,
    use_weight_rules=True,
    config=BeamConfig,
    memory_planner=BeamMemoryPlanner,
    multi_gpu_supported=True,
    format_spec=Path(__file__).parent / "formatspec.json",
    # The name Mach finds the fused DFlash2 graph by.
    spec_decode_graph_names={"dflash2": "unified_dflash2_beam"},
)
