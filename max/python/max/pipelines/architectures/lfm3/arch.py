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
"""LFM3 architecture registrations."""

from __future__ import annotations

from pathlib import Path

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .lfm3 import Lfm3
from .model_config import Lfm3Config

lfm3_arch = SupportedArchitecture(
    name="Lfm3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["LiquidAI/LFM3-1.7B", "LiquidAI/LFM3-1.7B-Base"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Lfm3Config.DEFAULT_ENCODING,
    supported_encodings=Lfm3Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Lfm3},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    config=Lfm3Config,
    memory_planner=PagedMemoryPlanner,
    checkpoints_recurrent_state=True,
    format_spec=Path(__file__).parent / "formatspec.json",
)
