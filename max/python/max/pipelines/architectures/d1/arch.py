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
"""d1 architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..lfm3_vl.tokenizer import Lfm3VlTokenizer
from .model import D1Model
from .model_config import D1Config

d1_arch = SupportedArchitecture(
    name="D1Model",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["LiquidAI/d1-2B", "LiquidAI/d1-Omni-2B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=D1Config.DEFAULT_ENCODING,
    supported_encodings=D1Config.SUPPORTED_ENCODINGS,
    pipeline_model=D1Model,
    tokenizer=Lfm3VlTokenizer,
    context_type=TextAndVisionContext,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    config=D1Config,
    batching=D1Model.batch_processor_cls,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(10 * 1024**3),
    checkpoints_recurrent_state=True,
    # Segments are read over whole prompts, and a cached prefix saw other
    # segments.
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
