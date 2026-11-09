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
"""GLM-5.3-Flash on ModuleV3.

Selected over the graph-API architecture by ``prefer_module_v3``; both read
the same checkpoint and the same cache tree.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.glm5_1.reasoning import (  # noqa: F401  registers "glm45"
    GlmReasoningParser,
)
from max.pipelines.architectures.glm5_1.tokenizer import GlmTokenizer
from max.pipelines.architectures.glm5_1.tool_parser import (  # noqa: F401  registers "glm45"
    GlmToolParser,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from .memory_planner import Glm5NextMemoryPlanner
from .model_config import Glm5NextConfig
from .modules import Glm5NextDecoder
from .weight_adapters import glm5_next_weight_rules

__all__ = ["glm5_next_modulev3_arch"]

glm5_next_modulev3_arch = SupportedArchitecture(
    name="Glm5NextForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["zai-org/GLM-5.3-Flash"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Glm5NextConfig.DEFAULT_ENCODING,
    supported_encodings=Glm5NextConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Glm5NextDecoder},
    tokenizer=GlmTokenizer,
    context_type=TextContext,
    weight_rules=glm5_next_weight_rules,
    use_weight_rules=True,
    config=Glm5NextConfig,
    memory_planner=Glm5NextMemoryPlanner,
    checkpoints_recurrent_state=True,
    # The KDA state checkpoints only at a page boundary, which a step's
    # accepted drafts move past.
    prefix_caching_under_spec=False,
    checkpoint_drafts={"mtp": "MTPGlm5NextDraftModel"},
    # The graph-API pairing's name, which engines that bind the fused graph
    # by name already know.
    spec_decode_graph_names={"mtp": "glm5_next_with_mtp_graph"},
    multi_gpu_supported=True,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    default_structured_output_any_whitespace=True,
    supports_device_graph_capture=True,
)
