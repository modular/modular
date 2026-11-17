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

Text, images and video. The vision tower compiles beside the decoder, and its
rows are scattered onto the image and video-frame placeholders after the
embedding.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Glm5NextBatchProcessor
from .context import Glm5NextTextAndVisionContext
from .model_config import Glm5NextConfig
from .modules import Glm5NextDecoder
from .reasoning import GlmReasoningParser  # noqa: F401  registers "glm5_next"
from .tokenizer import Glm5NextTokenizer
from .tool_parser import GlmToolParser  # noqa: F401  registers "glm5_next"
from .vision.tower import Glm5NextVisionEncoder
from .weight_adapters import glm5_next_weight_rules

__all__ = ["glm5_next_arch"]

glm5_next_arch = SupportedArchitecture(
    # The checkpoint's name resolves here without ``prefer_module_v3``, as the
    # only build registered for it. The suffix stays because Mach's export
    # takes the ModuleV3 input layout from ``gen_mef --prefer-module-v3``,
    # which refuses an architecture without it.
    name="Glm5NextForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["zai-org/GLM-5.3-Flash"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Glm5NextConfig.DEFAULT_ENCODING,
    supported_encodings=Glm5NextConfig.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: Glm5NextDecoder,
        GraphRole.VISION: Glm5NextVisionEncoder,
    },
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    batching=Glm5NextBatchProcessor,
    tokenizer=Glm5NextTokenizer,
    # Every request uses the vision context, text-only ones included: the
    # pipeline's vision seam asserts one for every request of a
    # vision-capable architecture.
    context_type=Glm5NextTextAndVisionContext,
    weight_rules=glm5_next_weight_rules,
    use_weight_rules=True,
    config=Glm5NextConfig,
    checkpoints_recurrent_state=True,
    # The KDA state checkpoints only at a page boundary, which a step's
    # accepted drafts move past.
    prefix_caching_under_spec=False,
    checkpoint_drafts={"mtp": "MTPGlm5NextDraftModel"},
    # Mach binds the fused graph by this name, the graph-API pairing's.
    spec_decode_graph_names={"mtp": "glm5_next_with_mtp_graph"},
    multi_gpu_supported=True,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    # GLM-5.x's `<think>` reasoning and `<tool_call>` formats, as the "glm45"
    # parsers read them, registered under this package's own name.
    tool_parser="glm5_next",
    reasoning_parser="glm5_next",
    default_structured_output_backend="xgrammar",
    default_structured_output_any_whitespace=True,
    supports_device_graph_capture=True,
)
