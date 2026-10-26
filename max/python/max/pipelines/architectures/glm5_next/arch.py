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
"""Architecture registration for GLM-5.3-Flash (``glm5_next``).

Text, images and video. The decoder is assembled in
:meth:`~.model.Glm5NextModel.graph_specs`, which calls
``attach_layers(build_layers())``, so this resolves to GLM-5.3-Flash's own
graph rather than to DeepSeek-V3.2's. The vision tower compiles beside it and
its rows are scattered onto the image and video-frame placeholders after the
embedding.

A load failure is a real failure, not bring-up noise.

The architecture name is the checkpoint's, not a MAX invention:
``zai-org/GLM-5.3-Flash`` declares ``architectures:
["Glm5NextForConditionalGeneration"]`` and ``model_type: "glm5_next"``, which
is what config detection keys on.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.glm5_1.reasoning import (  # noqa: F401  registers "glm45"
    GlmReasoningParser,
)
from max.pipelines.architectures.glm5_1.tool_parser import (  # noqa: F401  registers "glm45"
    GlmToolParser,
)
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Glm5NextBatchProcessor
from .context import Glm5NextTextAndVisionContext
from .memory_planner import Glm5NextMemoryPlanner
from .model import Glm5NextModel
from .model_config import Glm5NextConfig
from .tokenizer import Glm5NextTokenizer
from .weight_adapters import convert_glm5_next_state_dict

__all__ = ["glm5_next_arch"]

glm5_next_arch = SupportedArchitecture(
    name="Glm5NextForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "zai-org/GLM-5.3-Flash",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Glm5NextConfig.DEFAULT_ENCODING,
    supported_encodings=Glm5NextConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Glm5NextModel,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    batching=Glm5NextBatchProcessor,
    tokenizer=Glm5NextTokenizer,
    context_type=Glm5NextTextAndVisionContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_glm5_next_state_dict,
    },
    config=Glm5NextConfig,
    memory_planner=Glm5NextMemoryPlanner,
    multi_gpu_supported=True,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    # The wire format is the GLM-5.x family's, unchanged: `<think>` reasoning
    # and flat `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>`
    # blocks with no outer JSON. The reference engines select
    # `--reasoning-parser glm45` and `--tool-call-parser glm47`; a competitor
    # renaming a parser is not evidence the format changed.
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    # Inherited from GLM-5.2 for the same measured reason: under the compact
    # grammar GLM's content-bearing continuations are masked at the first array
    # decision and the schema's shortest terminator wins.
    default_structured_output_any_whitespace=True,
    # The KDA state is a leaf of the paged cache tree rather than a pool the
    # model owns, so releasing the warmup batch's cache entries releases it
    # too and there is nothing for `SupportsSSMStateWarmup` to do -- the same
    # reason Qwen3.5 captures without implementing it.
    supports_device_graph_capture=True,
)
