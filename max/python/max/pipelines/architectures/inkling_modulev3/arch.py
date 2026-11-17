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
"""Inkling ModuleV3 architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..inkling.reasoning import (
    InklingReasoningParser,  # noqa: F401  registers "inkling"
)
from ..inkling.tokenizer import InklingTokenizer
from ..inkling.tool_parser import (
    InklingToolParser,  # noqa: F401  registers "inkling"
)
from .model_config import InklingConfig
from .modules import InklingDecoder, InklingVision
from .registered_batch_processor import InklingModuleV3BatchProcessor
from .weight_adapters import inkling_modulev3_weight_rules

inkling_modulev3_arch = SupportedArchitecture(
    name="InklingForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["thinkingmachines/Inkling"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=InklingConfig.DEFAULT_ENCODING,
    supported_encodings=InklingConfig.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: InklingDecoder,
        GraphRole.VISION: InklingVision,
    },
    batching=InklingModuleV3BatchProcessor,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    config=InklingConfig,
    tool_parser="inkling",
    weight_rules=inkling_modulev3_weight_rules,
    use_weight_rules=True,
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    checkpoints_recurrent_state=True,
    reasoning_parser="inkling",
)
