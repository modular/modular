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
"""Inkling2 registration with MAX's architecture registry."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .model_config import Inkling2Config
from .modules import Inkling2Decoder, Inkling2Vision
from .reasoning import (
    InklingReasoningParser,  # noqa: F401  registers "inkling2"
)
from .registered_batch_processor import Inkling2BatchProcessor
from .tokenizer import InklingTokenizer
from .tool_parser import (
    InklingToolParser,  # noqa: F401  registers "inkling2"
)
from .weight_adapters import inkling2_weight_rules

inkling2_arch = SupportedArchitecture(
    name="Inkling2ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["thinkingmachines/Inkling-2-Small"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Inkling2Config.DEFAULT_ENCODING,
    supported_encodings=Inkling2Config.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: Inkling2Decoder,
        GraphRole.VISION: Inkling2Vision,
    },
    batching=Inkling2BatchProcessor,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    config=Inkling2Config,
    tool_parser="inkling2",
    weight_rules=inkling2_weight_rules,
    use_weight_rules=True,
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    checkpoints_recurrent_state=True,
    reasoning_parser="inkling2",
)
