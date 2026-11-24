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
"""Registers Qwen3-VL and Qwen3-VL-MoE."""

from __future__ import annotations

import dataclasses

from max.graph.weights import WeightsFormat
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Qwen3VLBatchProcessor
from .context import Qwen3VLTextAndVisionContext
from .decoder import Qwen3VLDecoder
from .model_config import Qwen3VLConfig
from .tokenizer import Qwen3VLTokenizer
from .vision import Qwen3VLVisionEncoder
from .weight_adapters import qwen3vl_weight_rules

__all__ = ["qwen3vl_arch", "qwen3vl_moe_arch"]


qwen3vl_moe_arch = SupportedArchitecture(
    name="Qwen3VLMoeForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-VL-30B-A3B-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    default_encoding=Qwen3VLConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen3VLConfig.SUPPORTED_ENCODINGS,
    weight_rules=qwen3vl_weight_rules,
    use_weight_rules=True,
    modules={
        GraphRole.DECODER: Qwen3VLDecoder,
        GraphRole.VISION: Qwen3VLVisionEncoder,
    },
    batching=Qwen3VLBatchProcessor,
    tokenizer=Qwen3VLTokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    config=Qwen3VLConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

# The dense checkpoints share the decoder, which routes a layer to experts
# only when the config gives it any.
qwen3vl_arch = dataclasses.replace(
    qwen3vl_moe_arch,
    name="Qwen3VLForConditionalGeneration",
    example_repo_ids=["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"],
)
