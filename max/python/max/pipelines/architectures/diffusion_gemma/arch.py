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
"""DiffusionGemma, served by the block-diffusion pipeline. Text input only."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .batch_processor import DiffusionGemmaBatchProcessor
from .context import Gemma4Context
from .diffusion_gemma import (
    DiffusionGemmaDenoiser,
    DiffusionGemmaEncoder,
    DiffusionGemmaVision,
)
from .model_config import DiffusionGemmaForBlockDiffusionConfig
from .tokenizer import Gemma4Tokenizer

diffusion_gemma_arch = SupportedArchitecture(
    name="DiffusionGemmaForBlockDiffusion",
    example_repo_ids=[
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "google/diffusiongemma-26B-A4B-it",
    ],
    default_encoding=DiffusionGemmaForBlockDiffusionConfig.DEFAULT_ENCODING,
    supported_encodings=(
        DiffusionGemmaForBlockDiffusionConfig.SUPPORTED_ENCODINGS
    ),
    modules={
        GraphRole.DECODER: DiffusionGemmaEncoder,
        GraphRole.VISION: DiffusionGemmaVision,
        GraphRole.BLOCK_DIFFUSION: DiffusionGemmaDenoiser,
    },
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    input_modalities={InputModality.TEXT},
    context_type=Gemma4Context,
    config=DiffusionGemmaForBlockDiffusionConfig,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    # The decoder writes canvas K/V into uncommitted cache slots and reads
    # them back within a step; prefix-cache block reuse across requests is
    # not yet audited against that pattern.
    required_arguments={"enable_prefix_caching": False},
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    # The denoise loop has no fixed-shape decode step to capture.
    supports_device_graph_capture=False,
    batching=DiffusionGemmaBatchProcessor,
)
