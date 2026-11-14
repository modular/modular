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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/mtp_qwen3_5/arch.py
# at main 4df8111afba.
"""Registers the Qwen3.5 MTP draft on ModuleV3."""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.qwen3_5_modulev3.model_config import (
    Qwen3_5Config,
)
from max.pipelines.architectures.qwen3_5_modulev3.weight_adapters import (
    QWEN3_5_MTP_WEIGHT_RULES,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    SupportedArchitecture,
    TextTokenizer,
    WeightTransform,
)
from max.pipelines.modeling.types import PipelineTask

from .model import MTPQwen3_5DraftArchConfig, mtp_draft_registration
from .mtp import Qwen3_5MTP

__all__ = ["mtp_qwen3_5_modulev3_draft_arch"]

# The head ships in the target's checkpoint, which has no config of its own,
# so spec decoding names this architecture from the target's.
mtp_qwen3_5_modulev3_draft_arch = SupportedArchitecture(
    name="MTPQwen3_5DraftModel_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["RadixArk/Qwen3.8-27B-NVFP4"],
    default_encoding=Qwen3_5Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    modules={GraphRole.DRAFT: Qwen3_5MTP},
    draft=mtp_draft_registration,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: WeightTransform(QWEN3_5_MTP_WEIGHT_RULES),
    },
    config=MTPQwen3_5DraftArchConfig,
    supports_device_graph_capture=False,
)
