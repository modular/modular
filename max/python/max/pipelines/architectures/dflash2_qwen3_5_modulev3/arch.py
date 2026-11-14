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
# max/python/max/pipelines/architectures/dflash2_qwen3_5/arch.py
# at main 4df8111afba.
"""Registers the DFlash2 drafter of a Qwen3.5 target on ModuleV3."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphRole,
    SupportedArchitecture,
    TextTokenizer,
    WeightTransform,
)
from max.pipelines.modeling.types import PipelineTask

from .dflash2_qwen3_5 import DFlash2Qwen3_5
from .model import DFlash2Qwen3_5DraftArchConfig, dflash2_draft_registration
from .model_config import dflash2_draft_width

__all__ = ["dflash2_qwen3_5_modulev3_draft_arch"]

# The draft checkpoint's own architecture. It runs only beside its target,
# and the width it was trained for comes from its checkpoint.
dflash2_qwen3_5_modulev3_draft_arch = SupportedArchitecture(
    name="DFlash2DraftModel_ModuleV3",
    example_repo_ids=["z-lab/Qwen3.8-27B-DFlash2"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    modules={GraphRole.DRAFT: DFlash2Qwen3_5},
    draft=dflash2_draft_registration,
    # The checkpoint names its tensors as the drafter's module does.
    weight_adapters={WeightsFormat.safetensors: WeightTransform(())},
    task=PipelineTask.TEXT_GENERATION,
    config=DFlash2Qwen3_5DraftArchConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
    checkpoint_draft_width=dflash2_draft_width,
)
