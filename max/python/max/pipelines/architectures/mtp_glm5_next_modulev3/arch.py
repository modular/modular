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
"""Architecture registration for the MTP layer of GLM-5.3-Flash on ModuleV3."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..glm5_next_modulev3.model_config import Glm5NextConfig
from ..glm5_next_modulev3.weight_adapters import glm5_next_mtp_weight_rules
from .model import MTPGlm5NextModuleV3DraftArchConfig

__all__ = ["mtp_glm5_next_modulev3_draft_arch"]

# The MTP layer ships in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's. It has no
# pipeline model of its own, and drafts for the text-only ModuleV3 target.
mtp_glm5_next_modulev3_draft_arch = SupportedArchitecture(
    name="MTPGlm5NextDraftModel_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["zai-org/GLM-5.3-Flash"],
    default_encoding=Glm5NextConfig.DEFAULT_ENCODING,
    supported_encodings=Glm5NextConfig.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    input_modalities={InputModality.TEXT},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_rules=glm5_next_mtp_weight_rules,
    use_weight_rules=True,
    config=MTPGlm5NextModuleV3DraftArchConfig,
    # The rollback never sizes a tensor by the accepted count, so a verify
    # step replays under capture like any decode step. Eager, the fused step
    # is launch-bound.
    supports_device_graph_capture=True,
)
