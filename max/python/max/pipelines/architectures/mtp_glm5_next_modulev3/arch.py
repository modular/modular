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

from dataclasses import replace

from max.pipelines.modeling.types import InputModality

from ..glm5_next_modulev3.weight_adapters import glm5_next_mtp_weight_rules
from ..mtp_glm5_next.arch import mtp_glm5_next_draft_arch
from .model import MTPGlm5NextModuleV3DraftArchConfig

__all__ = ["mtp_glm5_next_modulev3_draft_arch"]

# The graph-API draft's checkpoint and capture support, drafting for the
# text-only ModuleV3 target.
mtp_glm5_next_modulev3_draft_arch = replace(
    mtp_glm5_next_draft_arch,
    name="MTPGlm5NextDraftModel_ModuleV3",
    input_modalities={InputModality.TEXT},
    weight_rules=glm5_next_mtp_weight_rules,
    use_weight_rules=True,
    config=MTPGlm5NextModuleV3DraftArchConfig,
)
