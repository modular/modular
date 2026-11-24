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
"""Registers the Inkling MTP draft for the ModuleV3 target."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..inkling.model_config import InklingConfig
from ..inkling.tokenizer import InklingTokenizer
from ..mtp_inkling.model_config import (
    MTPInklingDraftArchConfig,
    inkling_mtp_draft_width,
)
from ..mtp_inkling.weight_adapters import inkling_mtp_weight_rules
from .draft import InklingMTPDraft, mtp_draft_registration

__all__ = ["mtp_inkling_modulev3_draft_arch"]

# The MTP depths ship in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's.
mtp_inkling_modulev3_draft_arch = SupportedArchitecture(
    name="MTPInklingDraftModel_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["thinkingmachines/Inkling"],
    default_encoding=InklingConfig.DEFAULT_ENCODING,
    supported_encodings=InklingConfig.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    modules={GraphRole.DRAFT: InklingMTPDraft},
    draft=mtp_draft_registration,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_rules=inkling_mtp_weight_rules,
    use_weight_rules=True,
    config=MTPInklingDraftArchConfig,
    checkpoint_draft_width=inkling_mtp_draft_width,
)
