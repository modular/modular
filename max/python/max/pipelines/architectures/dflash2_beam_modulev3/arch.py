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
"""Supported architecture entry for the Beam DFlash2 drafter."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import DFlash2BeamDraftArchConfig
from .model_config import ARCHITECTURE, dflash2_beam_draft_width

__all__ = ["dflash2_beam_modulev3_draft_arch"]

# The draft checkpoint's own architecture. It has no pipeline model: its arch
# config builds the drafter for a fused spec-decode graph beside a Beam
# target.
dflash2_beam_modulev3_draft_arch = SupportedArchitecture(
    name=ARCHITECTURE,
    example_repo_ids=["reflection/Beam-1.0-501B-A23B-DFlash2"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    # Sharded over the target's devices, which every leaf of the KV tree
    # must name.
    multi_gpu_supported=True,
    task=PipelineTask.TEXT_GENERATION,
    config=DFlash2BeamDraftArchConfig,
    checkpoint_draft_width=dflash2_beam_draft_width,
)
