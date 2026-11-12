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
"""Supported architecture entry for the Laguna DFlash drafter."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import DFlashLagunaDraftArchConfig
from .model_config import ARCHITECTURE, dflash_laguna_draft_width

__all__ = ["dflash_laguna_modulev3_draft_arch"]

# It has no pipeline model: its arch config builds the drafter for a fused
# spec-decode graph beside a Laguna target.
dflash_laguna_modulev3_draft_arch = SupportedArchitecture(
    name=ARCHITECTURE,
    example_repo_ids=["poolside/Laguna-S-2.1-DFlash-NVFP4"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    task=PipelineTask.TEXT_GENERATION,
    config=DFlashLagunaDraftArchConfig,
    checkpoint_draft_width=dflash_laguna_draft_width,
)
