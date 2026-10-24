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

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .drafter import DFlashMiMoV2DraftArchConfig, mimo_dflash_draft_width

__all__ = ["dflash_mimo_v2_draft_arch"]

# The drafter's own config.json declares the generic ["DFlashDraftModel"],
# which names the Llama 3 DFlash draft. Config resolution renames a DFlash
# draft served beside a MiMo-V2 target to this architecture, since the
# drafter is sharded like its target and reads a weight format of its own.
# It has no pipeline model of its own.
dflash_mimo_v2_draft_arch = SupportedArchitecture(
    name="DFlashMiMoV2DraftModel",
    # The drafter ships in the target checkpoint's dflash/ folder.
    example_repo_ids=["XiaomiMiMo/MiMo-V2.6-Flash-RL"],
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
    config=DFlashMiMoV2DraftArchConfig,
    checkpoint_draft_width=mimo_dflash_draft_width,
    # MiMo-V2's own graph doesn't support device graph capture either.
    supports_device_graph_capture=False,
)
