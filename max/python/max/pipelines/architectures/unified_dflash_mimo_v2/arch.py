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

from dataclasses import replace

from max.pipelines.lib import Speculator

from ..mimo_v2.arch import mimo_v2_arch
from .base_ctx import MiMoV2DFlashContextModel
from .batch_processor import UnifiedDflashMiMoV2BatchProcessor
from .memory_planner import MiMoV2DFlashMemoryPlanner
from .model import UnifiedDflashMiMoV2Model
from .model_config import (
    MiMoV2DFlashContextConfig,
    UnifiedDflashMiMoV2Config,
    mimo_dflash_draft_width,
)

unified_dflash_mimo_v2_speculator = Speculator(
    name="UnifiedDflashMiMoV2ForCausalLM",
    base=mimo_v2_arch,
    draft_arch="DFlashDraftModel",
    method="dflash",
    pipeline_model=UnifiedDflashMiMoV2Model,
    batching=UnifiedDflashMiMoV2BatchProcessor,
    config=UnifiedDflashMiMoV2Config,
    memory_planner=MiMoV2DFlashMemoryPlanner,
    checkpoint_draft_width=mimo_dflash_draft_width,
    # Prefill rows commit their whole chunk and verify nothing, so a batch
    # may mix them with decode rows.
    supports_spec_decode_mixed_batches=True,
)

# Construction renames a DFlash draft to LlamaForCausalLM before selecting; a
# worker process reloads the checkpoint's own name.
unified_dflash_mimo_v2_renamed_draft_speculator = replace(
    unified_dflash_mimo_v2_speculator, draft_arch="LlamaForCausalLM"
)

mimo_v2_dflash_context_arch = replace(
    mimo_v2_arch,
    name="MiMoV2DFlashContextForCausalLM",
    pipeline_model=MiMoV2DFlashContextModel,
    config=MiMoV2DFlashContextConfig,
    memory_planner=MiMoV2DFlashMemoryPlanner,
)
