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

unified_dflash_mimo_v2_arch = replace(
    mimo_v2_arch,
    name="UnifiedDflashMiMoV2ForCausalLM",
    pipeline_model=UnifiedDflashMiMoV2Model,
    config=UnifiedDflashMiMoV2Config,
    batching=UnifiedDflashMiMoV2BatchProcessor,
    checkpoint_draft_width=mimo_dflash_draft_width,
    memory_planner=MiMoV2DFlashMemoryPlanner,
    # Prefill rows commit their whole chunk and verify nothing, so a batch
    # may mix them with decode rows.
    supports_spec_decode_mixed_batches=True,
)

# The base graph with the drafter's context writer, for an engine that runs it
# beside the fused graph. Never chosen by checkpoint; an export names it.
mimo_v2_dflash_context_arch = replace(
    mimo_v2_arch,
    name="MiMoV2DFlashContextForCausalLM",
    pipeline_model=MiMoV2DFlashContextModel,
    config=MiMoV2DFlashContextConfig,
    memory_planner=MiMoV2DFlashMemoryPlanner,
)
