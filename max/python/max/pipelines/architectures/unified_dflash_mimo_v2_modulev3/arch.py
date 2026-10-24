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

from max.pipelines.lib import Speculator

from ..mimo_v2_modulev3.arch import mimo_v2_modulev3_arch
from .batch_processor import UnifiedDflashMiMoV2ModuleV3BatchProcessor
from .memory_planner import UnifiedDflashMiMoV2ModuleV3MemoryPlanner
from .model import UnifiedDflashMiMoV2ModuleV3Model
from .model_config import (
    UnifiedDflashMiMoV2Config,
    mimo_dflash_draft_width,
)

# Everything but the pipeline model matches the graph-API DFlash speculator:
# the same tokenizer, parsers, weight adapter (inherited from the ModuleV3
# base, whose E8M0 scale view the fused target's experts load through), draft
# checkpoint names and context. The fused graph is the same input tree under
# the same name, so Mach's block-spec executor serves either export.
unified_dflash_mimo_v2_modulev3_speculator = Speculator(
    name="UnifiedDflashMiMoV2ForCausalLM_ModuleV3",
    base=mimo_v2_modulev3_arch,
    # The target's ``draft_architectures`` map renames the checkpoint's
    # ``DFlashDraftModel`` to the registered MiMo draft placeholder before
    # selection runs, so this names the rename target.
    draft_arch="DFlashMiMoV2DraftModel",
    method="dflash",
    pipeline_model=UnifiedDflashMiMoV2ModuleV3Model,
    batching=UnifiedDflashMiMoV2ModuleV3BatchProcessor,
    config=UnifiedDflashMiMoV2Config,
    # The base planner's EP reservation undersizes the verify rows the
    # fused graph merges past the scheduler's active-token budget.
    memory_planner=UnifiedDflashMiMoV2ModuleV3MemoryPlanner,
    checkpoint_draft_width=mimo_dflash_draft_width,
    # Prefill rows commit their whole chunk and verify nothing, so a batch
    # may mix them with decode rows.
    supports_spec_decode_mixed_batches=True,
)

# The same fused architecture as a registry entry, not just a speculator
# derivation: main's convention is that a nameable fused graph -- one a
# gen_mef claim or the engine's block-spec executor keys by name -- is
# findable under that name, as GLM-5.2's ModuleV3 MTP graph is. The
# speculator above stays the selection mechanism; this entry makes the
# derived architecture's own name resolvable.
unified_dflash_mimo_v2_modulev3_arch = (
    unified_dflash_mimo_v2_modulev3_speculator.derive()
)
