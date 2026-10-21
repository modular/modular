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

from max.graph.weights import WeightsFormat

from ..deepseekV3_modulev3.memory_planner import (
    DeepseekV3ModuleV3MemoryPlanner,
)
from ..unified_mtp_glm5_2.arch import unified_mtp_glm5_2_arch
from .model import UnifiedMTPGlm5_2ModuleV3Model
from .weight_adapters import convert_with_mtp_state_dict

# Everything but the model matches the graph-API MTP architecture: the same
# tokenizer, parsers and context.
unified_mtp_glm5_2_modulev3_arch = replace(
    unified_mtp_glm5_2_arch,
    name="UnifiedMTPGlmMoeDsaForCausalLM_ModuleV3",
    pipeline_model=UnifiedMTPGlm5_2ModuleV3Model,
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
    # The ModuleV3 target never fuses the combine send, so the planner must
    # keep the FFN-output reserve.
    memory_planner=DeepseekV3ModuleV3MemoryPlanner,
)
