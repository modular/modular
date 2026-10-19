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

from ..unified_mtp_gemma4.arch import unified_mtp_gemma4_arch
from .batch_processor import UnifiedMTPGemma4ModuleV3BatchProcessor
from .model import UnifiedMTPGemma4ModuleV3Model
from .weight_adapters import convert_safetensor_state_dict

# Everything but the model and batch processor matches the graph-API MTP
# architecture: the same tokenizer, memory planner and context.
unified_mtp_gemma4_modulev3_arch = replace(
    unified_mtp_gemma4_arch,
    name="UnifiedMTPGemma4ForCausalLM_ModuleV3",
    pipeline_model=UnifiedMTPGemma4ModuleV3Model,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    batching=UnifiedMTPGemma4ModuleV3BatchProcessor,
)
