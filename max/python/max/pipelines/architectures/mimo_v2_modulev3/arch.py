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

from ..mimo_v2.arch import mimo_v2_arch
from .memory_planner import MiMoV2ModuleV3MemoryPlanner
from .model import MiMoV2ModuleV3BatchProcessor, MiMoV2ModuleV3Model
from .model_config import MiMoV2Config
from .weight_adapters import convert_safetensor_state_dict

# Everything but the pipeline model matches the graph-API MiMo architecture:
# the same tokenizer, parsers, context and memory planning. The batch
# processor must be replaced too: registration binds `batching` onto the
# pipeline model, which would otherwise keep the graph-API one's
# per-device-row-offsets, signals ABI. The adapter is this package's own:
# ModuleV3's loader checks the backing buffer's dtype, so the experts' E8M0
# scale bytes are retyped with a buffer view only this copy does. The
# planner is this package's own too: the expert-parallel build allocates
# communication buffers the graph-API MoE never does.
mimo_v2_modulev3_arch = replace(
    mimo_v2_arch,
    name="MiMoV2ForCausalLM_ModuleV3",
    # This package's config: it validates the dense-MLP scale-granule split
    # the V3 build needs at plan time, where the inherited graph-API config
    # would accept a degree the build later rejects.
    config=MiMoV2Config,
    pipeline_model=MiMoV2ModuleV3Model,
    batching=MiMoV2ModuleV3BatchProcessor,
    memory_planner=MiMoV2ModuleV3MemoryPlanner,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
)
