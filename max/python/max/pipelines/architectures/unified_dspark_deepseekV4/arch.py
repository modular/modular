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
from max.pipelines.lib import Speculator

from ..deepseekV4 import weight_adapters
from ..deepseekV4.arch import deepseekV4_arch
from .batch_processor import UnifiedDSparkDeepseekV4BatchProcessor
from .model import UnifiedDSparkDeepseekV4Model

unified_dspark_deepseekV4_speculator = Speculator(
    name="UnifiedDSparkDeepseekV4ForCausalLM",
    base=deepseekV4_arch,
    draft_arch=None,
    # The block-draft method, as for the Gemma4 DSpark drafts.
    method="dflash",
    pipeline_model=UnifiedDSparkDeepseekV4Model,
    batching=UnifiedDSparkDeepseekV4BatchProcessor,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_dspark_safetensor_state_dict,
    },
    opt_out_cascade=True,
    # The base captures its decode step; the spec-decode graph has not been
    # validated under capture yet.
    supports_device_graph_capture=False,
)
