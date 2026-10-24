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
"""Registers DeepSeek-V4.1-Flash with its built-in DSpark drafter.

The target model and the drafter, whose weights ship in the same checkpoint,
run in one graph per decoding step, so no separate draft model is loaded.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.lib import Speculator

from ..deepseekV41 import weight_adapters
from ..deepseekV41.arch import deepseekV41_arch
from .batch_processor import UnifiedDSparkDeepseekV41BatchProcessor
from .model import UnifiedDSparkDeepseekV41Model

unified_dspark_deepseekV41_speculator = Speculator(
    name="UnifiedDSparkDeepseekV41ForCausalLM",
    base=deepseekV41_arch,
    draft_arch=None,
    method="dspark",
    pipeline_model=UnifiedDSparkDeepseekV41Model,
    batching=UnifiedDSparkDeepseekV41BatchProcessor,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_dspark_safetensor_state_dict,
    },
    opt_out_cascade=True,
    # One captured graph per (batch size, verification width, cache-length
    # bucket); every input, including the pinned Engram tables, is bound as
    # a buffer so a replay sees fresh values.
    supports_device_graph_capture=True,
    supports_spec_decode_mixed_batches=True,
)
