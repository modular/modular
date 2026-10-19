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

from ..hy_v4.arch import hy_v4_arch
from .batch_processor import UnifiedMTPHYV4BatchProcessor
from .model import UnifiedMTPHYV4Model
from .model_config import UnifiedMTPHYV4Config
from .weight_adapters import convert_with_mtp_state_dict

# The MTP layer ships inside the target checkpoint.
unified_mtp_hy_v4_speculator = Speculator(
    name="UnifiedMTPHYV4ForCausalLM",
    base=hy_v4_arch,
    draft_arch=None,
    method="mtp",
    config=UnifiedMTPHYV4Config,
    pipeline_model=UnifiedMTPHYV4Model,
    batching=UnifiedMTPHYV4BatchProcessor,
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
)
