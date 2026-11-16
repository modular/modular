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

"""Supported architecture entry for the ModuleV3 dSpark V4.1 drafter."""

from dataclasses import replace

from max.graph.weights import WeightsFormat
from max.pipelines.lib import GraphRole

from ..deepseekV41_modulev3.model_config import dspark_draft_width
from ..dspark_deepseekV41.arch import dspark_deepseekV41_draft_arch
from ..dspark_deepseekV41.model import DSparkDeepseekV41DraftArchConfigBase
from .model import (
    DSparkDeepseekV41Draft,
    convert_dspark_modulev3_safetensor_state_dict,
    dspark_draft_registration,
)

# The same stages as the graph-API draft, drafting for a ModuleV3
# DeepSeek-V4.1 trunk, which has no device graph capture. They are a
# registered draft, named as the ModuleV3 stages name their weights.
dspark_deepseekV41_modulev3_draft_arch = replace(
    dspark_deepseekV41_draft_arch,
    name="DSparkDeepseekV41DraftModel_ModuleV3",
    config=DSparkDeepseekV41DraftArchConfigBase,
    modules={GraphRole.DRAFT: DSparkDeepseekV41Draft},
    draft=dspark_draft_registration,
    weight_adapters={
        WeightsFormat.safetensors: convert_dspark_modulev3_safetensor_state_dict,
    },
    checkpoint_draft_width=dspark_draft_width,
    supports_device_graph_capture=False,
)
