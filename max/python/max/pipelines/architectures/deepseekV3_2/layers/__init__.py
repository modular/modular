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

"""DeepseekV3.2 layer modules."""

from .mlp import DeepseekV3_2MLP
from .moe import DeepseekV3_2MoE
from .moe_gate import DeepseekV3_2TopKRouter
from .rms_norm_fused import RMSNormFused
from .transforms import HadamardTransform

__all__ = [
    "DeepseekV3_2MLP",
    "DeepseekV3_2MoE",
    "DeepseekV3_2TopKRouter",
    "HadamardTransform",
    "RMSNormFused",
]
