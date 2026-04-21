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
"""DeepSeek-V3 NextN multi-token prediction draft model for speculative decoding."""

from .arch import deepseekV3_nextn_arch
from .model import DeepseekV3NextNModel
from .model_config import DeepseekV3NextNConfig

__all__ = [
    "DeepseekV3NextNConfig",
    "DeepseekV3NextNModel",
    "deepseekV3_nextn_arch",
]
