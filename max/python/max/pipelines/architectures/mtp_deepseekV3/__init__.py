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
"""The NextN head a DeepseekV3 checkpoint ships beside its trunk."""

from .arch import mtp_deepseekV3_draft_arch
from .model import MTPDeepseekV3DraftArchConfig
from .weight_adapters import convert_mtp_safetensor_state_dict

__all__ = [
    "MTPDeepseekV3DraftArchConfig",
    "convert_mtp_safetensor_state_dict",
    "mtp_deepseekV3_draft_arch",
]
