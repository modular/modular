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
"""GLM-5.x MTP speculative decoding on ModuleV3."""

from .arch import unified_mtp_glm5_2_modulev3_arch
from .model import UnifiedMTPGlm5_2ModuleV3Model

__all__ = [
    "UnifiedMTPGlm5_2ModuleV3Model",
    "unified_mtp_glm5_2_modulev3_arch",
]
