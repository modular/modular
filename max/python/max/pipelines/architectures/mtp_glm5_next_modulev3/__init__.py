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
"""GLM-5.3-Flash's MTP draft for the ModuleV3 target.

It is the MTP layer the checkpoint ships.
"""

from .arch import mtp_glm5_next_modulev3_draft_arch
from .model import MTPGlm5NextModuleV3DraftArchConfig

__all__ = [
    "MTPGlm5NextModuleV3DraftArchConfig",
    "mtp_glm5_next_modulev3_draft_arch",
]
