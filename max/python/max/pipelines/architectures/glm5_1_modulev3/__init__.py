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
"""GLM-5.x (GlmMoeDsa) sparse-attention MoE architecture (ModuleV3)."""

from .arch import glm5_1_modulev3_arch, glm5_1_mtp_draft_arch
from .glm5_1 import Glm5_1Decoder
from .model_config import Glm5_1Config
from .mtp import Glm5_1MTP

__all__ = [
    "Glm5_1Config",
    "Glm5_1Decoder",
    "Glm5_1MTP",
    "glm5_1_modulev3_arch",
    "glm5_1_mtp_draft_arch",
]
