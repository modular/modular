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
"""The MTP layer a GLM-5.3-Flash checkpoint ships beside its trunk."""

from .arch import mtp_glm5_next_draft_arch
from .model import MTPGlm5NextDraftArchConfig

__all__ = ["MTPGlm5NextDraftArchConfig", "mtp_glm5_next_draft_arch"]
