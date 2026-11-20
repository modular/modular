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
"""The DSpark stages DeepSeek-V4's checkpoint ships beside its trunk."""

from .arch import dspark_deepseekV4_draft_arch
from .model import (
    DSparkDeepseekV4DraftArchConfig,
    DSparkDeepseekV4DraftConfig,
    dspark_draft_width,
)

__all__ = [
    "DSparkDeepseekV4DraftArchConfig",
    "DSparkDeepseekV4DraftConfig",
    "dspark_deepseekV4_draft_arch",
    "dspark_draft_width",
]
