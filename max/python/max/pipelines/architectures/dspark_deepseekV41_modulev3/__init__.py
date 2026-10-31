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
"""DeepSeek-V4.1's DSpark stages on ModuleV3, drafting for its ModuleV3 trunk."""

from .arch import dspark_deepseekV41_modulev3_draft_arch
from .model import DSparkDeepseekV41ModuleV3DraftArchConfig

__all__ = [
    "DSparkDeepseekV41ModuleV3DraftArchConfig",
    "dspark_deepseekV41_modulev3_draft_arch",
]
