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
"""The MTP head a Qwen3.5 checkpoint ships beside its trunk."""

from .arch import mtp_qwen3_5_draft_arch
from .model import MTPQwen3_5DraftArchConfig

__all__ = [
    "MTPQwen3_5DraftArchConfig",
    "mtp_qwen3_5_draft_arch",
]
