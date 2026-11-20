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
"""Qwen2 transformer architecture for text generation on ModuleV3."""

from .arch import qwen2_modulev3_arch
from .model_config import Qwen2Config
from .qwen2 import Qwen2

__all__ = [
    "Qwen2",
    "Qwen2Config",
    "qwen2_modulev3_arch",
]
