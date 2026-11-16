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
"""Qwen3.5 on ModuleV3.

A hybrid model interleaving Gated DeltaNet linear attention with full
attention, plus a vision tower.
"""

from .arch import qwen3_5_modulev3_arch, qwen3_5_text_modulev3_arch
from .model_config import Qwen3_5Config, VisionConfig
from .qwen3_5 import Qwen3_5Decoder, Qwen3_5TextModel

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5Decoder",
    "Qwen3_5TextModel",
    "VisionConfig",
    "qwen3_5_modulev3_arch",
    "qwen3_5_text_modulev3_arch",
]
