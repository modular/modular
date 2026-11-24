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
"""Kimi K2.5 mixture-of-experts architecture for text generation."""

from .arch import (
    kimik2_5_modulev3_arch,
    kimivl_modulev3_arch,
)
from .layers.language_model import KimiK2_5Decoder
from .layers.vision.transformer import KimiK2_5VisionEncoder
from .model_config import KimiK2_5Config, KimiK2_5TextConfig, VisionConfig
from .reasoning import KimiK2_5ReasoningParser
from .tool_parser import KimiToolParser

__all__ = [
    "KimiK2_5Config",
    "KimiK2_5Decoder",
    "KimiK2_5ReasoningParser",
    "KimiK2_5TextConfig",
    "KimiK2_5VisionEncoder",
    "KimiToolParser",
    "VisionConfig",
    "kimik2_5_modulev3_arch",
    "kimivl_modulev3_arch",
]
