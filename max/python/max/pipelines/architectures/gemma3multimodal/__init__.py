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
"""Gemma 3 vision-language architecture for multimodal text generation."""

from .arch import gemma3_multimodal_arch
from .model import Gemma3_MultiModalModel, Gemma3MultiModalModelInputs
from .model_config import (
    Gemma3ForConditionalGenerationConfig,
    Gemma3VisionConfig,
)

__all__ = [
    "Gemma3ForConditionalGenerationConfig",
    "Gemma3MultiModalModelInputs",
    "Gemma3VisionConfig",
    "Gemma3_MultiModalModel",
    "gemma3_multimodal_arch",
]
