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
"""Gemma 3 transformer architectures for text and vision-language generation."""

from .arch import gemma3_modulev3_arch, gemma3_multimodal_modulev3_arch
from .gemma3 import Gemma3
from .inputs import Gemma3MultiModalModelInputs
from .model_config import Gemma3Config
from .multimodal_config import (
    Gemma3ForConditionalGenerationConfig,
    Gemma3VisionConfig,
)

__all__ = [
    "Gemma3",
    "Gemma3Config",
    "Gemma3ForConditionalGenerationConfig",
    "Gemma3MultiModalModelInputs",
    "Gemma3VisionConfig",
    "gemma3_modulev3_arch",
    "gemma3_multimodal_modulev3_arch",
]
