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
"""Idefics3 vision-language architecture for multimodal text generation."""

from .arch import idefics3_arch
from .model import Idefics3Inputs, Idefics3Model
from .model_config import Idefics3Config, Idefics3VisionConfig

__all__ = [
    "Idefics3Config",
    "Idefics3Inputs",
    "Idefics3Model",
    "Idefics3VisionConfig",
    "idefics3_arch",
]
