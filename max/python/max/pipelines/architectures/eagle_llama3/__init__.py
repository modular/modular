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
"""EAGLE speculative decoding draft model for Llama 3."""

from .arch import eagle3_llama_arch, eagle_llama_arch
from .eagle_llama3 import EagleLlama3Draft
from .model_config import EagleLlama3Config

__all__ = [
    "EagleLlama3Config",
    "EagleLlama3Draft",
    "eagle3_llama_arch",
    "eagle_llama_arch",
]
