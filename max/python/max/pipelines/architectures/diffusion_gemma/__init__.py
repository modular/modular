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
"""DiffusionGemma block-diffusion text-generation architecture."""

# Importing the parsers registers them under "gemma4", the names the arch
# registration gives as tool_parser and reasoning_parser.
from . import reasoning as _gemma4_reasoning
from . import tool_parser as _gemma4_tool_parser
from .arch import diffusion_gemma_arch
from .diffusion_gemma import (
    DiffusionGemmaDenoiser,
    DiffusionGemmaEncoder,
    DiffusionGemmaVision,
)
from .model_config import DiffusionGemmaForBlockDiffusionConfig

__all__ = [
    "DiffusionGemmaDenoiser",
    "DiffusionGemmaEncoder",
    "DiffusionGemmaForBlockDiffusionConfig",
    "DiffusionGemmaVision",
    "diffusion_gemma_arch",
]
