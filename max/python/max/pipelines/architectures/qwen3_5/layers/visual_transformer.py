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
"""Vision transformer for Qwen3.5.

The vision architecture is identical to Qwen3VL-MoE; re-export everything
from that package so we have a single implementation.
"""

from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    BilinearInterpolationPositionEmbedding,
    VisionBlock,
    VisionMLP,
    VisionPatchEmbed,
    VisionPatchMerger,
    VisionRotaryEmbedding,
    VisionTransformer,
)

__all__ = [
    "BilinearInterpolationPositionEmbedding",
    "VisionBlock",
    "VisionMLP",
    "VisionPatchEmbed",
    "VisionPatchMerger",
    "VisionRotaryEmbedding",
    "VisionTransformer",
]
