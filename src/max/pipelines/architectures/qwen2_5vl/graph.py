# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Build a Qwen2.5VL model via Graph API from Safetensors weights."""

from max.dtype import DType
from max.graph import ops
from max.graph.weights import Weights
from max.nn import Conv3D

from .nn.visual_transformer import VisionPatchEmbed


def patch_embed(
    dtype: DType,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    embed_dim: int,
    out_channels: int,
    weights: Weights,
) -> VisionPatchEmbed:
    kernel_size = (temporal_patch_size, patch_size, patch_size)
    filter_weights = ops.permute(
        weights.weight.allocate(
            dtype,
            [
                out_channels,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            ],
            None,
        ),
        [2, 3, 4, 1, 0],
    )
    proj = Conv3D(
        filter=filter_weights,
        bias=None,
        stride=kernel_size,
    )
    return VisionPatchEmbed(
        proj=proj,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
