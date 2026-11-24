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
"""Configuration of the GLM-5.3-Flash vision tower."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from transformers import AutoConfig
from typing_extensions import Self

__all__ = ["Glm5NextVisionConfig"]


@dataclass(kw_only=True)
class Glm5NextVisionConfig:
    """Vision tower configuration for GLM-5.3-Flash."""

    depth: int = 24
    """Number of transformer blocks."""

    hidden_size: int = 1024
    """Block width."""

    num_heads: int = 16
    """Attention heads per block; ``head_dim`` is ``hidden_size // num_heads``."""

    intermediate_size: int = 4096
    """Width of each **block's** MLP. Not the merger's."""

    projection_intermediate_size: int = 10240
    """Width of the **patch merger's** clamped SwiGLU."""

    out_hidden_size: int = 4096
    """Tower output width. Equal to the decoder's ``hidden_size``, so the
    merger feeds the decoder directly with no bridge projector."""

    patch_size: int = 14
    """Spatial patch edge."""

    temporal_patch_size: int = 2
    """Temporal patch depth of the 3D patch embedding."""

    spatial_merge_size: int = 2
    """Edge of the strided-conv spatial merge."""

    in_channels: int = 3
    image_size: int = 448
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    attention_bias: bool = True
    """Every projection in the tower carries a bias, unlike MAX's GLM-4.6V
    tower, which sets ``has_bias=False`` on both ``qkv`` and ``proj``."""

    swiglu_limit: float = 10.0
    """Clamp applied to the block MLP and merger gate/up projections."""

    dtype: DType = DType.bfloat16
    """The vision tower is BF16 throughout; it carries no FP8 weight."""

    @property
    def head_dim(self) -> int:
        """The width of one vision attention head."""
        return self.hidden_size // self.num_heads

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig, *, dtype: DType
    ) -> Self:
        """Reads the tower's fields off the checkpoint's ``vision_config``."""
        return cls(
            depth=hf_vision_config.depth,
            hidden_size=hf_vision_config.hidden_size,
            num_heads=hf_vision_config.num_heads,
            intermediate_size=hf_vision_config.intermediate_size,
            projection_intermediate_size=(
                hf_vision_config.projection_intermediate_size
            ),
            out_hidden_size=hf_vision_config.out_hidden_size,
            patch_size=hf_vision_config.patch_size,
            temporal_patch_size=hf_vision_config.temporal_patch_size,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            in_channels=getattr(hf_vision_config, "in_channels", 3),
            image_size=getattr(hf_vision_config, "image_size", 448),
            rms_norm_eps=getattr(hf_vision_config, "rms_norm_eps", 1e-5),
            hidden_act=getattr(hf_vision_config, "hidden_act", "silu"),
            attention_bias=getattr(hf_vision_config, "attention_bias", True),
            swiglu_limit=getattr(hf_vision_config, "swiglu_limit", 10.0),
            dtype=dtype,
        )
