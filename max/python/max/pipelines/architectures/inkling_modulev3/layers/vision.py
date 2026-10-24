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
"""Inkling vision tower: a hierarchical MLP ("hmlp") over image patches.

Ported from ``vllm/models/inkling/common/towers.py`` at vLLM v0.26.0
``568afb3a13806beb53bb2e6bd518269357b237c0``.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

from ...inkling.layers.vision import plan_folds
from ...inkling.model_config import InklingVisionConfig

# The vision config carries no epsilon of its own, and the reference never
# reads the text config's.
_NORM_EPS = 1e-6


def fold_timespace_to_depth(x: Tensor, t_fold: int, hw_fold: int) -> Tensor:
    """Merges a ``t_fold x hw_fold x hw_fold`` neighborhood of
    ``[patches, t, h, w, c]`` into the channels, ordered frame, row, column,
    channel."""
    if t_fold == 1 and hw_fold == 1:
        return x
    patches = x.shape[0]
    t, h, w, c = (int(dim) for dim in x.shape[1:])
    if t % t_fold or h % hw_fold or w % hw_fold:
        raise ValueError(
            f"cannot fold {(t, h, w)} by {(t_fold, hw_fold, hw_fold)}"
        )
    split = x.reshape(
        [
            patches,
            t // t_fold,
            t_fold,
            h // hw_fold,
            hw_fold,
            w // hw_fold,
            hw_fold,
            c,
        ]
    )
    return F.permute(split, [0, 1, 3, 5, 2, 4, 6, 7]).reshape(
        [
            patches,
            t // t_fold,
            h // hw_fold,
            w // hw_fold,
            t_fold * hw_fold * hw_fold * c,
        ]
    )


class InklingVisionModel(Module[[Tensor], Tensor]):
    """Encodes image patches into one decoder-width token per patch."""

    def __init__(
        self,
        config: InklingVisionConfig,
        dtype: DType,
        *,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.folds = plan_folds(config)

        linears = []
        norms = []
        in_dim = config.n_channels
        for index, (t_fold, hw_fold, out_dim) in enumerate(self.folds):
            linears.append(
                Linear(in_dim * t_fold * hw_fold * hw_fold, out_dim, bias=False)
            )
            if index < len(self.folds) - 1:
                norms.append(RMSNorm(out_dim, eps=_NORM_EPS))
            in_dim = out_dim
        self.linears = ModuleList(linears)
        self.norms = ModuleList(norms)
        self.final_norm = (
            RMSNorm(config.decoder_dmodel, eps=_NORM_EPS)
            if config.use_vision_norm
            else None
        )

    def normalize(self, pixels: Tensor) -> Tensor:
        """Raw pixels ``[patches, patch, patch, channels]`` to the tower's
        input frames ``[patches, temporal_patch_size, patch, patch, channels]``.

        Normalizes in float32 in the reference's order, then repeats the one
        image frame across the temporal axis.
        """
        f32 = DType.float32
        device = pixels.device
        x = (
            pixels.cast(f32) * F.constant(1.0 / 255.0, f32, device)
            - F.constant(self.image_mean, f32, device)
        ) / F.constant(self.image_std, f32, device)
        frames = F.unsqueeze(x.cast(self.dtype), 1)
        return F.broadcast_to(
            frames,
            [
                pixels.shape[0],
                self.config.temporal_patch_size,
                *pixels.shape[1:],
            ],
        )

    def forward(self, patches: Tensor) -> Tensor:
        x = self.normalize(patches)
        for index, (t_fold, hw_fold, _) in enumerate(self.folds):
            x = fold_timespace_to_depth(x, t_fold, hw_fold)
            x = self.linears[index](x)
            if index < len(self.norms):
                x = F.gelu(self.norms[index](x))
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x.reshape([x.shape[0], self.config.decoder_dmodel])

    def input_types(self, device: DeviceRef) -> tuple[TensorType, ...]:
        """One block of raw patches, as the image processor emits them."""
        return (
            TensorType(
                DType.int16,
                shape=[
                    "total_patches",
                    self.config.patch_size,
                    self.config.patch_size,
                    self.config.n_channels,
                ],
                device=device,
            ),
        )
