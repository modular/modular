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

"""DeepSeek-V4.1 ViT, aligner and image span assembly (ModuleV3 API)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged_gpu,
)
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm.rms_norm import rms_norm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.nn.attention.mask_config import MHAMaskVariant


@dataclass(frozen=True)
class DeepseekV41VisionConfig:
    """The vision tower and aligner hyperparameters."""

    n_layers: int
    dim: int
    n_heads: int
    inter_dim: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    text_dim: int
    norm_eps: float = 1e-6

    @classmethod
    def from_hf(
        cls, vision_config: Mapping[str, object], text_dim: int
    ) -> DeepseekV41VisionConfig:
        """Builds the config from the checkpoint's ``vision_config``."""

        def get(key: str) -> int:
            value = vision_config[key]
            assert isinstance(value, int)
            return value

        return cls(
            n_layers=get("num_hidden_layers"),
            dim=get("hidden_size"),
            n_heads=get("num_attention_heads"),
            inter_dim=get("intermediate_size"),
            patch_size=get("patch_size"),
            rope_theta=float(get("rope_theta")),
            downsample_ratio=get("downsample_ratio"),
            text_dim=text_dim,
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def rope_dim(self) -> int:
        """Rotary angles per patch: one half by row, one half by column."""
        return self.head_dim // 2

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2


class VisionRMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm with a float32 weight, applied in float32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = Tensor.ones([dim], dtype=DType.float32)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x.cast(DType.float32), self.weight, self.eps).cast(
            x.dtype
        )


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates ``[P, heads, head_dim]`` by ``[P, head_dim // 2]`` tables."""
    x1, x2 = x.cast(DType.float32).split(int(x.shape[-1]) // 2, axis=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return F.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).cast(
        x.dtype
    )


class VisionAttention(Module[..., Tensor]):
    """Bidirectional attention within each image of a ragged batch."""

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.wqkv = Linear(config.dim, 3 * config.dim)
        self.wo = Linear(config.dim, config.dim)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        shape = (-1, self.n_heads, self.head_dim)
        q, k, v = (t.reshape(shape) for t in F.chunk(self.wqkv(x), 3, axis=-1))
        out = flash_attention_ragged_gpu(
            _apply_rotary(q, rot_cos, rot_sin),
            _apply_rotary(k, rot_cos, rot_sin),
            v,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.head_dim**-0.5,
        )
        return self.wo(out.reshape((-1, self.n_heads * self.head_dim)))


class VisionMLP(Module[[Tensor], Tensor]):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.w1 = Linear(config.dim, 2 * config.inter_dim, bias=False)
        self.w2 = Linear(config.inter_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = F.chunk(self.w1(x), 2, axis=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(Module[..., Tensor]):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.norm1 = VisionRMSNorm(config.dim, config.norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = VisionRMSNorm(config.dim, config.norm_eps)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        x = x + self.attn(
            self.norm1(x), rot_cos, rot_sin, cu_seqlens, max_seqlen
        )
        return x + self.mlp(self.norm2(x))


class PatchEmbed(Module[[Tensor], Tensor]):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.proj = Linear(config.patch_dim, config.dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DeepseekV41ViT(Module[..., Tensor]):
    """Patch embedding, the blocks and the final norm."""

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.blocks = ModuleList(
            [VisionBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = VisionRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        patches: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, rot_cos, rot_sin, cu_seqlens, max_seqlen)
        return self.norm(x)


class DeepseekV41Aligner(Module[..., Tensor]):
    """Merges each ``r x r`` patch block into one text-width row."""

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.blocks = config.downsample_ratio**2
        self.w1 = Linear(config.dim * self.blocks, config.text_dim)
        self.w2 = Linear(config.text_dim, config.text_dim)

    def forward(self, x: Tensor, aligner_index: Tensor) -> Tensor:
        # The appended zero row is what the grid padding gathers.
        x = F.gather(F.pad(x, [0, 1, 0, 0]), aligner_index, axis=0)
        n, dim = x.shape
        x = F.rebind(x, [(n // self.blocks) * self.blocks, dim])
        # Channel-major features within a block, as F.unfold orders them.
        x = (
            x.reshape((n // self.blocks, self.blocks, dim))
            .permute([0, 2, 1])
            .reshape((-1, self.blocks * dim))
        )
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV41VisionModel(Module[..., Tensor]):
    """Maps packed image patches to the embeddings of their image spans.

    Each span is ``[IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h
    + [IMAGE_END]``; the IMAGE slots take the aligner rows in reading order
    and the delimiters take the learned vectors. Attribute names equal the
    checkpoint's tensor names.
    """

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.vision = DeepseekV41ViT(config)
        self.aligner = DeepseekV41Aligner(config)
        self.image_start = Tensor.zeros([config.text_dim])
        self.image_newline = Tensor.zeros([config.text_dim])
        self.image_end = Tensor.zeros([config.text_dim])

    def forward(
        self,
        patches: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        aligner_index: Tensor,
        span_index: Tensor,
    ) -> Tensor:
        h = self.vision(patches, rot_cos, rot_sin, cu_seqlens, max_seqlen)
        rows = self.aligner(h, aligner_index)
        delimiters = F.stack(
            [self.image_start, self.image_newline, self.image_end]
        )
        return F.gather(F.concat([rows, delimiters]), span_index, axis=0)


def vision_input_types(
    config: DeepseekV41VisionConfig, device: DeviceRef, dtype: DType
) -> tuple[TensorType, ...]:
    """Returns the compile inputs of :class:`DeepseekV41VisionModel`."""
    p = "total_patches"
    return (
        TensorType(dtype, [p, config.patch_dim], device=device),
        *(
            TensorType(DType.float32, [p, config.rope_dim], device=device)
            for _ in range(2)
        ),
        TensorType(DType.uint32, ["num_images_plus_1"], device=device),
        TensorType(DType.uint32, [], device=DeviceRef.CPU()),
        TensorType(DType.int64, ["aligner_patches"], device=device),
        TensorType(DType.int64, ["span_tokens"], device=device),
    )
