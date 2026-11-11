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
"""GLM-5.3-Flash vision tower on ModuleV3.

Structurally GLM-4.6V's tower with two modules removed and three changed, which
is exactly how the reference builds it: ``Glm5NextVisionModel`` derives from
``GlmOcrVisionModel``, which derives from ``Glm4vVisionModel`` and whose
``__init__`` is little more than ``del self.embeddings`` and
``del self.post_conv_layernorm``.

Removed, because the checkpoint has no weights for them: the bicubically
interpolated absolute position embedding, and the post-convolution norm.
Changed: attention gains q/k norms and biases, and both SwiGLUs are clamped.

The tower runs on the first device. Tensor parallelism is not supported: the
vision encoder cache copies the rows to every device, and at 0.56B parameters
one replica is cheap.
"""

from __future__ import annotations

from typing import ClassVar

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import LayerNorm, Linear, Module, ModuleList, RMSNorm, as_subgraph
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor, default_dtype

from ..model_config import Glm5NextConfig
from .attention import Glm5NextVisionAttention
from .config import Glm5NextVisionConfig

__all__ = [
    "Glm5NextVisionBlock",
    "Glm5NextVisionEncoder",
    "Glm5NextVisionSpatialMerge",
    "Glm5NextVisionTransformer",
]


def clamped_swiglu(gate: Tensor, up: Tensor, limit: float) -> Tensor:
    """Returns ``silu(min(gate, limit)) * clamp(up, -limit, limit)``.

    The clamps come before the activation; clamping its output instead agrees
    below the limit and diverges above it, which only a real checkpoint
    reaches.
    """
    return F.silu(gate.clip(max=limit)) * up.clip(min=-limit, max=limit)


def patch_dim(config: Glm5NextVisionConfig) -> int:
    """Returns the width of one flattened pixel patch."""
    return (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )


class Glm5NextVisionMLP(Module[[Tensor], Tensor]):
    """Clamped SwiGLU MLP of every block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        has_bias: bool,
        swiglu_limit: float,
    ) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.gate_proj = Linear(hidden_dim, intermediate_dim, bias=has_bias)
        self.up_proj = Linear(hidden_dim, intermediate_dim, bias=has_bias)
        self.down_proj = Linear(intermediate_dim, hidden_dim, bias=has_bias)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the clamped SwiGLU over ``x``."""
        return self.down_proj(
            clamped_swiglu(
                self.gate_proj(x), self.up_proj(x), self.swiglu_limit
            )
        )


class _PatchProjection(Module[[Tensor], Tensor]):
    """The checkpoint's ``Conv3d`` patch projection, run as a matmul.

    Its kernel equals its stride and covers a whole patch, so it is a linear
    map of the already-flattened patch rows. The weight keeps the
    checkpoint's 5-D shape and is flattened here; its memory order is
    ``(channel, temporal, height, width)``, which is the order the image
    processor emits, so no permutation is involved.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.weight = Tensor.zeros(
            [
                config.hidden_size,
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ]
        )
        self.bias = Tensor.zeros([config.hidden_size])

    def forward(self, x: Tensor) -> Tensor:
        """Projects ``(patches, patch_dim)`` rows to the tower's width."""
        flat = self.weight.reshape([self.weight.shape[0], -1])
        return x.cast(flat.dtype) @ flat.T + self.bias


class Glm5NextVisionPatchEmbed(Module[[Tensor], Tensor]):
    """Embeds flattened pixel patches."""

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.proj = _PatchProjection(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Projects ``(patches, patch_dim)`` pixels to the tower's width."""
        return self.proj(pixel_values)


class Glm5NextVisionSpatialMerge(Module[[Tensor], Tensor]):
    """The tower's 2x2 strided-conv spatial merge, run as a matmul.

    The convolution sees exactly one ``merge x merge`` window per output token,
    so it is a matmul over the four merged patches -- but only under one
    flattening, and the wrong one is a silent accuracy loss rather than a shape
    error. The weight rules therefore leave ``downsample.weight`` in its
    checkpoint form, ``(out, channel, kh, kw)``, and this module owns the
    flattening on both sides:

    * the reference reshapes the token sequence to
      ``(-1, merge, merge, hidden)`` before permuting into NCHW, so within a
      merge block the four patch rows are ordered ``(h_in_block, w_in_block)``
      and a flat row of four concatenated rows is ``(h, w, channel)``;
    * the weight is therefore permuted to ``(out, kh, kw, channel)`` to match,
      which is a constant fold rather than runtime data movement.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        merge = config.spatial_merge_size
        self.out_dim = config.out_hidden_size
        self.weight = Tensor.zeros(
            [config.out_hidden_size, config.hidden_size, merge, merge]
        )
        self.bias = Tensor.zeros([config.out_hidden_size])

    def forward(self, x: Tensor) -> Tensor:
        """Merges ``(num_tokens, merge * merge * hidden)`` rows.

        Args:
            x: Merge windows already flattened as ``(h, w, channel)``.

        Returns:
            ``(num_tokens, out_hidden_size)``.
        """
        weight = self.weight.permute([0, 2, 3, 1]).reshape([self.out_dim, -1])
        return x @ weight.transpose(0, 1) + self.bias


class Glm5NextVisionRotaryEmbedding(Module[[Tensor], tuple[Tensor, Tensor]]):
    """2D vision RoPE tables, built from per-patch ``(h, w)`` grid coordinates.

    There is no temporal axis: the two halves of the ``head_dim``-wide table
    carry the height and width angles respectively.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = head_dim // 2
        self.theta = theta

    def forward(self, rot_pos_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Builds ``(cos, sin)`` of shape ``(seq_len, head_dim)``.

        Args:
            rot_pos_ids: ``(seq_len, 2)`` integer ``(h, w)`` grid coordinates.
        """
        # float64 keeps the exponent exact before the pow, matching the
        # reference's float32 `inv_freq`.
        iota = F.range(
            0,
            self.dim,
            2,
            out_dim=self.dim // 2,
            dtype=DType.float64,
            device=rot_pos_ids.device,
        )
        inv_freq = (1.0 / (self.theta ** (iota / self.dim))).cast(DType.float32)
        positions = rot_pos_ids.cast(DType.float32)
        # (seq_len, 2, 1) * (dim // 2,) flattened to (seq_len, dim), i.e. the
        # height angles followed by the width angles.
        freqs = positions.unsqueeze(-1) * inv_freq
        freqs = freqs.reshape([freqs.shape[0], self.dim])
        emb = F.concat((freqs, freqs), axis=-1)
        return F.cos(emb), F.sin(emb)


class Glm5NextVisionBlock(Module[..., tuple[Tensor, Tensor]]):
    """Pre-norm block: RMSNorm, full attention, RMSNorm, clamped SwiGLU MLP.

    The block's own ``norm1`` is applied by whatever precedes it, and the block
    ends by applying the norm that reads its output: the next block's
    ``norm1``, or the tower's ``post_layernorm``. That keeps each residual add
    in the same graph as its norm when the blocks are subgraphs, so the
    compiler still fuses the pair and the output stays bit-identical to the
    inlined tower; split across the boundary, they round differently.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            rms_norm_eps=config.rms_norm_eps,
            has_bias=config.attention_bias,
        )
        self.mlp = Glm5NextVisionMLP(
            hidden_dim=config.hidden_size,
            # The blocks' width, not the merger's. This is the field whose
            # meaning changed between GLM-4.6V and GLM-5.3-Flash.
            intermediate_dim=config.intermediate_size,
            # The reference passes `bias=config.attention_bias` here, so one
            # config field gates the biases on attention and on the block MLP.
            has_bias=config.attention_bias,
            swiglu_limit=config.swiglu_limit,
        )

    def forward(
        self,
        x: Tensor,
        normed_x: Tensor,
        next_norm_weight: Tensor,
        cos: Tensor,
        sin: Tensor,
        input_row_offsets: Tensor,
        max_seqlen: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns the block's output, and that output normalized.

        Args:
            x: The residual stream entering the block.
            normed_x: ``x`` through this block's ``norm1``.
            next_norm_weight: Weight of the norm that reads the output.
            cos: Rotary cosine table, ``(seq_len, head_dim)``.
            sin: Rotary sine table, ``(seq_len, head_dim)``.
            input_row_offsets: Per-frame attention segment offsets.
            max_seqlen: Longest frame length, as a ``(1,)`` uint32 tensor.
        """
        x = x + self.attn(normed_x, cos, sin, input_row_offsets, max_seqlen)
        x = x + self.mlp(self.norm2(x))
        return x, rms_norm(x, next_norm_weight, self.norm1.eps)


class Glm5NextVisionPatchMerger(Module[[Tensor], Tensor]):
    """Projects merged patches into the language model's embedding space."""

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        dim = config.out_hidden_size
        self.proj = Linear(dim, dim, bias=False)
        # The tower's only LayerNorm; every other norm here is an RMSNorm, and
        # this one carries a bias and normalizes in float32.
        self.post_projection_norm = LayerNorm(dim, eps=1e-5, keep_dtype=False)
        self.swiglu_limit = config.swiglu_limit
        # GLM stores these as siblings of `merger.proj`, with no `mlp.` level,
        # so they are declared here rather than in a nested MLP module.
        width = config.projection_intermediate_size
        self.gate_proj = Linear(dim, width, bias=False)
        self.up_proj = Linear(dim, width, bias=False)
        self.down_proj = Linear(width, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Projects ``(num_tokens, out_hidden_size)`` merged rows."""
        x = self.proj(x)
        # Exact GELU, not the tanh approximation.
        x = F.gelu(self.post_projection_norm(x))
        return self.down_proj(
            clamped_swiglu(
                self.gate_proj(x), self.up_proj(x), self.swiglu_limit
            )
        )


class Glm5NextVisionTransformer(Module[..., Tensor]):
    """The GLM-5.3-Flash vision tower.

    Attention is segmented per frame by ``input_row_offsets``, exactly as the
    reference builds ``cu_seqlens``. A clip is never merged into a single
    attention span, which is what keeps per-frame vision-cache granularity
    bit-equivalent to encoding a whole clip at once.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.vision_config = config
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config.head_dim)
        self.blocks = ModuleList(
            [Glm5NextVisionBlock(config) for _ in range(config.depth)]
        )
        self.post_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.downsample = Glm5NextVisionSpatialMerge(config)
        self.merger = Glm5NextVisionPatchMerger(config)

    def forward(
        self,
        pixel_values: Tensor,
        rot_pos_ids: Tensor,
        input_row_offsets: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Encodes patches into language-model embeddings.

        Args:
            pixel_values: ``(total_patches, patch_dim)`` flattened patches, in
                the image processor's ``(channel, temporal, height, width)``
                order.
            rot_pos_ids: ``(total_patches, 2)`` ``(h, w)`` grid coordinates.
            input_row_offsets: ``(num_frames + 1,)`` uint32 prefix sum of
                per-frame patch counts -- one attention segment per frame.
            max_seqlen: Longest frame length, as a ``(1,)`` uint32 tensor.

        Returns:
            ``(total_patches // spatial_merge_unit, out_hidden_size)``
            embeddings.
        """
        h = self.patch_embed(pixel_values)
        cos, sin = self.rotary_pos_emb(rot_pos_ids)

        blocks = list(self.blocks)
        norms = [block.norm1 for block in blocks] + [self.post_layernorm]
        normed = norms[0](h)
        # Every block shares one body; each call binds its own weights. Block
        # i applies norms[i + 1], passed in so the call binds that norm's
        # weight rather than its own block's.
        for block, next_norm in zip(blocks, norms[1:], strict=True):
            h, normed = as_subgraph(block, name="glm5_next_vision_block")(
                h,
                normed,
                next_norm.weight,
                cos,
                sin,
                input_row_offsets,
                max_seqlen,
            )
        # The last block's normalized output is `post_layernorm`'s.
        h = normed

        merge = self.vision_config.spatial_merge_size
        merge_unit = merge * merge
        seq_len, hidden = h.shape
        # The patch count is a multiple of the merge unit by construction, but
        # a dynamic dimension does not carry that fact into the reshape.
        h = F.rebind(h, [(seq_len // merge_unit) * merge_unit, hidden])
        h = self.downsample(h.reshape([-1, merge_unit * hidden]))
        return self.merger(h)


class Glm5NextVisionEncoder(Glm5NextVisionTransformer):
    """The vision tower, registered for the ``VISION`` role.

    It runs on the config's first device and yields one row per merged
    vision token, as wide as the decoder's hidden state.
    """

    graph_name: ClassVar[str] = "glm5_next_vision_graph"

    @classmethod
    def is_enabled(cls, config: Glm5NextConfig) -> bool:
        """Builds the tower only for a checkpoint that ships one."""
        return config.vision_config is not None

    def __init__(self, config: Glm5NextConfig) -> None:
        vision_config = config.vision_config
        assert vision_config is not None
        with default_dtype(vision_config.dtype):
            super().__init__(vision_config)
        self.device_ref = config.devices[0]
        self.embedding_dtype = vision_config.dtype
        self.embedding_dim = config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the vision graph's inputs, in :meth:`forward`'s order."""
        device = self.device_ref
        return (
            TensorType(
                DType.float32,
                ["vision_patches", patch_dim(self.vision_config)],
                device,
            ),
            TensorType(DType.int32, ["vision_patches", 2], device),
            TensorType(DType.uint32, ["vision_segments"], device),
            TensorType(DType.uint32, [1], DeviceRef.CPU()),
        )
