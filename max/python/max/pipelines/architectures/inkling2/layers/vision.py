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
"""The SigLIP-style vision tower with axial 2-D rope."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType
from max.nn import Embedding, LayerNorm, Module, ModuleList
from max.nn._graph_ops import flash_attention_gpu as _flash_attention_gpu
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.linear import Linear
from max.tensor import Tensor

from ..model_config import Inkling2VisionConfig

flash_attention_gpu = F.functional(_flash_attention_gpu)

_FP32 = DType.float32

_INTERP_TAPS = 4
"""``mode="bilinear"`` emits a 2x2 separable stencil per patch
(``get_vision_interpolation_indices_and_weights``)."""

_ROPE_THETA_DEFAULT = 10000.0
"""The reference's rope base. Its config declares ``rope_parameters`` as
``None``, so the config library's standardization fills ``rope_theta =
10000.0`` (``configuration_inkling2.py:171``) and the tower takes it as the
default."""


def vision_rope_frequencies(head_dim: int, *, theta: float) -> np.ndarray:
    """Axial 2-D rope inverse frequencies, ``head_dim // 4`` of them."""
    spatial_dim = head_dim // 2
    exponents = np.arange(0, spatial_dim, 2, dtype=np.float64) / spatial_dim
    return (1.0 / (float(theta) ** exponents)).astype(np.float32)


def _repeat_interleave_pairs(x: Tensor) -> Tensor:
    rows = x.shape[0]
    half = x.shape[1]
    widened = F.broadcast_to(F.unsqueeze(x, -1), [rows, half, 2])
    return F.reshape(widened, [rows, half * 2])


def vision_rope_cos_sin(
    position_ids: Tensor, frequencies: np.ndarray
) -> tuple[Tensor, Tensor]:
    """Builds the per-row cos/sin tables for the axial vision rope."""
    rows = position_ids.shape[0]
    taps = len(frequencies)
    inv_freq = F.constant(frequencies, _FP32, position_ids.device)
    # [rows, 2] -> [rows, 2, taps]: row 0 is the h coordinate, row 1 the w.
    angles = F.unsqueeze(position_ids.cast(_FP32), -1) * inv_freq
    # The reference concatenates width-then-height
    # (modeling_inkling2.py:927-930), so the (h, w) columns swap before
    # the row-major reshape.
    h_angles, w_angles = F.split(angles, [1, 1], axis=1)
    angles = F.concat([w_angles, h_angles], axis=1)
    cos = F.reshape(F.cos(angles), [rows, 2 * taps])
    sin = F.reshape(F.sin(angles), [rows, 2 * taps])
    return _repeat_interleave_pairs(cos), _repeat_interleave_pairs(sin)


def _rotate_pairs(x: Tensor) -> Tensor:
    batch, seq, heads, dim = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    paired = F.reshape(x, [batch, seq, heads, dim // 2, 2])
    first = paired[..., 0:1]
    second = paired[..., 1:2]
    rotated = F.concat([-second, first], axis=-1)
    return F.reshape(rotated, [batch, seq, heads, dim])


def _apply_rotary_vision(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    # One position row per sequence row is the reference's invariant (:1456):
    # rebind the tables onto q's own row axis.
    seq = q.shape[1]
    dim = q.shape[3]
    cos = F.unsqueeze(F.rebind(cos, [seq, dim]), -2)
    sin = F.unsqueeze(F.rebind(sin, [seq, dim]), -2)
    q_fp32 = q.cast(_FP32)
    k_fp32 = k.cast(_FP32)
    q_rot = q_fp32 * cos + _rotate_pairs(q_fp32) * sin
    k_rot = k_fp32 * cos + _rotate_pairs(k_fp32) * sin
    return q_rot.cast(q.dtype), k_rot.cast(k.dtype)


class _LayerScale(Module[[Tensor], Tensor]):
    gamma: Tensor

    def __init__(self, dim: int) -> None:
        self.gamma = Tensor.ones([dim])

    def forward(self, x: Tensor) -> Tensor:
        """Scales ``x`` by the per-channel gamma."""
        return x * self.gamma.cast(x.dtype)


class Inkling2VisionAttention(Module[[Tensor, Tensor, Tensor], Tensor]):
    """Bidirectional multi-head self-attention with axial rope (``:952-1055``)."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "unsupported Inkling2 vision_config: hidden_size "
                f"{config.hidden_size} is not divisible by "
                f"num_attention_heads {config.num_attention_heads}"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = Linear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Attends every row of one image to every other row of that image."""
        rows = x.shape[0]
        shape: list[int | Dim] = [1, rows, self.num_heads, self.head_dim]
        q = F.reshape(self.q_proj(x), shape)
        k = F.reshape(self.k_proj(x), shape)
        v = F.reshape(self.v_proj(x), shape)
        q, k = _apply_rotary_vision(q, k, cos, sin)
        # NULL_MASK: no causal masking, one segment per image.
        attn = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )
        return self.out_proj(
            F.reshape(attn, [rows, self.num_heads * self.head_dim])
        )


class Inkling2VisionMLP(Module[[Tensor], Tensor]):
    """Two-layer exact-gelu MLP, ``c_fc`` then ``c_proj`` (``:1058-1070``)."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        self.c_fc = Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.c_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies ``c_proj(gelu(c_fc(x)))``."""
        return self.c_proj(F.gelu(self.c_fc(x), approximate="none"))


class Inkling2VisionEncoderLayer(Module[[Tensor, Tensor, Tensor], Tensor]):
    """One pre-LayerNorm block with its own LayerScale pair (``:1073-1100``)."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Inkling2VisionAttention(config)
        self.ls_1 = _LayerScale(config.hidden_size)
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Inkling2VisionMLP(config)
        self.ls_2 = _LayerScale(config.hidden_size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """``x + ls_1 * attn(ln_1(x))``, then ``x + ls_2 * mlp(ln_2(x))``."""
        x = x + self.ls_1(self.self_attn(self.ln_1(x), cos, sin))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Inkling2VisionEmbeddings(Module[[Tensor, Tensor, Tensor], Tensor]):
    """Patch embedding, interpolated position grid, and the class token row."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: DType) -> None:
        self.config = config
        self.patch_embedding = Linear(
            config.patch_elements, config.hidden_size, bias=False
        )
        # An nn.Embedding table on the reference side; the adapter puts the
        # checkpoint's suffixless ``position_embedding`` leaf under ``.weight``.
        self.position_embedding = Embedding(
            config.posemb_grid_size**2, dim=config.hidden_size
        )
        self.class_embedding = Tensor.zeros([config.hidden_size], dtype=dtype)
        self.cls_position_embedding = Tensor.zeros(
            [config.hidden_size], dtype=dtype
        )

    def forward(
        self,
        pixel_values: Tensor,
        interp_indices: Tensor,
        interp_weights: Tensor,
    ) -> Tensor:
        """Returns ``[1 + num_patches, hidden]``: one class token, then patches."""
        config = self.config
        merge = config.spatial_merge_size
        patch = config.patch_size
        channels = config.num_channels
        # ``pixel_values.to(patch_embedding.weight.dtype)`` (``:1450``).
        pixel_values = pixel_values.cast(self.patch_embedding.weight.dtype)
        num_tokens = pixel_values.shape[0]
        num_patches = num_tokens * (merge * merge)

        blocks = F.reshape(
            pixel_values, [num_tokens, merge, patch, merge, patch, channels]
        )
        blocks = F.permute(blocks, [0, 1, 3, 2, 4, 5])
        patch_rows = F.reshape(blocks, [num_patches, config.patch_elements])
        patch_embeddings = self.patch_embedding(patch_rows)

        # ``(table[interp_indices] * interp_weights).sum(1)``, multiplied in
        # fp32 and rounded once on the way back to the serve dtype
        # (``:1135-1136``).
        table = F.gather(self.position_embedding.weight, interp_indices, axis=0)
        # The gathered table's axis 0 IS the input's ``num_patches``; rebind both
        # operands onto it.
        patches = table.shape[0]
        interp_weights = F.rebind(interp_weights, [patches, _INTERP_TAPS])
        position = F.sum(
            table.cast(_FP32) * F.unsqueeze(interp_weights, -1), axis=1
        )
        # MAX's reduction keeps the reduced axis; the reference's ``.sum(1)``
        # drops it.
        position = F.squeeze(position, 1)
        patch_embeddings = F.rebind(
            patch_embeddings, [patches, config.hidden_size]
        )
        patch_embeddings = patch_embeddings + position.cast(
            patch_embeddings.dtype
        )

        # ``class_embedding + cls_position_embedding``, repeated to one row
        # because this call encodes one image (``:1138-1141``).
        class_token = self.class_embedding + self.cls_position_embedding
        class_row = F.reshape(class_token, [1, config.hidden_size])
        return F.concat([class_row, patch_embeddings], axis=0)


class Inkling2VisionBackbone(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
):
    """Embeddings, one LayerNorm (``ln_pre``), then the block stack."""

    def __init__(self, config: Inkling2VisionConfig, *, dtype: DType) -> None:
        self.embeddings = Inkling2VisionEmbeddings(config, dtype=dtype)
        self.ln_pre = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = ModuleList(
            [
                Inkling2VisionEncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        pixel_values: Tensor,
        interp_indices: Tensor,
        interp_weights: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Runs the blocks over one image's packed sequence, class token first."""
        x = self.ln_pre(
            self.embeddings(pixel_values, interp_indices, interp_weights)
        )
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class InklingVisionModel(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Encodes one image into one decoder-width row per image token."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: DType,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.backbone = Inkling2VisionBackbone(config, dtype=dtype)
        self.proj_in = Linear(
            config.merged_hidden_size,
            config.projection_hidden_dim,
            bias=config.projector_bias,
        )
        self.proj_out = Linear(
            config.projection_hidden_dim,
            config.text_hidden_size,
            bias=config.projector_bias,
        )
        self._rope_frequencies = vision_rope_frequencies(
            self.head_dim, theta=_ROPE_THETA_DEFAULT
        )

    def forward(
        self,
        pixel_values: Tensor,
        interp_indices: Tensor,
        interp_weights: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Encodes one image."""
        config = self.config
        merge_squared = config.spatial_merge_size**2
        num_tokens = pixel_values.shape[0]
        num_patches = num_tokens * merge_squared
        cos, sin = vision_rope_cos_sin(position_ids, self._rope_frequencies)
        hidden = self.backbone(
            pixel_values, interp_indices, interp_weights, cos, sin
        )
        # The class token row is dropped before the merger (``:1465-1466``:
        # ``hidden_states[patch_index]``, and every patch index is >= 1).
        patch_states = F.rebind(
            hidden[1:, :], [num_patches, config.hidden_size]
        )
        # One merged row per image token: merge**2 consecutive patch features
        # back to back, which is the reference's ``reshape(-1, merge**2 * C)``.
        merged = F.reshape(
            patch_states, [num_tokens, config.merged_hidden_size]
        )
        return self.proj_out(F.gelu(self.proj_in(merged), approximate="none"))

    def input_types(self, device: DeviceRef) -> tuple[TensorType, ...]:
        """Graph inputs: patch blocks, interpolation taps, and position ids."""
        config = self.config
        side = config.spatial_merge_size * config.patch_size
        return (
            TensorType(
                DType.float32,
                shape=["num_tokens", side, side, config.num_channels],
                device=device,
            ),
            TensorType(
                DType.int32,
                shape=["num_patches", _INTERP_TAPS],
                device=device,
            ),
            TensorType(
                DType.float32,
                shape=["num_patches", _INTERP_TAPS],
                device=device,
            ),
            TensorType(
                DType.int32,
                shape=["num_rows", 2],
                device=device,
            ),
        )
