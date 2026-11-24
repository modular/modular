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
"""Inkling 2 vision tower and projector (ModuleV3 API).

A pre-LayerNorm ViT with layer scale and axial 2D RoPE. Every image token
covers a ``merge x merge`` square of native patches, whose features the
projector concatenates and maps to the text width. Attribute names equal the
checkpoint's names below ``model.visual.``, so the weight rules only rename the
prefix.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from max import functional as F
from max._artifact.vision import (
    Inkling2Blocks,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType
from max.nn import Module, as_subgraph
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.nn.linear import Linear
from max.nn.norm import LayerNorm
from max.nn.sequential import ModuleList
from max.pipelines.lib import PipelineConfig, WeightRenaming, WeightRule
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor, default_dtype
from transformers import AutoTokenizer, PretrainedConfig

from .preprocess_graph import Inkling2ImagePreprocess
from .processor import Inkling2Processor, load_processor_config

NUM_CHANNELS = 3

DEFAULT_ROPE_THETA = 10_000.0
"""The reference hardcodes 10000; the checkpoint sets no rope base."""

INKLING2_VISION_WEIGHT_RULES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^model\.visual\.", "vision."),
)
"""Puts the checkpoint's vision tensors under the ``vision.`` namespace."""


@dataclass(frozen=True, kw_only=True)
class Inkling2VisionConfig:
    """The checkpoint's ``vision_config``."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    """Side of a native patch in pixels."""
    spatial_merge_size: int
    """Native patches per image token along each axis."""
    posemb_grid_size: int
    projection_hidden_dim: int
    text_hidden_size: int
    layer_norm_eps: float
    rope_theta: float

    @classmethod
    def from_hf(
        cls, vision_config: PretrainedConfig, *, text_hidden_size: int
    ) -> Inkling2VisionConfig:
        """Builds the config from the checkpoint's ``vision_config``.

        Reads the raw ``config.json`` keys, which MAX's config shim passes.

        Args:
            vision_config: The ``vision_config`` of the checkpoint's config.
            text_hidden_size: The text model's hidden size, which the
                checkpoint's section leaves at the full model's.
        """
        hidden_size = vision_config.hidden_size
        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * vision_config.mlp_ratio),
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            patch_size=vision_config.native_patch_size,
            spatial_merge_size=vision_config.adapter_merge_size,
            posemb_grid_size=vision_config.posemb_grid_size,
            projection_hidden_dim=vision_config.projection_hidden_dim,
            text_hidden_size=text_hidden_size,
            layer_norm_eps=vision_config.layer_norm_eps,
            rope_theta=DEFAULT_ROPE_THETA,
        )


class Inkling2LayerScale(Module[[Tensor], Tensor]):
    """Scales a residual branch per channel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = Tensor.zeros([dim])

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x * gamma``."""
        return x * self.gamma


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates interleaved channel pairs of ``[S, heads, head_dim]`` in fp32."""
    heads, head_dim = int(x.shape[1]), int(x.shape[2])
    pairs = x.cast(DType.float32).reshape([-1, heads, head_dim // 2, 2])
    x1, x2 = pairs[..., 0], pairs[..., 1]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    rotated = F.stack([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    return rotated.reshape([-1, heads, head_dim]).cast(x.dtype)


class Inkling2VisionAttention(Module[..., Tensor]):
    """Bidirectional attention within each image's ragged sequence."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.out_proj = Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Returns the rotary attention of ``x`` within each image."""
        shape = (-1, self.num_heads, self.head_dim)
        q = _apply_rope(self.q_proj(x).reshape(shape), rot_cos, rot_sin)
        k = _apply_rope(self.k_proj(x).reshape(shape), rot_cos, rot_sin)
        out = flash_attention_ragged_gpu(
            q,
            k,
            self.v_proj(x).reshape(shape),
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.head_dim**-0.5,
        )
        return self.out_proj(out.reshape((-1, self.num_heads * self.head_dim)))


class Inkling2VisionMLP(Module[[Tensor], Tensor]):
    """The tower's feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = Linear(hidden_size, intermediate_size)
        self.c_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the feed-forward of ``x``."""
        return self.c_proj(F.gelu(self.c_fc(x)))


class Inkling2VisionLayer(Module[..., Tensor]):
    """A pre-LayerNorm block whose branches are layer-scaled."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.layer_norm_eps
        self.ln_1 = LayerNorm(hidden, eps=eps, keep_dtype=False)
        self.self_attn = Inkling2VisionAttention(
            hidden, config.num_attention_heads
        )
        self.ls_1 = Inkling2LayerScale(hidden)
        self.ln_2 = LayerNorm(hidden, eps=eps, keep_dtype=False)
        self.mlp = Inkling2VisionMLP(hidden, config.intermediate_size)
        self.ls_2 = Inkling2LayerScale(hidden)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Returns ``x`` plus its attention and feed-forward branches."""
        x = x + self.ls_1(
            self.self_attn(
                self.ln_1(x), rot_cos, rot_sin, cu_seqlens, max_seqlen
            )
        )
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Inkling2VisionEmbeddings(Module[..., Tensor]):
    """Patch projection, interpolated position table and class row."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_embedding = Linear(
            NUM_CHANNELS * config.patch_size**2, config.hidden_size, bias=False
        )
        self.position_embedding = Tensor.zeros(
            [config.posemb_grid_size**2, config.hidden_size]
        )
        self.class_embedding = Tensor.zeros([config.hidden_size])
        self.cls_position_embedding = Tensor.zeros([config.hidden_size])

    def forward(
        self,
        patches: Tensor,
        interp_idx: Tensor,
        interp_w: Tensor,
        seq_index: Tensor,
    ) -> Tensor:
        """Returns each image's class row followed by its patch rows.

        ``patches`` is ``[N, patch * patch * channels]``.
        """
        # The tables count the patches on their own symbol.
        patches = F.rebind(patches, [interp_idx.shape[0], patches.shape[1]])
        h = self.patch_embedding(patches)
        taps = F.gather(self.position_embedding, interp_idx, axis=0)
        pos = (taps.cast(DType.float32) * interp_w.unsqueeze(-1)).sum(axis=1)
        h = h + pos.squeeze(1).cast(h.dtype)
        cls = (self.class_embedding + self.cls_position_embedding).reshape(
            [1, self.hidden_size]
        )
        return F.gather(F.concat([h, cls], axis=0), seq_index, axis=0)


class Inkling2VisionBackbone(Module[..., Tensor]):
    """Embeddings, ``ln_pre`` and the layer stack."""

    def __init__(self, config: Inkling2VisionConfig) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, keep_dtype=False
        )
        self.layers = ModuleList(
            [
                Inkling2VisionLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        patches: Tensor,
        interp_idx: Tensor,
        interp_w: Tensor,
        seq_index: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Encodes the patches into one sequence per image."""
        h = self.ln_pre(
            self.embeddings(patches, interp_idx, interp_w, seq_index)
        )
        for layer in self.layers:
            h = as_subgraph(layer, name="inkling2_vision_layer")(
                h, rot_cos, rot_sin, cu_seqlens, max_seqlen
            )
        return h


class VisionHostConfig(Protocol):
    """What the registered vision role reads of the architecture config."""

    dtype: DType
    devices: Sequence[DeviceRef]
    vision_config: Inkling2VisionConfig


class Inkling2Vision(Module[..., Tensor]):
    """The vision tower and projector, registered for the ``VISION`` role.

    Maps image pixel blocks to embeddings as wide as the text tower's hidden
    state, on the config's first device.
    """

    graph_name = "inkling2_vision"

    def __init__(self, config: VisionHostConfig) -> None:
        super().__init__()
        vision = config.vision_config
        self.vision_config = vision
        self._device_ref = config.devices[0]
        self.embedding_dtype = config.dtype
        self.embedding_dim = vision.text_hidden_size
        with default_dtype(config.dtype):
            self.backbone = Inkling2VisionBackbone(vision)
            merged = vision.spatial_merge_size**2 * vision.hidden_size
            self.proj_in = Linear(
                merged, vision.projection_hidden_dim, bias=False
            )
            self.proj_out = Linear(
                vision.projection_hidden_dim,
                vision.text_hidden_size,
                bias=False,
            )

    def normalize(self, pixels: Tensor) -> Tensor:
        """Rescales and normalizes raw 0 to 255 pixels, in float32.

        The constants are those of ``processor_config.json``: rescale 1/255,
        mean 0.5 and std 0.5. Padding holds -1 and normalizes like any other
        value, as in the reference.
        """
        x = pixels.cast(DType.float32) * (1.0 / 255.0)
        x = (x - 0.5) / 0.5
        return x.cast(self.backbone.embeddings.patch_embedding.weight.dtype)

    def patchify(self, pixels: Tensor) -> Tensor:
        """Splits ``[tokens, side, side, channels]`` into native patches.

        Returns ``[tokens * merge**2, patch * patch * channels]``, the patches
        of one token consecutive in row-major order, each flattened row,
        column, channel.
        """
        merge = self.vision_config.spatial_merge_size
        patch = self.vision_config.patch_size
        x = self.normalize(pixels).reshape(
            [-1, merge, patch, merge, patch, NUM_CHANNELS]
        )
        x = F.permute(x, [0, 1, 3, 2, 4, 5])
        return x.reshape([-1, patch * patch * NUM_CHANNELS])

    def forward(
        self,
        interp_idx: Tensor,
        interp_w: Tensor,
        positions: Tensor,
        pixel_values: Tensor,
        cu_seqlens: Tensor,
        seq_index: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Returns one text-width embedding per image token.

        The engine's packer offsets the bounds of an image by the patch rows
        before it, so ``cu_seqlens`` counts patches and no input names the
        sequence row of each patch. This adds the class row of each image to
        the bounds and finds the patch rows from ``seq_index``.
        """
        bounds = cu_seqlens.shape[0]
        cu_seqlens = cu_seqlens + F.arange(
            0,
            bounds,
            out_dim=bounds,
            dtype=DType.uint32,
            device=cu_seqlens.device,
        )
        # `seq_index` holds each patch's index at its sequence row and the
        # patch count at every class row, so the patch rows come first in the
        # sorted order and their keys are the patch indices.
        patch_index = F.argsort(seq_index)[: interp_idx.shape[0]]
        rot_cos, rot_sin = (
            table.to(cu_seqlens.device)
            for table in self._rotary_angles(positions)
        )
        h = self.backbone(
            self.patchify(pixel_values),
            interp_idx,
            interp_w,
            seq_index,
            rot_cos,
            rot_sin,
            cu_seqlens,
            max_seqlen,
        )
        h = F.gather(h, patch_index, axis=0)
        blocks = self.vision_config.spatial_merge_size**2
        n, hidden = h.shape
        # Every image's patch count is a multiple of the merge area.
        h = F.rebind(h, [(n // blocks) * blocks, hidden])
        h = h.reshape([-1, blocks * int(hidden)])
        return self.proj_out(F.gelu(self.proj_in(h)))

    def _rotary_angles(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the axial 2D RoPE cos and sin of host ``[S, 2]`` positions.

        Each axis rotates half the channels in interleaved pairs; the first
        half of the pairs follows the column, the second half the row. The
        host computes them, since the GPU's float32 cos and sin err by 1e-4
        at an angle of 1000.
        """
        config = self.vision_config
        spatial = config.hidden_size // config.num_attention_heads // 2
        # Rounded from float64, as the reference constants.
        iota = F.arange(0, spatial, step=2, dtype=DType.float64, device=CPU())
        inv_freq = (1.0 / (config.rope_theta ** (iota / spatial))).cast(
            DType.float32
        )
        rows = positions[:, 0:1].cast(DType.float32)
        cols = positions[:, 1:2].cast(DType.float32)
        angles = F.concat([cols * inv_freq, rows * inv_freq], axis=-1)
        return F.cos(angles), F.sin(angles)

    def input_types(self) -> OrderedDict[str, object]:
        """Groups the tower's inputs as a serving engine packs them.

        The first field counts patches, which the other fields' bounds and
        indices are counted against. The layer subgraph takes the sequence and
        image counts as plain symbols, since a subgraph signature can't hold
        an expression of them.
        """
        patches, sequence = Dim("total_patches"), Dim("sequence_length")
        config = self.vision_config
        side = config.patch_size * config.spatial_merge_size
        device = self._device_ref
        return OrderedDict(
            patches=VisionRows(
                interp_idx=TensorType(DType.int32, [patches, 4], device=device),
                interp_w=TensorType(DType.float32, [patches, 4], device=device),
                positions=TensorType(
                    DType.int32, [sequence, 2], device=DeviceRef.CPU()
                ),
                pixel_values=TensorType(
                    DType.int16,
                    [Dim("total_image_tokens"), side, side, NUM_CHANNELS],
                    device=device,
                ),
            ),
            segments=VisionSegments(
                cu_seqlens=TensorType(
                    DType.uint32, [Dim("num_images_plus_1")], device=device
                )
            ),
            sequence=VisionRowIndex(
                seq_index=TensorType(DType.int64, [sequence], device=device)
            ),
            bound=VisionHostMax(
                max_seqlen=TensorType(DType.uint32, [], device=DeviceRef.CPU())
            ),
        )

    @classmethod
    def vision_metadata(
        cls, pipeline_config: PipelineConfig, config: VisionHostConfig
    ) -> VisionMetadata:
        """Returns the media layout the checkpoint is served with.

        An image is padded to whole blocks, each one image token, so the
        engine sizes it with :class:`~max._artifact.vision.Inkling2Blocks`.
        """
        vision = config.vision_config
        model = pipeline_config.model
        processor = Inkling2Processor(
            AutoTokenizer.from_pretrained(
                model.model_path,
                revision=model.huggingface_model_revision,
                trust_remote_code=model.trust_remote_code,
            ),
            load_processor_config(
                model.model_path, model.huggingface_model_revision
            ),
        )
        return VisionMetadata.from_layouts(
            image=ModalityLayout(
                patch_size=vision.patch_size,
                merge_size=vision.spatial_merge_size,
                strategy=Inkling2Blocks(),
                span=Span(processor.image_token_id),
            ),
            video=None,
            max_image_bytes=0,
            max_video_bytes=0,
            prompt=PromptText(),
        )

    @classmethod
    def vision_preprocess_spec(
        cls, pipeline_config: PipelineConfig, config: VisionHostConfig
    ) -> GraphSpec:
        """Returns the graph that preprocesses one image for the tower."""
        del pipeline_config
        vision = config.vision_config
        return GraphSpec(
            Inkling2ImagePreprocess(
                patch_size=vision.patch_size,
                merge_size=vision.spatial_merge_size,
                posemb_side=vision.posemb_grid_size,
            ),
            "inkling2_vision_preprocess",
            image_preprocess_input_types(),
            weights={},
        )
