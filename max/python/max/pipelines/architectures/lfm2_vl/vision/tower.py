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
"""The SigLIP2 NaFlex tower for LFM2-VL, plus the projector, one graph.

HF reference: modeling_siglip2.py::Siglip2VisionModel.forward —

    h = embeddings(pixel_values, spatial_shapes)
        # patch_embedding Linear(768 -> 1152) over the processor's
        # per-patch rows, then a position table ADDED
    h = encoder(h, bidirectional_mask)      # 27 pre-LN layers
    h = post_layernorm(h)                   # vision_use_head=false: no head

and modeling_lfm2_vl.py::Lfm2VlModel.get_image_features, which unpads
each tile row, refolds its tower output 2x2 (see vision/projector.py)
and projects it to the text width.

The module names are the reference's (`embeddings`, `encoder.layers.N`,
`post_layernorm`, `multi_modal_projector`), so the vision adapter needs
no renames at all — every rule the weight mapping rests on was checked
against the shipped 707-tensor safetensors header.

Several values that HF derives inside the forward are graph INPUTS here,
because the host already knows them when it builds the request (all
built in numpy by the batch processor):

- `pos_idx` / `pos_w` — the bilinear placement of each patch: which
  four rows of the learned 256-entry position table (a 16x16 grid, a
  checkpoint WEIGHT) the patch reads and with what weights. The batch
  processor computes torch-bilinear coordinates per grid; the graph
  gathers with them. Plain bilinear, no antialias: antialias only
  engages when a per-axis target is SMALLER than 16 patches, and the
  batch processor refuses those grids rather than place patches against
  the wrong embeddings — TODO(lfm2vl): replicate torch's antialiased
  downscale and lift the refusal.
- `tile_offsets` — the `cu_seqlens` that make each tile its own
  attention row, [num_tiles + 1] uint32;
- `max_tile_len` — the largest tile's patch count, a CPU uint32 the
  cache-less ragged kernel requires;
- `block_gather` — the projector's 2x2 fold indices,
  [num_image_tokens, 4] uint32.

`num_patches` (rows) and tokens are symbolic: one compiled tower serves
any tile set, because nothing here reshapes into a static 2-D grid (the
fold is a gather, not a reshape).
"""

from __future__ import annotations

from collections import OrderedDict

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module, as_subgraph
from max.nn.embedding import Embedding
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.nn.linear import Linear
from max.nn.norm import LayerNorm
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from ..model_config import Lfm2VlConfig
from .attention import Lfm2VlVisionAttention
from .projector import Lfm2VlMultiModalProjector


class Lfm2VlVisionEmbeddings(Module[..., Tensor]):
    """The patch Linear plus the learned position table it adds.

    HF applies the patch projection over the processor's per-patch rows
    (already flattened by `convert_image_to_patches`, channel-last: 16 x
    16 x 3 = 768) and adds the NaFlex position TERM, interpolated from
    the [num_patches, hidden] table (read as a
    sqrt(num_patches) x sqrt(num_patches) grid) with the four per-patch
    gathers and weights the batch processor supplies.
    """

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.patch_embedding = Linear(
            in_dim=config.vision_patch_dim,
            out_dim=config.vision_hidden_size,
            bias=True,
        )
        # The learned table as shipped: [256, 1152].
        self.position_embedding = Embedding(
            config.vision_num_patches, dim=config.vision_hidden_size
        )

    def forward(
        self, pixel_values: Tensor, pos_idx: Tensor, pos_w: Tensor
    ) -> Tensor:
        """Patch-embed the rows and add their bilinear position terms."""
        table = self.position_embedding.weight
        # The interpolation weights arrive fp32 (host-computed), but the
        # whole tower graph runs at the graph dtype: a bf16 table row
        # times an fp32 weight PROMOTES the position term to fp32, and
        # the post-encoder LayerNorm then sees an fp32 input against its
        # bf16 gamma — the graph compiler rejects the mixed dtype pair
        # at kernel binding. Cast the weights at the boundary; the
        # checkpoint's own reference keeps the tower uniformly bf16.
        w = F.cast(pos_w, table.dtype)
        # Same boundary rule for the pixels: the processor hands fp32
        # rows; the reference casts inputs to the model dtype on entry,
        # and an fp32 pixel row would otherwise promote the patch
        # matmul, the residual chain, and post_layernorm's input to
        # fp32 while its bf16 gamma stays bf16.
        pixels = F.cast(pixel_values, self.patch_embedding.weight.dtype)
        # Bilinear position rows from the four table entries each patch
        # reads; the gather indices and weights are host-computed.
        g00 = F.gather(table, pos_idx[:, 0], axis=0) * w[:, 0][:, None]
        g10 = F.gather(table, pos_idx[:, 1], axis=0) * w[:, 1][:, None]
        g01 = F.gather(table, pos_idx[:, 2], axis=0) * w[:, 2][:, None]
        g11 = F.gather(table, pos_idx[:, 3], axis=0) * w[:, 3][:, None]
        positions = (g00 + g10) + (g01 + g11)
        return self.patch_embedding(pixels) + positions


class _SiglipMLP(Module[..., Tensor]):
    """fc1 -> gelu_pytorch_tanh -> fc2, both with bias."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.fc1 = Linear(
            in_dim=config.vision_hidden_size,
            out_dim=config.vision_intermediate_size,
            bias=True,
        )
        self.fc2 = Linear(
            in_dim=config.vision_intermediate_size,
            out_dim=config.vision_hidden_size,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # gelu_pytorch_tanh is the tanh approximation; MAX spells it
        # approximate="tanh" (the exact-erf form is the PROJECTOR's
        # activation, not the tower's).
        """fc1 -> tanh-approximate gelu -> fc2."""
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Lfm2VlVisionBlock(Module[..., Tensor]):
    """One SigLIP2 encoder layer: pre-LN attention, pre-LN MLP.

    HF modeling_siglip2.py::Siglip2EncoderLayer — layer_norm1/layer_norm2
    (WITH bias, eps 1e-6) before each branch, residual adds around both,
    and an MLP of fc1 -> gelu_pytorch_tanh -> fc2, both with bias.
    """

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        eps = config.vision_layer_norm_eps
        self.layer_norm1 = LayerNorm(hidden, eps=eps)
        self.self_attn = Lfm2VlVisionAttention(config)
        self.layer_norm2 = LayerNorm(hidden, eps=eps)
        self.mlp = _SiglipMLP(config)

    def forward(
        self,
        x: Tensor,
        tile_offsets: Tensor,
        max_tile_len: Tensor,
    ) -> Tensor:
        """One pre-LN encoder layer: tile-local attention, then the MLP."""
        h = x + self.self_attn(self.layer_norm1(x), tile_offsets, max_tile_len)
        return h + self.mlp(self.layer_norm2(h))


class Lfm2VlVisionEncoder(Module[..., Tensor]):
    """The SigLIP2 encoder layers, named as the checkpoint spells them."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.layers = ModuleList(
            [
                Lfm2VlVisionBlock(config)
                for _ in range(config.vision_num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        tile_offsets: Tensor,
        max_tile_len: Tensor,
    ) -> Tensor:
        """Run the encoder blocks over the unpadded patch rows."""
        for layer in self.layers:
            # One compiled subgraph serves all 27 identical blocks.
            block = as_subgraph(layer, name="vision_encoder_block")
            x = block(x, tile_offsets, max_tile_len)
        return x


class Lfm2VlVisionModel(Module[..., Tensor]):
    """Patch embedding, the encoder, the final norm, and the projector."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Lfm2VlVisionEmbeddings(config)
        self.encoder = Lfm2VlVisionEncoder(config)
        self.post_layernorm = LayerNorm(
            config.vision_hidden_size, eps=config.vision_layer_norm_eps
        )
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)

    def forward(
        self,
        pixel_values: Tensor,
        pos_idx: Tensor,
        pos_w: Tensor,
        tile_offsets: Tensor,
        max_tile_len: Tensor,
        block_gather: Tensor,
    ) -> Tensor:
        """The whole vision graph: embed, encode, norm, fold, project."""
        h = self.embeddings(pixel_values, pos_idx, pos_w)
        h = self.encoder(h, tile_offsets, max_tile_len)
        h = self.post_layernorm(h)
        return self.multi_modal_projector(h, block_gather)


class Lfm2VlVision(Lfm2VlVisionModel):
    """The registered VISION role: the tower as the pipeline packs it.

    Registered under ``GraphRole.VISION`` (see ``arch.py``); the
    framework's registered pipeline model constructs it from the
    finalized config, wraps it in a graph whose inputs are the named
    groups below and hands it to the batch processor's vision cache.
    The groups are the serving engine's packing vocabulary: rows
    concatenate along the patch axis, segments concatenate after their
    first element, fold indices offset by the rows packed before them,
    and the host scalar takes the packed batch's max.
    """

    graph_name = "Lfm2VlVisionModel"

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__(config)
        # The decoder's vision input types come from these two: the merge
        # carries rows at the graph's own dtype — fp32 under the fp32
        # parity serve, never hard-cast — and hidden_size wide.
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.hidden_size

    def input_types(self) -> OrderedDict[str, object]:
        """Groups the tower's inputs as the serving engine packs them."""
        device = self.config.devices[0]
        return OrderedDict(
            patches=VisionRows(
                pixel_values=TensorType(
                    DType.float32,
                    ["num_patches", self.config.vision_patch_dim],
                    device=device,
                ),
                pos_idx=TensorType(
                    DType.uint32, ["num_patches", 4], device=device
                ),
                pos_w=TensorType(
                    DType.float32, ["num_patches", 4], device=device
                ),
            ),
            segments=VisionSegments(
                tile_offsets=TensorType(
                    DType.uint32, ["num_tiles_plus_one"], device=device
                )
            ),
            segment_bound=VisionHostMax(
                max_tile_len=TensorType(
                    DType.uint32, [1], device=DeviceRef.CPU()
                )
            ),
            fold=VisionRowIndex(
                block_gather=TensorType(
                    DType.uint32, ["num_image_tokens", 4], device=device
                )
            ),
        )
