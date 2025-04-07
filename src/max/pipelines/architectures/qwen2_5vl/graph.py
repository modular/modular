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
from max.nn import MLP, Conv3D, Linear, RMSNorm, Sequential
from transformers import AutoConfig

from .nn.visual_transformer import (
    PatchMerger,
    VisionBlock,
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    VisionTransformer,
    VisionWindowSdpaAttention,
)


def patch_embed(
    dtype: DType,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    embed_dim: int,
    weights: Weights,
) -> VisionPatchEmbed:
    kernel_size = (temporal_patch_size, patch_size, patch_size)
    filter_weights = ops.permute(
        weights.weight.allocate(
            dtype,
            [
                embed_dim,
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


def rotary_embedding_3d(
    hidden_size: int,
    num_heads: int,
    theta: float,
) -> VisionRotaryEmbedding:
    return VisionRotaryEmbedding(
        dim=hidden_size, n_heads=num_heads, theta=theta
    )


def linear_with_bias(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: Weights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype,
            [in_features, out_features],
        ),
        bias=weights.bias.allocate(
            dtype,
            [out_features],
        ),
    )


def mlp_with_bias(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
) -> MLP:
    return MLP(
        linear_with_bias(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.mlp.gate_proj,
        ),
        linear_with_bias(
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.mlp.down_proj,
        ),
        linear_with_bias(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.mlp.up_proj,
        ),
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(
        weight=weights.weight.allocate(DType.float32, [dims]),
        eps=eps,
    )


def vision_window_attention(
    dtype: DType,
    hidden_size: int,
    num_heads: int,
    weights: Weights,
) -> VisionWindowSdpaAttention:
    qkv = linear_with_bias(dtype, hidden_size, hidden_size, weights.qkv)
    proj = linear_with_bias(
        dtype,
        in_features=hidden_size,
        out_features=hidden_size * 3,
        weights=weights.proj,
    )
    return VisionWindowSdpaAttention(hidden_size, num_heads, qkv, proj)


def vision_transformer_block(
    dtype: DType,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    rms_norm_eps: float,
    weights: Weights,
) -> VisionBlock:
    return VisionBlock(
        norm1=rms_norm(
            hidden_size,
            rms_norm_eps,
            weights.norm1,
        ),
        norm2=rms_norm(
            hidden_size,
            rms_norm_eps,
            weights.norm2,
        ),
        attn=vision_window_attention(
            dtype,
            hidden_size,
            num_heads,
            weights.attn,
        ),
        mlp=mlp_with_bias(
            dtype,
            hidden_size,
            intermediate_size,
            weights.mlp,
        ),
    )


def merger(
    dtype: DType,
    hidden_size: int,
    out_hidden_size: int,
    spatial_merge_size: int,
    weights: Weights,
) -> PatchMerger:
    norm = rms_norm(hidden_size, 1e-6, weights.ln_q)
    hidden_size = hidden_size * (spatial_merge_size**2)
    mlp = Sequential(
        [
            linear_with_bias(
                dtype=dtype,
                in_features=hidden_size,
                out_features=hidden_size,
                weights=weights.mlp[0],
            ),
            ops.gelu,  # type: ignore
            linear_with_bias(
                dtype=dtype,
                in_features=hidden_size,
                out_features=out_hidden_size,
                weights=weights.mlp[2],
            ),
        ]
    )
    return PatchMerger(
        norm=norm,
        mlp=mlp,
        dim=out_hidden_size,
    )


def vision_transformer(
    dtype: DType,
    huggingface_config: AutoConfig,
    weights: Weights,
) -> VisionTransformer:
    patch_embed_layer = patch_embed(
        dtype=dtype,
        patch_size=huggingface_config.vision_config.patch_size,
        temporal_patch_size=huggingface_config.vision_config.temporal_patch_size,
        in_channels=huggingface_config.vision_config.in_chans,
        embed_dim=huggingface_config.vision_config.hidden_size,
        weights=weights.patch_embed.proj,
    )
    rotary_pos_emb_layer = rotary_embedding_3d(
        hidden_size=huggingface_config.vision_config.hidden_size,
        num_heads=huggingface_config.vision_config.num_heads,
        theta=10000.0,
    )
    blocks = [
        vision_transformer_block(
            dtype=dtype,
            hidden_size=huggingface_config.vision_config.hidden_size,
            num_heads=huggingface_config.vision_config.num_heads,
            intermediate_size=huggingface_config.vision_config.intermediate_size,
            rms_norm_eps=1e-6,
            weights=weights.blocks[i],
        )
        for i in range(huggingface_config.vision_config.depth)
    ]
    merger_layer = merger(
        dtype=dtype,
        hidden_size=huggingface_config.vision_config.hidden_size,
        out_hidden_size=huggingface_config.vision_config.out_hidden_size,
        spatial_merge_size=huggingface_config.vision_config.spatial_merge_size,
        weights=weights.merger,
    )

    return VisionTransformer(
        patch_embed=patch_embed_layer,
        rotary_pos_emb=rotary_pos_emb_layer,
        blocks=blocks,
        fullatt_block_indexes=huggingface_config.vision_config.fullatt_block_indexes,
        spatial_merge_unit=huggingface_config.vision_config.spatial_merge_size
        * huggingface_config.vision_config.spatial_merge_size,
        merger=merger_layer,
    )
