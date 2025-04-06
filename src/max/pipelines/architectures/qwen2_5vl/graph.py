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
from max.nn import MLP, Conv3D, Linear, RMSNorm
from transformers import AutoConfig

from .nn.visual_transformer import (
    VisionBlock,
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    VisionWindowSdpaAttention,
)


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
    huggingface_config: AutoConfig,
    weights: Weights,
) -> VisionBlock:
    return VisionBlock(
        norm1=rms_norm(
            huggingface_config.hidden_size,
            huggingface_config.rms_norm_eps,
            weights.norm1,
        ),
        norm2=rms_norm(
            huggingface_config.hidden_size,
            huggingface_config.rms_norm_eps,
            weights.norm2,
        ),
        attn=vision_window_attention(
            dtype,
            huggingface_config.hidden_size,
            huggingface_config.num_heads,
            weights.attn,
        ),
        mlp=mlp_with_bias(
            dtype,
            huggingface_config.hidden_size,
            huggingface_config.intermediate_size,
            weights.mlp,
        ),
    )
