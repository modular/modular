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


from max.dtype import DType
from max.graph import Graph, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.nn import Conv2D, Linear, RMSNorm

from .attention import Attention
from .rotary_embedding_2d import RotaryEmbedding2D
from .transformer import MLP, Transformer, TransformerBlock
from .vision_encoder import VisionEncoder


def _patch_conv2d(
    dtype: DType,
    in_channels: int,
    patch_size: int,
    out_channels: int,
    weights: SafetensorWeights,
) -> Conv2D:
    """Creates a 2D convolution layer with the following assumptions:
    - kernel size = (patch_size, patch_size)
    - stride = (patch_size, patch_size)
    - padding = (0, 0, 0, 0)

    This convolution splits the image into patches and then learns an embedding
    of each patch. The embedding dim is out_channels.
    """
    # Loaded torch weights shape = torch.Size([1024, 3, 16, 16]).
    # Conv2D expects (height, width, in_channels, out_channels) = [16, 16, 3, 1024].
    weights = ops.permute(  # type: ignore
        weights.weight.allocate(
            dtype, [out_channels, in_channels, patch_size, patch_size], None
        ),
        [2, 3, 1, 0],
    )
    return Conv2D(
        weights,  # type: ignore
        stride=(patch_size, patch_size),
    )


def _linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


def _feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return MLP(
        _linear(  # gate_proj
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.gate_proj,
        ),
        _linear(  # down_proj
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.feed_forward.down_proj,
        ),
        _linear(  # up_proj
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.up_proj,
        ),
    )


def _rms_norm(dims: int, eps: float, weights: SafetensorWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def _encoder_attention(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
) -> Attention:
    # TODO: Do we need to transpose weights? Not obvious from shapes. Both dims are the same.
    hidden_dim = pipeline_config.huggingface_config.vision_config.hidden_size
    wq = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.q_proj,
    )
    wk = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.k_proj,
    )
    wv = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.v_proj,
    )
    wo = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.o_proj,
    )

    return Attention(
        n_heads=pipeline_config.huggingface_config.vision_config.num_attention_heads,
        dim=hidden_dim,
        head_dim=pipeline_config.huggingface_config.vision_config.head_dim,
        dropout=pipeline_config.huggingface_config.vision_config.attention_dropout,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
    )


def _transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
):
    with graph:
        layers = [
            TransformerBlock(
                attention=_encoder_attention(
                    pipeline_config, weights.layers[i]
                ),
                mlp=_feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.huggingface_config.vision_config.hidden_size,
                    pipeline_config.huggingface_config.vision_config.intermediate_size,
                    weights.layers[i],
                ),
                attention_norm=_rms_norm(
                    pipeline_config.huggingface_config.vision_config.hidden_size,
                    1e-5,
                    weights.layers[i].attention_norm,
                ),
                mlp_norm=_rms_norm(
                    pipeline_config.huggingface_config.vision_config.hidden_size,
                    1e-5,
                    weights.layers[i].ffn_norm,
                ),
            )
            for i in range(
                pipeline_config.huggingface_config.vision_config.num_hidden_layers
            )
        ]

        return Transformer(
            n_heads=pipeline_config.huggingface_config.vision_config.num_attention_heads,
            layers=layers,
            dtype=pipeline_config.dtype,
        )


def _vision_encoder(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
) -> VisionEncoder:
    patch_conv = _patch_conv2d(
        pipeline_config.dtype,
        pipeline_config.huggingface_config.vision_config.num_channels,
        pipeline_config.huggingface_config.vision_config.patch_size,
        pipeline_config.huggingface_config.vision_config.hidden_size,
        weights.vision_tower.patch_conv,
    )
    ln_pre = _rms_norm(
        pipeline_config.huggingface_config.vision_config.hidden_size,
        1e-5,
        weights.vision_tower.ln_pre,
    )
    patch_rope = RotaryEmbedding2D(
        dim=pipeline_config.huggingface_config.vision_config.hidden_size,
        n_heads=pipeline_config.huggingface_config.vision_config.num_attention_heads,
        theta=pipeline_config.huggingface_config.vision_config.rope_theta,
        max_patches_per_side=pipeline_config.huggingface_config.vision_config.image_size
        // pipeline_config.huggingface_config.vision_config.patch_size,
    )
    encoder_transformer = _transformer(
        graph, pipeline_config, weights.vision_tower.transformer
    )

    return VisionEncoder(
        patch_conv=patch_conv,
        layer_norm=ln_pre,
        patch_positional_embedding=patch_rope,
        transformer=encoder_transformer,
        dtype=pipeline_config.dtype,
        patch_size=pipeline_config.huggingface_config.vision_config.patch_size,
        max_image_size=pipeline_config.huggingface_config.vision_config.image_size,
    )
