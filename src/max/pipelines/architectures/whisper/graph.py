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


import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from max.nn import Conv1D, Embedding, LayerNorm, Linear, Sequential
from max.pipelines import PipelineConfig
from transformers import AutoConfig

from .encoder import WhisperEncoder, WhisperEncoderLayer, WhisperSdpaAttention


def conv1d(
    dtype: DType,
    in_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    out_channels: int,
    weights: SafetensorWeights,
) -> Conv1D:
    """Creates a 1D convolution layer.
    For conv1: ( hugging_face weights: model.encoder.conv1.weight)
    in_channels = 128
    out_channels = 1280
    kernel_size = 3
    stride = 1
    padding = 1

    For conv2: ( hugging_face weights: model.encoder.conv2.weight)
    in_channels = 1280
    out_channels = 1280
    kernel_size = 3
    stride = 2
    padding = 1
    """
    # Loaded torch weights shape = (out_channels, in_channels, kernel_size) = [1280, 128, 3].
    # Graph-API Conv1D expects (kernel_size, in_channels, out_channels) = [3, 128, 1280].
    # TODO: Implement Conv1D with bias and use it here.
    bias = weights.bias.allocate(dtype, [out_channels])
    return Conv1D(
        filter=ops.permute(
            weights.weight.allocate(
                dtype, [out_channels, in_channels, 1, kernel_size], None
            ),
            [2, 1, 0],
        ),
        stride=stride,
        padding=padding,
    )


def embedding(
    dtype: DType,
    max_source_positions: int,
    hidden_dim: int,
    weights: SafetensorWeights,
):
    return Embedding(
        weights.weight.allocate(
            dtype,
            [max_source_positions, hidden_dim],
        )
    )


def layer_norm(dims: int, eps: float, weights: SafetensorWeights) -> LayerNorm:
    # TODO: check the shape of bias
    return LayerNorm(
        weight=weights.weight.allocate(DType.bfloat16, [dims]),
        eps=eps,
        bias=weights.bias.allocate(DType.bfloat16, [dims]),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    # TODO: Check we are passing the correct dim for bias
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None),
        bias=weights.bias.allocate(dtype, [out_features], None),
    )


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return Sequential(
        layers=[
            linear(
                dtype,
                feed_forward_length,
                hidden_dim,
                weights.fc1,
            ),
            ops.gelu,  # type: ignore
            linear(
                dtype,
                hidden_dim,
                feed_forward_length,
                weights.fc2,
            ),
        ]
    )


def attention(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    layer_index: int,
    huggingface_config: AutoConfig,
    dtype: DType,
):
    wq = weights.self_attn.q_proj.weight.allocate(
        dtype,
        [
            huggingface_config.d_model,
            huggingface_config.d_model,
        ],
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        dtype,
        [
            huggingface_config.d_model,
            huggingface_config.d_model,
        ],
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        dtype,
        [
            huggingface_config.d_model,
            huggingface_config.d_model,
        ],
    )

    bias_q = weights.self_attn.q_proj.bias.allocate(
        dtype, [huggingface_config.d_model]
    )
    bias_v = weights.self_attn.v_proj.bias.allocate(
        dtype, [huggingface_config.d_model]
    )
    bias_k = ops.constant(
        np.zeros(huggingface_config.d_model),
        dtype,
    )

    wo = weights.attn_output.weight.allocate(
        dtype,
        [
            huggingface_config.d_model,
            huggingface_config.d_model,
        ],
    )
    bias_o = weights.self_attn.out_proj.bias.allocate(
        dtype, [huggingface_config.d_model]
    )
    return WhisperSdpaAttention(
        n_heads=huggingface_config.n_heads,
        head_dim=huggingface_config.d_model
        // huggingface_config.encoder_attention_heads,
        wq=Linear(wq, bias=bias_q),
        wk=Linear(wk, bias=bias_k),
        wv=Linear(wv, bias=bias_v),
        wo=Linear(
            wo,
            bias=bias_o,
        ),
    )


def encoder(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    huggingface_config: AutoConfig,
    dtype: DType,
) -> WhisperEncoder:
    conv1 = conv1d(
        dtype=dtype,
        in_channels=huggingface_config.num_mel_bins,
        kernel_size=3,
        stride=1,
        padding=1,
        out_channels=huggingface_config.d_model,
        weights=weights.model.encoder.conv1,
    )

    conv2 = conv1d(
        dtype=dtype,
        in_channels=huggingface_config.d_model,
        kernel_size=3,
        stride=2,
        padding=1,
        out_channels=huggingface_config.d_model,
        weights=weights.model.encoder.conv2,
    )

    # TODO: Not sure how to handle this. It learns embeddings to a max size.
    embed_positions = embedding(
        dtype=dtype,
        max_source_positions=huggingface_config.max_source_positions,
        hidden_dim=huggingface_config.d_model,
        weights=weights.model.encoder.embed_positions,
    )

    # EncoderBlocks
    # TODO: Which cache strategy to use? Will both Continuous and paged will work?
    layers = [
        WhisperEncoderLayer(
            attention=attention(
                pipeline_config,
                weights.language_model.model.layers[i],
                layer_index=i,
                huggingface_config=huggingface_config,
                dtype=dtype,
            ),
            mlp=feed_forward(
                dtype,
                huggingface_config.d_model,
                huggingface_config.encoder_ffn_dim,
                weights.model.encoder.layers[i],
            ),
            attention_norm=layer_norm(
                dims=huggingface_config.d_model,
                eps=1e-5,
                weights=weights.model.encoder.layers[i].self_attn_layer_norm,
            ),
            mlp_norm=layer_norm(
                dims=huggingface_config.d_model,
                eps=1e-5,
                weights=weights.model.encoder.layers[i].final_layer_norm,
            ),
        )
        for i in range(huggingface_config.encoder_layers)
    ]

    # Hugging Face model uses default eps for nn.LayerNorm which is = 1e-5
    norm = layer_norm(
        dims=huggingface_config.d_model,
        eps=1e-5,
        weights=weights.model.encoder.layer_norm,
    )

    return WhisperEncoder(
        conv1=conv1,
        conv2=conv2,
        embed_positions=embed_positions,
        layers=layers,
        norm=norm,
        all_logits=False,
    )


def build_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    huggingface_config: AutoConfig,
    dtype: DType,
) -> Graph:
    # Audio input_features.
    input_features_type = TensorType(
        DType.float32,
        shape=["batch_size", "num_mel_bins", "sequence_length"],
        device=DeviceRef.CPU(),
    )

    # Initialize Graph.
    with Graph(
        "whisper_audio_encoder",
        input_types=[
            input_features_type,
        ],
    ) as graph:
        model = encoder(
            pipeline_config,
            weights,
            huggingface_config=huggingface_config,
            dtype=dtype,
        )
        input_features = graph.inputs[0]
        outputs = model(
            input_features=input_features.tensor,
        )
        graph.output(*outputs)
        return graph
