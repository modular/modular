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
from __future__ import annotations

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Conv1D, Embedding, LayerNorm, Linear, Sequential
from max.nn.layer import Module
from transformers import AutoConfig


class WhisperSdpaAttention(Module):
    def __init__(
        self, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        d_model = huggingface_config.d_model
        self.n_heads = huggingface_config.n_heads
        self.head_dim = d_model // huggingface_config.encoder_attention_heads

        self.wq = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wk = Linear(d_model, d_model, dtype, device, has_bias=False)
        self.wv = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wo = Linear(d_model, d_model, dtype, device, has_bias=True)

    def scaled_dot_product_attention(
        self, xq: TensorValue, xk: TensorValue, xv: TensorValue
    ) -> TensorValue:
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        return ops.softmax(scores * scale) @ xv

    def __call__(self, x: TensorValue) -> TensorValue:
        """Computes attention on x.

        Args:
            x: Activations with shape (batch, seq_len, dim).

        Returns the result of WhisperSdpaAttention self attention on the input.
        """
        batch, seq_len = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, [batch, seq_len, self.n_heads, self.head_dim])
        xk = ops.reshape(xk, [batch, seq_len, self.n_heads, self.head_dim])
        xv = ops.reshape(xv, [batch, seq_len, self.n_heads, self.head_dim])

        output = (
            self.scaled_dot_product_attention(xq, xk, xv)
            .transpose(1, 2)
            .reshape([batch, seq_len, -1])
        )
        return self.wo(output)


class MLP(Module):
    def __init__(
        self, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.fc1 = Linear(
            huggingface_config.d_model,
            huggingface_config.encoder_ffn_dim,
            dtype,
            device,
            has_bias=True,
        )
        self.fc2 = Linear(
            huggingface_config.encoder_ffn_dim,
            huggingface_config.d_model,
            dtype,
            device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x


class WhisperEncoderLayer(Module):
    """Stack of Attention, FeedForward, and LayerNorm layers."""

    def __init__(
        self, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.attention = WhisperSdpaAttention(huggingface_config, dtype, device)
        self.mlp = MLP(huggingface_config, dtype, device)
        self.attention_norm = LayerNorm(
            huggingface_config.d_model, device, dtype, eps=1e-5
        )
        self.mlp_norm = LayerNorm(
            huggingface_config.d_model, device, dtype, eps=1e-5
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        attn_out = self.attention(self.attention_norm(x))

        h = x + attn_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


class WhisperEncoder(Module):
    """A Transformer consisting of a stem, positional embeddings, and self attention layers.

    The differences between this transformer and `nn.Transformer` are:
        1. Whisper passes the input through a stem of:
        Two convolution layers with a filter width of 3 and the GELU activation
        function where the second convolution layer has a stride of two.

        2. After that, Sinusoidal position embeddings are then added to the output of the stem.

        After that, the usual Transformer blocks (with pre-activation residual blocks) are applied.
        We use naive attention where the linear projections have a bias.

        3. No final transformer linear layer "output".
    """

    def __init__(
        self, huggingface_config: AutoConfig, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.conv1 = Conv1D(
            kernel_size=3,
            in_channels=huggingface_config.num_mel_bins,
            out_channels=huggingface_config.d_model,
            dtype=dtype,
            stride=1,
            padding=1,
            device=device,
            has_bias=True,
        )
        self.conv2 = Conv1D(
            kernel_size=3,
            in_channels=huggingface_config.d_model,
            out_channels=huggingface_config.d_model,
            dtype=dtype,
            stride=2,
            padding=1,
            device=device,
            has_bias=True,
        )
        # TODO: Not sure how to handle this. It learns embeddings to a max size.
        self.embed_positions = Embedding(
            vocab_size=huggingface_config.max_source_positions,
            hidden_dim=huggingface_config.d_model,
            dtype=dtype,
            device=device,
        )
        self.layers = Sequential(
            [
                WhisperEncoderLayer(huggingface_config, dtype, device)
                for i in range(huggingface_config.encoder_layers)
            ]
        )
        # Hugging Face model uses default eps for nn.LayerNormV1 which is = 1e-5
        # TODO: Is LayerNorm here not the same as nn.LayerNorm
        self.norm = LayerNorm(
            huggingface_config.d_model, device, dtype, eps=1e-5
        )

    def __call__(self, input_features: TensorValue) -> tuple[TensorValue, ...]:
        """
        Args:
            input_features: Tensor of shape (batch_size, feature_size, sequence_length)
        """
        # Encoder stem: two convolution layers and the GELU activation function.
        inputs_embeds = ops.gelu(self.conv1(input_features))
        inputs_embeds = ops.gelu(self.conv2(inputs_embeds))

        # self.embed_positions.weights layers is of shape = (1500, 1280)
        # TODO: Do we need the reshape to (batch_size, sequence_length, feature_size) or is it already in the right shape?
        # inputs_embeds = ops.permute(inputs_embeds, [0, 2, 1])

        # Add sinusoidal position embeddings to the output of the stem
        h = inputs_embeds + self.embed_positions.weight

        h = self.layers(h)

        # # A final layer normalization is applied to the encoder output
        normalized = self.norm(h)

        # Always return float32 logits, no matter the activation type.
        return (ops.cast(normalized, DType.float32),)
