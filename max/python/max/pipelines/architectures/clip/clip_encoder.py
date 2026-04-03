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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import LayerNorm

from .layers import CLIPVisionAttention, CLIPVisionEmbeddings
from .model_config import ClipVisionConfig


class CLIPVisionMLP(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.config = config
        self.fc1 = Linear(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.fc2 = Linear(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.fc1(hidden_states)
        if self.config.hidden_act == "quick_gelu":
            hidden_states = ops.gelu(hidden_states, approximate="quick")
        else:
            hidden_states = ops.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPVisionEncoderLayer(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.self_attn = CLIPVisionAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dtype=dtype,
            device=device,
        )
        self.layer_norm1 = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
            devices=[device],
            dtype=dtype,
            use_bias=True,
        )
        self.layer_norm2 = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
            devices=[device],
            dtype=dtype,
            use_bias=True,
        )
        self.mlp = CLIPVisionMLP(config, dtype=dtype, device=device)

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPVisionEncoder(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.layers = LayerList(
            [
                CLIPVisionEncoderLayer(
                    config,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.return_penultimate = config.return_penultimate

    @property
    def num_hidden_layers(self) -> int:
        if self.return_penultimate:
            return len(self.layers) - 1
        return len(self.layers)

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states)
        return hidden_states


class CLIPVisionTransformer(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            config,
            dtype=dtype,
            device=device,
        )
        self.pre_layrnorm = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
            devices=[device],
            dtype=dtype,
            use_bias=True,
        )
        self.encoder = CLIPVisionEncoder(
            config,
            dtype=dtype,
            device=device,
        )
        self.post_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=config.layer_norm_eps,
            devices=[device],
            dtype=dtype,
            use_bias=True,
        )

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(hidden_states)
        return hidden_states


class CLIPVisionModel(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config,
            dtype=dtype,
            device=device,
        )

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        return self.vision_model(pixel_values)
