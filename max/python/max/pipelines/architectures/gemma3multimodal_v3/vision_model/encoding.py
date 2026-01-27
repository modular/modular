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

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.nn.legacy.norm import LayerNorm
from max.tensor import Tensor

from ..model_config import Gemma3ForConditionalGenerationConfig
from .attention import Gemma3VisionAttention
from .projection import Gemma3VisionMLP


class Gemma3VisionEncoderLayer(Module[[Tensor], Tensor]):
    """An individual layer of encoding within a stack of encoding layers"""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ):
        """prepare the two normalisation layers, the self attention, and the
        multi-layer perceptrion"""
        self.config = config
        vision_config = config.vision_config
        vision_dtype = DType.bfloat16

        self.device = device if device is not None else config.devices[0]
        self.embed_dim = vision_config.hidden_size
        self.layer_idx = layer_idx

        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            devices=[self.device],
            dtype=vision_dtype,
        )

        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.mlp = Gemma3VisionMLP(config, device=self.device)

        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            devices=[self.device],
            dtype=vision_dtype,
        )

    def __call__(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """process the input hidden states through each of the sub-layers"""
        residual = hidden_states
        hidden_states_tv = self.layer_norm1(hidden_states.__tensorvalue__())
        hidden_states = self.self_attn(
            Tensor.from_graph_value(hidden_states_tv)
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states_tv = self.layer_norm2(hidden_states.__tensorvalue__())
        hidden_states = self.mlp(Tensor.from_graph_value(hidden_states_tv))
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3VisionEncoder(
    Module[[Tensor | Sequence[Tensor]], Tensor | Sequence[Tensor]]
):
    """Wrapper class for a stack of vision encoder layers"""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        """Intialise the stack of encoder layers based on config"""
        super().__init__()
        self.config = config
        self.devices = config.devices

        encoder_layers = [
            Gemma3VisionEncoderLayer(config, layer_idx)
            for layer_idx in range(config.vision_config.num_hidden_layers)
        ]

        self.layers = encoder_layers  # FIXME trying this approach
        # self.layers = Sequential(*encoder_layers) #TODO could this be used

    def __call__(
        self,
        hidden_states: Tensor | Sequence[Tensor],
    ) -> Tensor | Sequence[Tensor]:
        """Process hidden states through the stack of encoder layers"""
        # if hidden_states is a list, we are sharding across devices.  each device has a replication of the weights
        # TODO this is a temporary workaround while working single GPU and
        # keeping type checking happy
        if isinstance(hidden_states, Sequence):
            outputs = []
            for device_idx, state in enumerate(hidden_states):
                device_output = state
                for layer in self.layers_per_device[device_idx]:
                    device_output = layer(device_output)
                outputs.append(device_output)
            return outputs
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            # hidden_states = self.layers(hidden_states) #TODO could this be used
        return hidden_states
