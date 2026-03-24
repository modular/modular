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

from collections.abc import Callable, Mapping
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import WeightData, Weights
from max.nn.layer import Module
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import AutoencoderKLConfigBase
from .vae import DiagonalGaussianDistribution


class BaseAutoencoderModel(ComponentModel):
    """Base class for graph-based Module V2 autoencoders."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        config_class: type[AutoencoderKLConfigBase],
        autoencoder_class: type,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, encoding, devices, weights, **kwargs)
        self.config = config_class.generate(config, encoding, devices)  # type: ignore[attr-defined]
        self.autoencoder_class = autoencoder_class
        self.session = InferenceSession(devices=[*devices])
        self.encoder_model: Callable[..., Any] | None = None
        self.load_model()

    @staticmethod
    def _unwrap_single(output: Any) -> Any:
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def _compile_module(
        self,
        module: Module,
        input_types: tuple[TensorType, ...],
        state_dict: Mapping[str, Buffer | WeightData | Any],
        graph_name: str,
    ) -> Callable[..., Any]:
        normalized_state_dict = dict(state_dict)
        for name, weight in module.raw_state_dict().items():
            if name not in normalized_state_dict:
                continue
            value = normalized_state_dict[name]
            value_dtype = getattr(value, "dtype", None)
            if value_dtype != weight.dtype:
                if (
                    value_dtype is not None
                    and value_dtype.is_float()
                    and weight.dtype.is_float()
                    and hasattr(value, "astype")
                ):
                    normalized_state_dict[name] = value.astype(weight.dtype)

        module.load_state_dict(
            normalized_state_dict,
            weight_alignment=1,
            strict=True,
        )
        weights_registry = module.state_dict(auto_initialize=False)

        with Graph(graph_name, input_types=input_types) as graph:
            output = module(*(value.tensor for value in graph.inputs))
            if isinstance(output, (list, tuple)):
                graph.output(*output)
            else:
                graph.output(output)

        model: Model = self.session.load(
            graph, weights_registry=weights_registry
        )
        return model.execute

    def load_model(self) -> Callable[..., Any]:
        decoder_state_dict = {}
        encoder_state_dict = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)

            if key.startswith("decoder."):
                decoder_state_dict[key.removeprefix("decoder.")] = weight_data
            elif key.startswith("post_quant_conv."):
                decoder_state_dict[key] = weight_data
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = weight_data
            elif key.startswith("quant_conv."):
                encoder_state_dict[key] = weight_data

        autoencoder = self.autoencoder_class(self.config)
        self.model = self._compile_module(
            autoencoder.decoder,
            autoencoder.decoder.input_types(),
            decoder_state_dict,
            type(autoencoder.decoder).__name__.lower(),
        )
        if encoder_state_dict and hasattr(autoencoder, "encoder"):
            self.encoder_model = self._compile_module(
                autoencoder.encoder,
                autoencoder.encoder.input_types(),
                encoder_state_dict,
                type(autoencoder.encoder).__name__.lower(),
            )
        return self.model

    def encode(
        self, sample: Buffer, return_dict: bool = True
    ) -> dict[str, DiagonalGaussianDistribution] | DiagonalGaussianDistribution:
        if self.encoder_model is None:
            raise ValueError(
                "Encoder not loaded. Check if encoder weights exist in the model."
            )

        moments = self._unwrap_single(self.encoder_model(sample))
        posterior = DiagonalGaussianDistribution(moments, moments)
        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    def decode(self, z: Buffer) -> Buffer:
        return self._unwrap_single(self.model(z))

    def __call__(self, z: Buffer) -> Buffer:
        return self.decode(z)
