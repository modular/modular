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
    """Base class for autoencoder models with shared logic.

    This base class provides common functionality for loading and running
    autoencoder decoders. Subclasses should specify the config and autoencoder
    classes to use.
    """

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
        """Initialize base autoencoder model.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
            config_class: Configuration class to use.
            autoencoder_class: Autoencoder class to use.
            **kwargs: Additional keyword arguments forwarded to ComponentModel.
        """
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
        """Load and compile decoder and encoder from full model weights.

        Splits weights by prefix (decoder/post_quant_conv vs encoder/quant_conv)
        and compiles each subgraph. quant_conv is included in the encoder when
        config.use_quant_conv is True. Encoder is compiled only when the model
        has an encoder and encoder weights are present.

        Returns:
            Compiled decoder model callable.
        """
        decoder_state_dict = {}
        encoder_state_dict = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            adapted_key = key
            # Some checkpoints nest VAE params under a top-level module prefix.
            # Normalize to raw autoencoder names before routing to encoder/decoder.
            while adapted_key.startswith(("vae.", "model.")):
                if adapted_key.startswith("vae."):
                    adapted_key = adapted_key.removeprefix("vae.")
                    continue
                adapted_key = adapted_key.removeprefix("model.")

            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)
                # Non-float weights are left as-is and skipped for decoder/encoder
                # state dicts if their prefixes do not match.

            if adapted_key.startswith("decoder."):
                decoder_state_dict[adapted_key.removeprefix("decoder.")] = (
                    weight_data
                )
            elif adapted_key.startswith("post_quant_conv."):
                decoder_state_dict[adapted_key] = weight_data
            elif adapted_key.startswith("encoder."):
                encoder_state_dict[adapted_key.removeprefix("encoder.")] = (
                    weight_data
                )
            elif adapted_key.startswith("quant_conv."):
                encoder_state_dict[adapted_key] = weight_data

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
        """Encode images to latent distribution using compiled encoder.

        Args:
            sample: Input image tensor of shape [N, C_in, H, W].
            return_dict: If True, returns a dictionary with "latent_dist" key.
                If False, returns DiagonalGaussianDistribution directly.

        Returns:
            If return_dict=True: Dictionary with "latent_dist" key containing
                DiagonalGaussianDistribution.
            If return_dict=False: DiagonalGaussianDistribution directly.

        Raises:
            ValueError: If encoder is not loaded.
        """
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
        """Decode latents to images using compiled decoder.

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].

        Returns:
            Decoded image tensor.
        """
        return self._unwrap_single(self.model(z))

    def __call__(self, z: Buffer) -> Buffer:
        """Call the decoder model to decode latents to images.

        This method provides a consistent interface with other ComponentModel
        implementations. It is an alias for decode().

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].

        Returns:
            Decoded image tensor.
        """
        return self.decode(z)
