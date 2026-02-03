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

from collections.abc import Callable
from typing import Any, TypeVar

from max import functional as F
from max.driver import Device
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.nn import Conv2d, Module
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .model_config import AutoencoderKLConfigBase
from .vae import DiagonalGaussianDistribution

TConfig = TypeVar("TConfig", bound=AutoencoderKLConfigBase)


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
        config_class: type[TConfig],
        autoencoder_class: type,
    ) -> None:
        """Initialize base autoencoder model.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
            config_class: Configuration class to use (e.g., AutoencoderKLConfig).
            autoencoder_class: Autoencoder class to use (e.g., AutoencoderKL).
        """
        super().__init__(config, encoding, devices, weights)
        self.config = config_class.generate(config, encoding, devices)  # type: ignore[attr-defined]
        self.autoencoder_class = autoencoder_class
        self.encoder_model: Callable[[Tensor], Tensor] | None = None
        self.quant_conv_model: Callable[[Tensor], Tensor] | None = None
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the decoder and encoder models.

        Extracts decoder weights (decoder.*, post_quant_conv.*) and encoder weights
        (encoder.*, quant_conv.*) from the full model weights and compiles them
        for inference. Encoder is optional (only loaded if weights exist).

        Returns:
            Compiled decoder model callable.
        """
        decoder_state_dict = {}
        encoder_state_dict = {}
        quant_conv_state_dict = {}

        for key, value in self.weights.items():
            if key.startswith("decoder."):
                decoder_state_dict[key.removeprefix("decoder.")] = value.data()
            elif key.startswith("post_quant_conv."):
                decoder_state_dict[key] = value.data()
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = value.data()
            elif key.startswith("quant_conv."):
                quant_conv_state_dict[key] = value.data()

        with F.lazy():
            autoencoder = self.autoencoder_class(self.config)

            autoencoder.decoder.to(self.devices[0])
            self.model = autoencoder.decoder.compile(
                *autoencoder.decoder.input_types(), weights=decoder_state_dict
            )

            if encoder_state_dict and hasattr(autoencoder, "encoder"):
                autoencoder.encoder.to(self.devices[0])
                self.encoder_model = autoencoder.encoder.compile(
                    *autoencoder.encoder.input_types(),
                    weights=encoder_state_dict,
                )

            if (
                quant_conv_state_dict
                and hasattr(autoencoder, "quant_conv")
                and autoencoder.quant_conv is not None
                and self.encoder_model is not None
            ):

                class QuantConvModule(Module[[Tensor], Tensor]):
                    def __init__(self, quant_conv: Conv2d) -> None:
                        super().__init__()
                        self.quant_conv = quant_conv

                    def forward(self, x: Tensor) -> Tensor:
                        return self.quant_conv(x)

                quant_conv_module = QuantConvModule(autoencoder.quant_conv)
                quant_conv_module.to(self.devices[0])
                quant_conv_input_type = TensorType(
                    self.config.dtype,
                    shape=[
                        "batch_size",
                        2 * self.config.latent_channels,
                        "latent_height",
                        "latent_width",
                    ],
                    device=DeviceRef.from_device(self.devices[0]),
                )
                self.quant_conv_model = quant_conv_module.compile(
                    quant_conv_input_type, weights=quant_conv_state_dict
                )

        return self.model

    def encode(
        self, sample: Tensor, return_dict: bool = True
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

        h = self.encoder_model(sample)

        if self.quant_conv_model is not None:
            moments = self.quant_conv_model(h)
        else:
            moments = h

        posterior = DiagonalGaussianDistribution(moments)

        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    def decode(self, *args, **kwargs) -> Tensor:
        """Decode latents to images using compiled decoder.

        Args:
            *args: Input arguments (typically latents as Tensor).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Decoded image tensor.
        """
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the decoder model to decode latents to images.

        This method provides a consistent interface with other ComponentModel
        implementations. It is an alias for decode().

        Args:
            *args: Input arguments (typically latents as Tensor).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Decoded image tensor.
        """
        return self.decode(*args, **kwargs)
