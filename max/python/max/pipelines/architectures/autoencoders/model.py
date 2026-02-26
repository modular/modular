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
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
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
        self.load_model()

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
            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)
                # Non-float weights (e.g. bn.num_batches_tracked int64) are left as-is and
                # skipped for decoder/encoder state dicts (no prefix match).

            if key.startswith("decoder."):
                decoder_state_dict[key.removeprefix("decoder.")] = weight_data
            elif key.startswith("post_quant_conv."):
                decoder_state_dict[key] = weight_data
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = weight_data
            elif key.startswith("quant_conv."):
                encoder_state_dict[key] = weight_data

        with F.lazy():
            autoencoder = self.autoencoder_class(self.config)

            autoencoder.decoder.to(self.devices[0])
            self.model = autoencoder.decoder.compile(
                *autoencoder.decoder.input_types(), weights=decoder_state_dict
            )
            # Flux.1 does not have an encoder.
            if encoder_state_dict and hasattr(autoencoder, "encoder"):
                autoencoder.encoder.to(self.devices[0])
                self.encoder_model = autoencoder.encoder.compile(
                    *autoencoder.encoder.input_types(),
                    weights=encoder_state_dict,
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

        moments = self.encoder_model(sample)
        posterior = DiagonalGaussianDistribution(moments)

        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        """Decode latents to images using compiled decoder.

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].

        Returns:
            Tensor: Decoded image tensor.
        """
        return self.model(z)

    def __call__(self, z: Tensor) -> Tensor:
        """Call the decoder model to decode latents to images.

        This method provides a consistent interface with other ComponentModel
        implementations. It is an alias for decode().

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].

        Returns:
            Tensor: Decoded image tensor.
        """
        return self.decode(z)
