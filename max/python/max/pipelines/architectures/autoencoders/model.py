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

from collections.abc import Callable
from typing import Any, TypeVar

from max import functional as F
from max.driver import Device
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .model_config import AutoencoderKLConfigBase

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
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the decoder model.

        Extracts decoder weights from the full model weights and compiles
        the decoder for inference.

        Returns:
            Compiled model callable.
        """
        state_dict = {
            key.removeprefix("decoder."): value.data()
            for key, value in self.weights.items()
            if not key.startswith("encoder.")
        }
        with F.lazy():
            autoencoder = self.autoencoder_class(self.config)
            autoencoder.decoder.to(self.devices[0])

        self.model = autoencoder.decoder.compile(
            *autoencoder.decoder.input_types(), weights=state_dict
        )
        return self.model

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

        This method provides a consistent interface with other MaxModel
        implementations. It is an alias for decode().

        Args:
            *args: Input arguments (typically latents as Tensor).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Decoded image tensor.
        """
        return self.decode(*args, **kwargs)
