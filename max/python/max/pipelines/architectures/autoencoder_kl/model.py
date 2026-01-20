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

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.max_model import MaxModel

from .autoencoder_kl import AutoencoderKL
from .model_config import AutoencoderKLConfig


class AutoencoderKLModel(MaxModel):
    config_name = AutoencoderKLConfig.config_name

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderKLConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> None:
        state_dict = {
            key.removeprefix("decoder."): value.data()
            for key, value in self.weights.items()
            if not key.startswith("encoder.")
        }
        with F.lazy():
            autoencoder_kl = AutoencoderKL(self.config)
            autoencoder_kl.decoder.to(self.devices[0])

        self.model = autoencoder_kl.decoder.compile(
            *autoencoder_kl.decoder.input_types(), weights=state_dict
        )

    def decode(self, *args, **kwargs) -> Tensor:
        """Decode latents to images using module_v3 compiled decoder.

        Args:
            *args: Input arguments (typically latents as Tensor).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Decoded image tensor (module_v3 Tensor, V3).
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
            Tensor: Decoded image tensor (module_v3 Tensor, V3).
        """
        return self.decode(*args, **kwargs)
