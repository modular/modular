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

from types import SimpleNamespace
from typing import Any

from max.driver import Device
from max.graph.weights import Weights
from max.nn import Module
from max.pipelines.lib import SupportedEncoding
from max.tensor import Tensor

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLFlux2Config
from .vae import Decoder, Encoder


class AutoencoderKLFlux2(Module[[Tensor, Tensor | None], Tensor]):
    r"""A VAE model with KL loss for encoding images into latents and decoding latent representations into images."""

    def __init__(
        self,
        config: AutoencoderKLFlux2Config,
    ) -> None:
        """Initialize VAE AutoencoderKLFlux2 model.

        Args:
            config: AutoencoderKLFlux2 configuration containing channel sizes, block
                structure, normalization settings, BatchNorm parameters, and device/dtype information.
        """
        super().__init__()
        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=tuple(config.down_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            double_z=True,
            mid_block_add_attention=config.mid_block_add_attention,
            use_quant_conv=config.use_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            norm_type="group",
            mid_block_add_attention=config.mid_block_add_attention,
            use_post_quant_conv=config.use_post_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, z: Tensor, temb: Tensor | None = None) -> Tensor:
        """Apply AutoencoderKLFlux2 forward pass (decoding only).

        Args:
            z: Input latent tensor of shape [N, C_latent, H_latent, W_latent].
            temb: Optional time embedding tensor.

        Returns:
            Decoded image tensor of shape [N, C_out, H, W].
        """
        return self.decoder(z, temb)


class AutoencoderKLFlux2Model(BaseAutoencoderModel):
    """ComponentModel wrapper for AutoencoderKLFlux2.

    This class provides the ComponentModel interface for AutoencoderKLFlux2,
    handling configuration, weight loading, model compilation, and BatchNorm
    statistics for Flux2's latent patchification.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize AutoencoderKLFlux2Model.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        self.bn_running_mean: Tensor | None = None
        self.bn_running_var: Tensor | None = None

        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLFlux2Config,
            autoencoder_class=AutoencoderKLFlux2,
        )

    def load_model(self) -> Any:
        """Load and compile the decoder and encoder models with BatchNorm statistics.

        Extracts BatchNorm statistics (bn.*) which are specific to Flux2, then
        delegates to base class for weight loading and model compilation.

        Returns:
            Compiled decoder model callable.
        """
        bn_stats = {}

        for key, value in self.weights.items():
            if key in ("bn.running_mean", "bn.running_var"):
                weight_data = value.data()
                target_dtype = self.config.dtype
                if weight_data.dtype != target_dtype:
                    if weight_data.dtype.is_float() and target_dtype.is_float():
                        weight_data = weight_data.astype(target_dtype)
                    # Non-float left as-is; running_mean/var are typically float.
                bn_stats[key] = weight_data.data

        bn_mean_data = bn_stats.get("bn.running_mean")
        bn_var_data = bn_stats.get("bn.running_var")

        super().load_model()

        if bn_mean_data is not None or bn_var_data is not None:
            if bn_mean_data is not None:
                self.bn_running_mean = Tensor.from_dlpack(bn_mean_data).to(
                    self.devices[0]
                )
            if bn_var_data is not None:
                self.bn_running_var = Tensor.from_dlpack(bn_var_data).to(
                    self.devices[0]
                )

        return self.model

    @property
    def bn(self) -> SimpleNamespace:
        """Property to access BatchNorm statistics, compatible with diffusers API.

        Returns a SimpleNamespace with running_mean and running_var attributes
        for compatibility with pipeline code that accesses self.vae.bn.running_mean.

        Returns:
            SimpleNamespace: Object containing running_mean and running_var attributes.

        Raises:
            ValueError: If BatchNorm statistics are not loaded.
        """
        if self.bn_running_mean is None or self.bn_running_var is None:
            raise ValueError(
                "BatchNorm statistics (running_mean, running_var) not loaded. "
                "Make sure the model weights contain 'bn.running_mean' and 'bn.running_var'."
            )

        return SimpleNamespace(
            running_mean=self.bn_running_mean, running_var=self.bn_running_var
        )
