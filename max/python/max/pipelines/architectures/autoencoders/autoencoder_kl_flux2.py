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

from typing import Any, ClassVar

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.nn import Conv2d, Module
from max.pipelines.lib import SupportedEncoding
from max.tensor import Tensor

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLFlux2Config
from .vae import Decoder, Encoder


class AutoencoderKLFlux2(Module[[Tensor, Tensor | None], Tensor]):
    r"""A VAE model with KL loss for Flux2, encoding images into latents and decoding latent representations into images.

    This is similar to AutoencoderKL but uses Flux2-specific configuration
    with 32 latent channels (vs 4 for Flux1) and supports BatchNorm statistics
    for latent patchification.
    """

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
        # Encoder: images -> latents (mean and logvar)
        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=tuple(config.down_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            double_z=True,  # Output 2*latent_channels for mean and logvar
            mid_block_add_attention=config.mid_block_add_attention,
            device=config.device,
            dtype=config.dtype,
        )

        # Quantization convolution: encoder output -> latent distribution parameters
        self.quant_conv: Conv2d | None = None
        if config.use_quant_conv:
            self.quant_conv = Conv2d(
                kernel_size=1,
                in_channels=2 * config.latent_channels,  # mean + logvar
                out_channels=2 * config.latent_channels,
                dtype=config.dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=True,
                device=config.device,
                permute=True,
            )

        # Decoder: latents -> images
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


class BatchNormStats:
    """Container for BatchNorm statistics.

    This class provides a simple interface to access BatchNorm running statistics
    (mean and variance) for Flux2's latent patchification process.
    """

    def __init__(self, running_mean: Tensor, running_var: Tensor) -> None:
        """Initialize BatchNormStats.

        Args:
            running_mean: Running mean tensor.
            running_var: Running variance tensor.
        """
        self.running_mean = running_mean
        self.running_var = running_var


class AutoencoderKLFlux2Model(BaseAutoencoderModel):
    """ComponentModel wrapper for AutoencoderKLFlux2.

    This class provides the ComponentModel interface for AutoencoderKLFlux2, handling
    configuration, weight loading, model compilation, and BatchNorm statistics
    for Flux2's latent patchification.
    """

    config_name: ClassVar[str] = AutoencoderKLFlux2Config.config_name

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
        # Initialize BatchNorm statistics attributes BEFORE super().__init__()
        # because super().__init__() calls load_model() which may set these values
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

    def convert_weights_to_target_dtype(
        self, target_dtype: DType | None = None
    ) -> dict[str, Any]:
        """Convert all weights to target dtype.

        This utility method ensures all weights are converted to the target dtype
        (from config.dtype) if needed. This provides automatic dtype conversion
        for Flux2 VAE weights, which are typically float32 but need to be bfloat16
        for efficiency.

        Args:
            target_dtype: Target dtype. If None, uses self.config.dtype.

        Returns:
            Dictionary of converted weights in DLPackArray format, ready for
            use in compile() calls. Keys are weight names, values are DLPackArray.
        """
        if target_dtype is None:
            target_dtype = self.config.dtype

        converted_weights = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            # Automatically convert dtype if it doesn't match target
            if weight_data.dtype != target_dtype:
                weight_data = weight_data.astype(target_dtype)
            # Extract DLPackArray for compile() (compile() expects DLPackArray)
            converted_weights[key] = weight_data.data

        return converted_weights

    def load_model(self) -> Any:
        """Load and compile the decoder and encoder models with BatchNorm statistics.

        Converts all weights to target dtype (bfloat16 for Flux2), then loads
        decoder/encoder/post_quant_conv/quant_conv weights, and finally processes
        BatchNorm statistics (bn.*) which are specific to Flux2.

        Returns:
            Compiled decoder model callable.
        """
        # Convert all weights to target dtype (bfloat16 for Flux2)
        target_dtype = self.config.dtype
        converted_weights = self.convert_weights_to_target_dtype(target_dtype)

        # Extract BatchNorm statistics
        bn_mean_data = None
        bn_var_data = None

        # Prepare decoder weights
        decoder_state_dict = {}
        # Prepare encoder weights
        encoder_state_dict = {}
        quant_conv_state_dict = {}

        for key, dlpack_data in converted_weights.items():
            if key in ("bn.running_mean", "latent_bn.running_mean"):
                bn_mean_data = dlpack_data
            elif key in ("bn.running_var", "latent_bn.running_var"):
                bn_var_data = dlpack_data
            elif key.startswith("decoder."):
                # Remove "decoder." prefix for decoder weights
                decoder_state_dict[key.removeprefix("decoder.")] = dlpack_data
            elif key.startswith("post_quant_conv."):
                # Keep post_quant_conv prefix as-is (used by decoder)
                decoder_state_dict[key] = dlpack_data
            elif key.startswith("encoder."):
                # Remove "encoder." prefix for encoder weights
                encoder_state_dict[key.removeprefix("encoder.")] = dlpack_data
            elif key.startswith("quant_conv."):
                # Keep quant_conv prefix as-is (used by encoder)
                quant_conv_state_dict[key] = dlpack_data

        with F.lazy():
            autoencoder = self.autoencoder_class(self.config)

            # Compile decoder (always needed)
            autoencoder.decoder.to(self.devices[0])
            self.model = autoencoder.decoder.compile(
                *autoencoder.decoder.input_types(), weights=decoder_state_dict
            )

            # Compile encoder (optional, only if weights exist)
            if encoder_state_dict and hasattr(autoencoder, "encoder"):
                autoencoder.encoder.to(self.devices[0])
                self.encoder_model = autoencoder.encoder.compile(
                    *autoencoder.encoder.input_types(),
                    weights=encoder_state_dict,
                )

            # Compile quant_conv (optional, only if weights exist and encoder exists)
            if (
                quant_conv_state_dict
                and hasattr(autoencoder, "quant_conv")
                and autoencoder.quant_conv is not None
            ):
                # quant_conv is a Conv2d layer, compile it separately
                # Create a simple wrapper module for quant_conv
                class QuantConvModule(Module[[Tensor], Tensor]):
                    def __init__(self, quant_conv: Conv2d):
                        super().__init__()
                        self.quant_conv = quant_conv

                    def forward(self, x: Tensor) -> Tensor:
                        return self.quant_conv(x)

                quant_conv_module = QuantConvModule(autoencoder.quant_conv)
                quant_conv_module.to(self.devices[0])
                # quant_conv input: [B, 2*latent_channels, H_latent, W_latent]
                if self.encoder_model is not None:
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

        # Convert BatchNorm statistics to Tensor in lazy context
        if bn_mean_data is not None or bn_var_data is not None:
            with F.lazy():
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
    def bn(self) -> BatchNormStats:
        """Property to access BatchNorm statistics, compatible with diffusers API.

        This returns a simple object with running_mean and running_var attributes
        for compatibility with pipeline code that accesses self.vae.bn.running_mean.
        The statistics are returned as MAX Tensors.

        Returns:
            BatchNormStats: Object containing running_mean and running_var.

        Raises:
            ValueError: If BatchNorm statistics are not loaded.
        """
        if self.bn_running_mean is None or self.bn_running_var is None:
            raise ValueError(
                "BatchNorm statistics (running_mean, running_var) not loaded. "
                "Make sure the model weights contain 'bn.running_mean' and 'bn.running_var'."
            )

        return BatchNormStats(self.bn_running_mean, self.bn_running_var)
