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
from types import SimpleNamespace
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.nn.module_v3 import Module
from max.pipelines.lib import SupportedEncoding

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


class PostprocessAndDecode(Module[[Tensor, Tensor, Tensor], Tensor]):
    """Fused BN-denorm + unpatchify + VAE decode in a single compiled graph.

    Eliminates the inter-graph boundary and intermediate tensor materialization
    that previously existed between _postprocess_latents and vae.decode().
    """

    def __init__(
        self,
        decoder: Decoder,
        batch_norm_eps: float,
        num_channels: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.batch_norm_eps = batch_norm_eps
        self._num_channels = num_channels
        self._device = device
        self._dtype = dtype

    def forward(
        self,
        latents_bhwc: Tensor,
        bn_mean: Tensor,
        bn_var: Tensor,
    ) -> Tensor:
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]

        # (B, H, W, C) -> (B, C, H, W)
        latents = F.permute(latents_bhwc, (0, 3, 1, 2))

        # BN denormalization
        bn_mean_r = F.reshape(bn_mean, (1, c, 1, 1))
        bn_var_r = F.reshape(bn_var, (1, c, 1, 1))
        bn_std = F.sqrt(bn_var_r + self.batch_norm_eps)
        latents = latents * bn_std + bn_mean_r

        # Unpatchify: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents = F.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(latents, (batch, c // 4, h * 2, w * 2))

        decoded = self.decoder(latents, None)
        decoded = F.cast(decoded, DType.float32)
        decoded = F.permute(
            decoded, (0, 2, 3, 1)
        )  # (B, C, H, W) -> (B, H, W, C)
        return F.transfer_to(decoded, DeviceRef.CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", "height", "width", self._num_channels],
                device=self._device,
            ),
            TensorType(
                self._dtype,
                shape=[self._num_channels],
                device=self._device,
            ),
            TensorType(
                self._dtype,
                shape=[self._num_channels],
                device=self._device,
            ),
        )


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

    def load_model(self) -> Callable[..., Any]:
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

    def build_fused_decode(
        self, device: Device, num_channels: int
    ) -> Callable[..., Any]:
        """Build a fused postprocess + VAE decode compiled graph.

        Combines BN denormalization, unpatchify, and VAE decoding into a single
        compiled graph, eliminating the intermediate tensor and device sync
        between the two previously separate compiled graphs.

        Args:
            device: Target device for the compiled graph.
            num_channels: Number of latent channels (bn.running_mean shape[0]).

        Returns:
            Compiled callable taking (latents_bhwc, bn_mean, bn_var) and
            returning the decoded image tensor.
        """
        dtype = self.config.dtype
        device_ref = DeviceRef.from_device(device)

        fused_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != dtype:
                if weight_data.dtype.is_float() and dtype.is_float():
                    weight_data = weight_data.astype(dtype)
            if key.startswith("decoder."):
                # decoder.X -> decoder.X (PostprocessAndDecode.decoder.X)
                fused_weights[key] = weight_data
            elif key.startswith("post_quant_conv."):
                # post_quant_conv.X -> decoder.post_quant_conv.X
                fused_weights[f"decoder.{key}"] = weight_data

        with F.lazy():
            autoencoder = AutoencoderKLFlux2(self.config)
            fused = PostprocessAndDecode(
                decoder=autoencoder.decoder,
                batch_norm_eps=self.config.batch_norm_eps,
                num_channels=num_channels,
                device=device_ref,
                dtype=dtype,
            )
            fused.to(device)
            self._fused_model = fused.compile(
                *fused.input_types(), weights=fused_weights
            )

        return self._fused_model

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
