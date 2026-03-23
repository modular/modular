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
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Conv2d,
    ConvTranspose2d,
    GroupNorm,
    Module,
    Sequential,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .autoencoder_tiny import DecoderTiny, EncoderTiny, _ActivationModule
from .model_config import Flux2TinyAutoEncoderConfig


class _TinyVAEEncoder(Module[[Tensor], Tensor]):
    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.encoder = EncoderTiny(
            in_channels=config.in_channels,
            out_channels=config.latent_channels // 4,
            num_blocks=tuple(config.num_encoder_blocks),
            block_out_channels=tuple(config.encoder_block_out_channels),
            act_fn=config.act_fn,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _TinyVAEDecoder(Module[[Tensor], Tensor]):
    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.decoder = DecoderTiny(
            in_channels=config.latent_channels // 4,
            out_channels=config.out_channels,
            num_blocks=tuple(config.num_decoder_blocks),
            block_out_channels=tuple(config.decoder_block_out_channels),
            upsampling_scaling_factor=config.upsampling_scaling_factor,
            act_fn=config.act_fn,
            upsample_fn=config.upsample_fn,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class Flux2TinyEncoder(Module[[Tensor], Tensor]):
    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.dtype = config.dtype
        self.device = config.device
        self.tiny_vae = _TinyVAEEncoder(config)
        self.extra_encoder = Conv2d(
            kernel_size=4,
            in_channels=config.latent_channels // 4,
            out_channels=config.latent_channels,
            dtype=config.dtype,
            stride=2,
            padding=1,
            has_bias=True,
            device=config.device,
            permute=True,
        )
        self.residual_encoder = Sequential(
            Conv2d(
                kernel_size=3,
                in_channels=config.latent_channels,
                out_channels=config.latent_channels,
                dtype=config.dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=config.device,
                permute=True,
            ),
            GroupNorm(
                num_groups=8,
                num_channels=config.latent_channels,
                eps=1e-5,
                affine=True,
            ),
            _ActivationModule("silu"),
            Conv2d(
                kernel_size=3,
                in_channels=config.latent_channels,
                out_channels=config.latent_channels,
                dtype=config.dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=config.device,
                permute=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.tiny_vae(x)
        compressed = self.extra_encoder(encoded)
        return self.residual_encoder(compressed) + compressed

    def input_types(self) -> tuple[TensorType, ...]:
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "image_height",
                    "image_width",
                ],
                device=self.device,
            ),
        )


class Flux2TinyConcreteFusedDecode(Module[..., Tensor]):
    """Concrete-shape fused packed decode using diffusers-style module names."""

    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.latent_channels // 4
        self.dtype = config.dtype
        self.device = config.device
        self.tiny_vae = _TinyVAEDecoder(config)
        self.extra_decoder = ConvTranspose2d(
            kernel_size=4,
            in_channels=config.latent_channels,
            out_channels=config.latent_channels // 4,
            dtype=DType.float32,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=0,
            device=config.device,
            has_bias=True,
            permute=True,
        )
        self.residual_decoder = Sequential(
            Conv2d(
                kernel_size=3,
                in_channels=config.latent_channels // 4,
                out_channels=config.latent_channels // 4,
                dtype=config.dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=config.device,
                permute=True,
            ),
            GroupNorm(
                num_groups=8,
                num_channels=config.latent_channels // 4,
                eps=1e-5,
                affine=True,
            ),
            _ActivationModule("silu"),
            Conv2d(
                kernel_size=3,
                in_channels=config.latent_channels // 4,
                out_channels=config.latent_channels // 4,
                dtype=config.dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=config.device,
                permute=True,
            ),
        )

    def forward(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        batch = latents_bsc.shape[0]
        channels = latents_bsc.shape[2]
        latent_height = h_carrier.shape[0]
        latent_width = w_carrier.shape[0]
        latents_bsc = F.rebind(
            latents_bsc,
            [batch, latent_height * latent_width, channels],
        )
        latents = F.reshape(
            latents_bsc,
            (batch, latent_height, latent_width, channels),
        )
        latents = F.permute(latents, (0, 3, 1, 2))
        latents = latents.to(self.config.device).cast(self.config.dtype)
        scaled = latents / (2 * self.config.latent_magnitude)
        scaled = F.min(F.max(scaled + self.config.latent_shift, 0.0), 1.0)
        latents = (
            (F.round(scaled * 255.0) / 255.0) - self.config.latent_shift
        ) * (2 * self.config.latent_magnitude)
        decompressed = self.extra_decoder(latents.cast(DType.float32))
        enhanced = self.residual_decoder(decompressed.cast(self.dtype))
        decoded = self.tiny_vae(enhanced + decompressed.cast(self.dtype))
        decoded = F.permute(decoded, (0, 2, 3, 1))
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0
        return F.transfer_to(F.cast(decoded, DType.uint8), DeviceRef.CPU())


class Flux2TinyAutoEncoderModel(ComponentModel):
    """ComponentModel wrapper for the FLUX.2 tiny autoencoder."""

    vae_mode = "tiny"

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = Flux2TinyAutoEncoderConfig.generate(
            config, encoding, devices
        )
        self.encoder_model = None
        self._state_dict: dict[str, Any] = {}
        self.load_model()

    def load_model(self) -> Any:
        target_dtype = self.config.dtype
        state_dict = {}

        for key, value in self.weights.items():
            normalized_key = key
            while normalized_key.startswith(("vae.", "model.")):
                if normalized_key.startswith("vae."):
                    normalized_key = normalized_key.removeprefix("vae.")
                    continue
                normalized_key = normalized_key.removeprefix("model.")
            weight_data = value.data()
            # Match the fp32 transposed-convolution path above when loading the
            # extra_decoder weights; the rest of the module stays in model dtype.
            desired_dtype = (
                DType.float32
                if normalized_key.startswith("extra_decoder.")
                else target_dtype
            )
            if (
                weight_data.dtype != desired_dtype
                and weight_data.dtype.is_float()
                and desired_dtype.is_float()
            ):
                weight_data = weight_data.astype(desired_dtype)
            state_dict[normalized_key] = weight_data

        self._state_dict = state_dict

        with F.lazy():
            encoder = Flux2TinyEncoder(self.config)
            encoder.to(self.devices[0])
            self.encoder_model = encoder.compile(
                *encoder.input_types(),
                weights=self._state_dict,
            )

        return None

    def _compile_packed_fused_model(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        fused_decode = Flux2TinyConcreteFusedDecode(config=self.config)
        fused_decode.to(self.devices[0])
        return fused_decode.compile(
            TensorType(
                self.config.dtype,
                [
                    batch_size,
                    latent_height * latent_width,
                    self.config.latent_channels,
                ],
                device=self.devices[0],
            ),
            TensorType(
                DType.float32,
                [latent_height],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                DType.float32,
                [latent_width],
                device=DeviceRef.CPU(),
            ),
            weights=self._state_dict,
        )

    def build_fused_decode(self, device: Device) -> Callable[..., Any]:
        if device != self.devices[0]:
            raise ValueError(
                "Flux2TinyAutoEncoderModel.build_fused_decode only supports the primary VAE device."
            )
        compiled_by_shape: dict[tuple[int, int, int], Any] = {}

        def fused_decode(
            latents_bsc: Tensor, h_carrier: Tensor, w_carrier: Tensor
        ) -> Tensor:
            shape_key = (
                int(latents_bsc.shape[0]),
                int(h_carrier.shape[0]),
                int(w_carrier.shape[0]),
            )
            model = compiled_by_shape.get(shape_key)
            if model is None:
                model = self._compile_packed_fused_model(
                    batch_size=shape_key[0],
                    latent_height=shape_key[1],
                    latent_width=shape_key[2],
                )
                compiled_by_shape[shape_key] = model
            return model(latents_bsc, h_carrier, w_carrier)

        return fused_decode

    def encode(self, sample: Tensor) -> Tensor:
        if self.encoder_model is None:
            raise ValueError(
                "Encoder not loaded. Check if encoder weights exist in the model."
            )
        return self.encoder_model(sample)
