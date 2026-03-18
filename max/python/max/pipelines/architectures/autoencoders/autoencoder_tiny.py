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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d, Module, Sequential
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import AutoencoderTinyConfig


def _apply_activation(x: Tensor, act_fn: str) -> Tensor:
    if act_fn in ("relu",):
        return F.relu(x)
    if act_fn in ("silu", "swish"):
        return F.silu(x)
    if act_fn == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported activation function: {act_fn}")


class AutoencoderTinyBlock(Module[[Tensor], Tensor]):
    """Residual block used by AutoencoderTiny."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: str,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.act_fn = act_fn
        self.conv = Sequential(
            Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
            _ActivationModule(act_fn),
            Conv2d(
                kernel_size=3,
                in_channels=out_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
            _ActivationModule(act_fn),
            Conv2d(
                kernel_size=3,
                in_channels=out_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
        )
        self.skip: Conv2d | None = None
        if in_channels != out_channels:
            self.skip = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                has_bias=False,
                device=device,
                permute=True,
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x) if self.skip is not None else x
        h = self.conv(x)
        return F.relu(h + residual)


class _ActivationModule(Module[[Tensor], Tensor]):
    def __init__(self, act_fn: str) -> None:
        super().__init__()
        self.act_fn = act_fn

    def forward(self, x: Tensor) -> Tensor:
        return _apply_activation(x, self.act_fn)


class EncoderTiny(Module[[Tensor], Tensor]):
    """A MAX-native port of diffusers EncoderTiny."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: tuple[int, ...],
        block_out_channels: tuple[int, ...],
        act_fn: str,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        if len(num_blocks) != len(block_out_channels):
            raise ValueError(
                "`num_blocks` and `block_out_channels` must have the same length."
            )

        self.in_channels = in_channels
        self.dtype = dtype
        self.device = device
        layers: list[Module[[Tensor], Tensor]] = []

        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]
            if i == 0:
                layers.append(
                    Conv2d(
                        kernel_size=3,
                        in_channels=in_channels,
                        out_channels=num_channels,
                        dtype=dtype,
                        stride=1,
                        padding=1,
                        has_bias=True,
                        device=device,
                        permute=True,
                    )
                )
            else:
                layers.append(
                    Conv2d(
                        kernel_size=3,
                        in_channels=num_channels,
                        out_channels=num_channels,
                        dtype=dtype,
                        stride=2,
                        padding=1,
                        has_bias=False,
                        device=device,
                        permute=True,
                    )
                )

            for _ in range(num_block):
                layers.append(
                    AutoencoderTinyBlock(
                        num_channels,
                        num_channels,
                        act_fn,
                        device=device,
                        dtype=dtype,
                    )
                )

        layers.append(
            Conv2d(
                kernel_size=3,
                in_channels=block_out_channels[-1],
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            )
        )
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers((x + 1) / 2)

    def input_types(self) -> tuple[TensorType, ...]:
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        return (
            TensorType(
                self.dtype,
                shape=["batch_size", self.in_channels, "image_height", "image_width"],
                device=self.device,
            ),
        )


class DecoderTiny(Module[[Tensor], Tensor]):
    """A MAX-native port of diffusers DecoderTiny."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: tuple[int, ...],
        block_out_channels: tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        if upsample_fn != "nearest":
            raise NotImplementedError(
                "AutoencoderTiny currently supports only upsample_fn='nearest'."
            )
        if len(num_blocks) != len(block_out_channels):
            raise ValueError(
                "`num_blocks` and `block_out_channels` must have the same length."
            )

        self.in_channels = in_channels
        self.dtype = dtype
        self.device = device
        self.act_fn = act_fn
        self.upsampling_scaling_factor = upsampling_scaling_factor
        layers: list[Module[[Tensor], Tensor]] = [
            Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=block_out_channels[0],
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
            _ActivationModule(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]
            is_final_block = i == len(num_blocks) - 1
            for _ in range(num_block):
                layers.append(
                    AutoencoderTinyBlock(
                        num_channels,
                        num_channels,
                        act_fn,
                        device=device,
                        dtype=dtype,
                    )
                )
            if not is_final_block:
                layers.append(
                    _NearestUpsample2D(scale_factor=upsampling_scaling_factor)
                )
            conv_out_channels = num_channels if not is_final_block else out_channels
            layers.append(
                Conv2d(
                    kernel_size=3,
                    in_channels=num_channels,
                    out_channels=conv_out_channels,
                    dtype=dtype,
                    stride=1,
                    padding=1,
                    has_bias=is_final_block,
                    device=device,
                    permute=True,
                )
            )
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = F.tanh(x / 3) * 3
        return (self.layers(x) * 2) - 1

    def input_types(self) -> tuple[TensorType, ...]:
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")
        return (
            TensorType(
                self.dtype,
                shape=["batch_size", self.in_channels, "latent_height", "latent_width"],
                device=self.device,
            ),
        )


class _NearestUpsample2D(Module[[Tensor], Tensor]):
    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        if scale_factor != 2:
            raise NotImplementedError(
                "AutoencoderTiny currently supports only scale_factor=2."
            )
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = F.reshape(x, [n, c, h, 1, w, 1])
        ones = F.broadcast_to(
            F.constant(1.0, dtype=x.dtype, device=x.device),
            [1, 1, 1, self.scale_factor, 1, self.scale_factor],
        )
        x = F.mul(x, ones)
        return F.reshape(x, [n, c, h * self.scale_factor, w * self.scale_factor])


class AutoencoderTiny(Module[[Tensor], Tensor]):
    """A tiny deterministic autoencoder compatible with diffusers TAESD."""

    def __init__(self, config: AutoencoderTinyConfig) -> None:
        super().__init__()
        self.encoder = EncoderTiny(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            num_blocks=tuple(config.num_encoder_blocks),
            block_out_channels=tuple(config.encoder_block_out_channels),
            act_fn=config.act_fn,
            device=config.device,
            dtype=config.dtype,
        )
        self.decoder = DecoderTiny(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            num_blocks=tuple(config.num_decoder_blocks),
            block_out_channels=tuple(config.decoder_block_out_channels),
            upsampling_scaling_factor=config.upsampling_scaling_factor,
            act_fn=config.act_fn,
            upsample_fn=config.upsample_fn,
            device=config.device,
            dtype=config.dtype,
        )
        self.latent_magnitude = config.latent_magnitude
        self.latent_shift = config.latent_shift
        self.scaling_factor = config.scaling_factor

    def scale_latents(self, x: Tensor) -> Tensor:
        return F.min(
            F.max((x / (2 * self.latent_magnitude)) + self.latent_shift, 0.0),
            1.0,
        )

    def unscale_latents(self, x: Tensor) -> Tensor:
        return (x - self.latent_shift) * (2 * self.latent_magnitude)

    def encode(self, x: Tensor, return_dict: bool = True) -> dict[str, Tensor] | Tensor:
        latents = self.encoder(x)
        if return_dict:
            return {"latents": latents}
        return latents

    def decode(self, x: Tensor, return_dict: bool = True) -> dict[str, Tensor] | Tensor:
        sample = self.decoder(x)
        if return_dict:
            return {"sample": sample}
        return sample

    def forward(self, sample: Tensor) -> Tensor:
        return self.decoder(self.encoder(sample))


class AutoencoderTinyModel(ComponentModel):
    """ComponentModel wrapper for deterministic AutoencoderTiny."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = AutoencoderTinyConfig.generate(config, encoding, devices)
        self.encoder_model = None
        self.load_model()

    def load_model(self) -> Any:
        encoder_state_dict = {}
        decoder_state_dict = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            adapted_key = key
            while adapted_key.startswith(("vae.", "model.")):
                if adapted_key.startswith("vae."):
                    adapted_key = adapted_key.removeprefix("vae.")
                    continue
                adapted_key = adapted_key.removeprefix("model.")

            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)

            if adapted_key.startswith("encoder."):
                encoder_state_dict[adapted_key.removeprefix("encoder.")] = (
                    weight_data
                )
            elif adapted_key.startswith("decoder."):
                decoder_state_dict[adapted_key.removeprefix("decoder.")] = (
                    weight_data
                )

        with F.lazy():
            autoencoder = AutoencoderTiny(self.config)
            autoencoder.decoder.to(self.devices[0])
            self.model = autoencoder.decoder.compile(
                *autoencoder.decoder.input_types(),
                weights=decoder_state_dict,
            )
            if encoder_state_dict:
                autoencoder.encoder.to(self.devices[0])
                self.encoder_model = autoencoder.encoder.compile(
                    *autoencoder.encoder.input_types(),
                    weights=encoder_state_dict,
                )
        return self.model

    def encode(self, sample: Tensor, return_dict: bool = True) -> dict[str, Tensor] | Tensor:
        if self.encoder_model is None:
            raise ValueError(
                "Encoder not loaded. Check if encoder weights exist in the model."
            )
        latents = self.encoder_model(sample)
        if return_dict:
            return {"latents": latents}
        return latents

    def decode(self, z: Tensor) -> Tensor:
        return self.model(z)
