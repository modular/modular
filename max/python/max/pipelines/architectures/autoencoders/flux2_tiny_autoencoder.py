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
from max.experimental.nn import Conv2d, GroupNorm, Module, Sequential
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.type import ConvInputLayout, FilterLayout
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .autoencoder_kl_flux2 import AutoencoderKLFlux2Model
from .autoencoder_tiny import DecoderTiny, EncoderTiny, _ActivationModule
from .model_config import Flux2TinyAutoEncoderConfig


class ConvTranspose2d(Module[[Tensor], Tensor]):
    """Minimal MAX transposed-convolution layer with PyTorch-compatible weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        dtype: DType | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        device: Device | DeviceRef | None = None,
        has_bias: bool = True,
        permute: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.has_bias = has_bias

        if isinstance(kernel_size, int):
            kernel_height = kernel_width = kernel_size
        else:
            kernel_height, kernel_width = kernel_size
        self.kernel_size = (kernel_height, kernel_width)

        if permute:
            self.weight = Tensor.zeros(
                [in_channels, out_channels, kernel_height, kernel_width],
                dtype=dtype,
                device=device,
            )
        else:
            self.weight = Tensor.zeros(
                [kernel_height, kernel_width, out_channels, in_channels],
                dtype=dtype,
                device=device,
            )
        self.bias = (
            Tensor.zeros([out_channels], dtype=dtype, device=device)
            if has_bias
            else 0
        )

        self.stride = (stride, stride) if isinstance(stride, int) else stride
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            pad_h, pad_w = padding
            padding = (pad_h, pad_h, pad_w, pad_w)
        self.padding = padding
        self.dilation = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.device)
        return F.conv2d_transpose(
            x,
            weight,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            bias=self.bias if isinstance(self.bias, Tensor) else None,
            input_layout=ConvInputLayout.NCHW
            if self.permute
            else ConvInputLayout.NHWC,
            filter_layout=FilterLayout.CFRS
            if self.permute
            else FilterLayout.RSCF,
        )


class _ResidualRefinement(Module[[Tensor], Tensor]):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        act_fn: str,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.layers = Sequential(
            Conv2d(
                kernel_size=3,
                in_channels=channels,
                out_channels=channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
            GroupNorm(
                num_groups=num_groups,
                num_channels=channels,
                eps=1e-5,
                affine=True,
            ),
            _ActivationModule(act_fn),
            Conv2d(
                kernel_size=3,
                in_channels=channels,
                out_channels=channels,
                dtype=dtype,
                stride=1,
                padding=1,
                has_bias=True,
                device=device,
                permute=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + x


class _TinyVAEEncoderContainer(Module[[Tensor], Tensor]):
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


class _TinyVAEDecoderContainer(Module[[Tensor], Tensor]):
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
        self.tiny_vae = _TinyVAEEncoderContainer(config)
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
        self.residual_encoder = _ResidualRefinement(
            channels=config.latent_channels,
            num_groups=8,
            act_fn="silu",
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.tiny_vae(x)
        compressed = self.extra_encoder(encoded)
        return self.residual_encoder(compressed)

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


class Flux2TinyDecoder(Module[[Tensor], Tensor]):
    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.in_channels = config.latent_channels
        self.dtype = config.dtype
        self.device = config.device
        self.tiny_vae = _TinyVAEDecoderContainer(config)
        self.extra_decoder = ConvTranspose2d(
            in_channels=config.latent_channels,
            out_channels=config.latent_channels // 4,
            kernel_size=4,
            dtype=config.dtype,
            stride=2,
            padding=1,
            has_bias=True,
            device=config.device,
            permute=True,
        )
        self.residual_decoder = _ResidualRefinement(
            channels=config.latent_channels // 4,
            num_groups=8,
            act_fn="silu",
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, z: Tensor) -> Tensor:
        decompressed = self.extra_decoder(z)
        enhanced = self.residual_decoder(decompressed)
        return self.tiny_vae(enhanced)

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


class Flux2TinyAutoEncoder(Module[[Tensor, Tensor | None], Tensor]):
    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.encoder = Flux2TinyEncoder(config)
        self.decoder = Flux2TinyDecoder(config)

    def forward(self, z: Tensor, temb: Tensor | None = None) -> Tensor:
        return self.decoder(z)


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

            if adapted_key.startswith("residual_encoder."):
                adapted_key = adapted_key.replace(
                    "residual_encoder.",
                    "residual_encoder.layers.",
                    1,
                )
            elif adapted_key.startswith("residual_decoder."):
                adapted_key = adapted_key.replace(
                    "residual_decoder.",
                    "residual_decoder.layers.",
                    1,
                )

            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)

            if adapted_key.startswith(
                ("tiny_vae.", "extra_encoder.", "residual_encoder.")
            ):
                encoder_state_dict[adapted_key] = weight_data
            if adapted_key.startswith(
                ("tiny_vae.", "extra_decoder.", "residual_decoder.")
            ):
                decoder_state_dict[adapted_key] = weight_data

        with F.lazy():
            autoencoder = Flux2TinyAutoEncoder(self.config)
            autoencoder.decoder.to(self.devices[0])
            self.model = autoencoder.decoder.compile(
                *autoencoder.decoder.input_types(),
                weights=decoder_state_dict,
            )
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


class Flux2AutoencoderModel(ComponentModel):
    """Dispatch between KL and tiny FLUX.2 VAE implementations."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        class_name = config.get("_class_name")
        requested_mode = config.get("vae_mode")
        if requested_mode == "tiny" or class_name == "Flux2TinyAutoEncoder":
            self.delegate = Flux2TinyAutoEncoderModel(
                config=config,
                encoding=encoding,
                devices=devices,
                weights=weights,
            )
        else:
            self.delegate = AutoencoderKLFlux2Model(
                config=config,
                encoding=encoding,
                devices=devices,
                weights=weights,
            )
        self.config = self.delegate.config
        self.devices = self.delegate.devices
        self.weights = self.delegate.weights
        self.vae_mode = getattr(self.delegate, "vae_mode", "kl")

    def load_model(self) -> Any:
        return self.delegate.load_model()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self.delegate.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.delegate.decode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "delegate":
            raise AttributeError(name)
        return getattr(self.delegate, name)
