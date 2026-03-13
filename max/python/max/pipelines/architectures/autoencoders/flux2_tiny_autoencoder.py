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

import numpy as np

from max.driver import Accelerator, Device, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn import Conv2d, GroupNorm, Module, Sequential
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType
from max.graph.type import ConvInputLayout, FilterLayout
from max.graph.weights import Weights
from max.nn import ConvTranspose2d as GraphConvTranspose2d
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer

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

        if self.stride != (2, 2):
            raise NotImplementedError(
                "Flux2 tiny VAE ConvTranspose2d currently supports only stride=2."
            )
        if self.dilation != (1, 1):
            raise NotImplementedError(
                "Flux2 tiny VAE ConvTranspose2d currently supports only dilation=1."
            )

    def _weight_cell(
        self,
        weight: Tensor,
        kernel_h: int | None,
        kernel_w: int | None,
    ) -> Tensor:
        """Extract one 1x1 weight cell in [1, 1, in_channels, out_channels] form."""
        if kernel_h is None or kernel_w is None:
            zeros = F.constant(0.0, dtype=weight.dtype, device=weight.device)
            return F.broadcast_to(
                zeros,
                [1, 1, self.in_channels, self.out_channels],
            )
        cell = F.slice_tensor(
            weight,
            [
                slice(None),
                slice(None),
                slice(kernel_h, kernel_h + 1),
                slice(kernel_w, kernel_w + 1),
            ],
        )
        return F.permute(cell, [2, 3, 0, 1])

    def _phase_kernel(
        self,
        weight: Tensor,
        row_indices: tuple[int | None, int | None, int | None],
        col_indices: tuple[int | None, int | None, int | None],
    ) -> Tensor:
        rows = []
        for kh in row_indices:
            cols = [
                self._weight_cell(weight, kh, kw) for kw in col_indices
            ]
            rows.append(F.concat(cols, axis=1))
        return F.concat(rows, axis=0)

    def _to_subpixel_weight(self, weight: Tensor) -> Tensor:
        """Convert ConvTranspose2d weights into subpixel Conv2d weights."""
        phase00 = self._phase_kernel(
            weight,
            row_indices=(3, 1, None),
            col_indices=(3, 1, None),
        )
        phase01 = self._phase_kernel(
            weight,
            row_indices=(3, 1, None),
            col_indices=(None, 2, 0),
        )
        phase10 = self._phase_kernel(
            weight,
            row_indices=(None, 2, 0),
            col_indices=(3, 1, None),
        )
        phase11 = self._phase_kernel(
            weight,
            row_indices=(None, 2, 0),
            col_indices=(None, 2, 0),
        )

        kernels = F.concat(
            [
                F.reshape(phase00, [3, 3, self.in_channels, self.out_channels, 1]),
                F.reshape(phase10, [3, 3, self.in_channels, self.out_channels, 1]),
                F.reshape(phase01, [3, 3, self.in_channels, self.out_channels, 1]),
                F.reshape(phase11, [3, 3, self.in_channels, self.out_channels, 1]),
            ],
            axis=4,
        )
        return F.reshape(kernels, [3, 3, self.in_channels, self.out_channels * 4])

    def _repeat_bias_for_subpixel(self, bias: Tensor) -> Tensor:
        bias = F.reshape(bias, [self.out_channels, 1])
        ones = F.broadcast_to(
            F.constant(1.0, dtype=bias.dtype, device=bias.device),
            [self.out_channels, 4],
        )
        return F.reshape(bias * ones, [self.out_channels * 4])

    def _pixel_shuffle_2x(self, x: Tensor) -> Tensor:
        batch, channels4, height, width = x.shape
        x = F.reshape(
            x,
            [batch, self.out_channels, 2, 2, height, width],
        )
        x = F.permute(x, [0, 1, 4, 2, 5, 3])
        return F.reshape(
            x,
            [batch, self.out_channels, height * 2, width * 2],
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.kernel_size != (4, 4):
            raise NotImplementedError(
                "Flux2 tiny VAE ConvTranspose2d currently supports only kernel_size=4."
            )
        if self.padding != (1, 1, 1, 1):
            raise NotImplementedError(
                "Flux2 tiny VAE ConvTranspose2d currently supports only padding=1."
            )
        if not self.permute:
            raise NotImplementedError(
                "Flux2 tiny VAE ConvTranspose2d currently expects permute=True."
            )

        weight = self._to_subpixel_weight(self.weight.to(x.device))
        bias = (
            self._repeat_bias_for_subpixel(self.bias.to(x.device))
            if isinstance(self.bias, Tensor)
            else None
        )

        x = F.permute(x, [0, 2, 3, 1])

        output = F.conv2d(
            x,
            weight,
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            filter_layout=FilterLayout.RSCF,
            bias=bias,
        )
        output = F.permute(output, [0, 3, 1, 2])
        return self._pixel_shuffle_2x(output)


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


class Flux2TinyDecoderTail(Module[[Tensor], Tensor]):
    """Stable decoder tail after the problematic transposed-convolution stage."""

    def __init__(self, config: Flux2TinyAutoEncoderConfig) -> None:
        super().__init__()
        self.in_channels = config.latent_channels // 4
        self.dtype = config.dtype
        self.device = config.device
        self.residual_decoder = _ResidualRefinement(
            channels=config.latent_channels // 4,
            num_groups=8,
            act_fn="silu",
            device=config.device,
            dtype=config.dtype,
        )
        self.tiny_vae = _TinyVAEDecoderContainer(config)

    def forward(self, x: Tensor) -> Tensor:
        enhanced = self.residual_decoder(x)
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
        self.extra_decoder_model = None
        self.decoder_tail_model = None
        self._compiled_decode_shape: tuple[int, int, int] | None = None
        self._extra_decoder_template = None
        self._decoder_tail_state_dict: dict[str, Any] = {}
        self.load_model()

    def load_model(self) -> Any:
        encoder_state_dict = {}
        decoder_tail_state_dict = {}
        extra_decoder_state_dict = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            normalized_key = key
            while normalized_key.startswith(("vae.", "model.")):
                if normalized_key.startswith("vae."):
                    normalized_key = normalized_key.removeprefix("vae.")
                    continue
                normalized_key = normalized_key.removeprefix("model.")

            adapted_key = normalized_key

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
                ("tiny_vae.decoder.", "residual_decoder.layers.")
            ):
                decoder_tail_state_dict[adapted_key] = weight_data
            if adapted_key.startswith("extra_decoder."):
                extra_decoder_state_dict[
                    adapted_key.removeprefix("extra_decoder.")
                ] = (
                    weight_data.astype(DType.float32)
                    if weight_data.dtype != DType.float32
                    and weight_data.dtype.is_float()
                    else weight_data
                )

        with F.lazy():
            autoencoder = Flux2TinyAutoEncoder(self.config)
            autoencoder.encoder.to(self.devices[0])
            self.encoder_model = autoencoder.encoder.compile(
                *autoencoder.encoder.input_types(),
                weights=encoder_state_dict,
            )

        self._extra_decoder_template = GraphConvTranspose2d(
            kernel_size=4,
            in_channels=self.config.latent_channels,
            out_channels=self.config.latent_channels // 4,
            dtype=DType.float32,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=0,
            device=DeviceRef.from_device(self.devices[0]),
            has_bias=True,
            permute=True,
        )
        self._extra_decoder_template.load_state_dict(extra_decoder_state_dict)
        self._decoder_tail_state_dict = decoder_tail_state_dict

        self.model = self.decoder_tail_model
        return self.model

    def _ensure_decode_models_compiled(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> None:
        shape_key = (batch_size, latent_height, latent_width)
        if self._compiled_decode_shape == shape_key:
            return
        if self._extra_decoder_template is None:
            raise ValueError("Decoder templates not initialized.")

        session = InferenceSession(devices=[self.devices[0]])
        device_ref = DeviceRef.from_device(self.devices[0])

        extra_decoder_graph = Graph(
            "flux2_tiny_extra_decoder",
            self._extra_decoder_template,
            input_types=(
                TensorType(
                    DType.float32,
                    [
                        batch_size,
                        self.config.latent_channels,
                        latent_height,
                        latent_width,
                    ],
                    device=device_ref,
                ),
            ),
        )
        self.extra_decoder_model = session.load(
            extra_decoder_graph,
            weights_registry=self._extra_decoder_template.state_dict(),
        )

        decoder_tail = Flux2TinyDecoderTail(self.config)
        decoder_tail.to(self.devices[0])
        self.decoder_tail_model = decoder_tail.compile(
            TensorType(
                self.config.dtype,
                [
                    batch_size,
                    self.config.latent_channels // 4,
                    latent_height * 2,
                    latent_width * 2,
                ],
                device=self.devices[0],
            ),
            weights=self._decoder_tail_state_dict,
        )
        self._compiled_decode_shape = shape_key

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
        batch_size = int(z.shape[0])
        latent_height = int(z.shape[2])
        latent_width = int(z.shape[3])
        self._ensure_decode_models_compiled(
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        if self.extra_decoder_model is None or self.decoder_tail_model is None:
            raise ValueError("Decoder models not loaded.")
        with Tracer("vae_quantize"):
            scaled = z / (2 * self.config.latent_magnitude)
            scaled = F.min(F.max(scaled + self.config.latent_shift, 0.0), 1.0)
            z = (
                (F.round(scaled * 255.0) / 255.0) - self.config.latent_shift
            ) * (2 * self.config.latent_magnitude)
            z = z.cast(DType.float32)
        with Tracer("vae_extra_decoder"):
            decompressed = self.extra_decoder_model.execute(z.driver_tensor)[0]
        with Tracer("vae_buffer_cast"):
            decompressed_tensor = Tensor(storage=decompressed).cast(
                self.config.dtype
            )
        with Tracer("vae_decoder_tail"):
            return self.decoder_tail_model(decompressed_tensor)

    def decode_to_numpy(self, z: Tensor) -> np.ndarray:
        decoded = self.decode(z)
        decoded = F.permute(decoded, (0, 2, 3, 1))
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0
        decoded = F.transfer_to(F.cast(decoded, DType.uint8), DeviceRef.CPU())
        return np.from_dlpack(decoded)


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
