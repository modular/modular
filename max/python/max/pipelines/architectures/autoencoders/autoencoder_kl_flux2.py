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

from max.driver import Buffer, Device
from max.graph import Graph, TensorType, TensorValue, ops
from max.graph.weights import Weights
from max.nn.layer import Module
from max.pipelines.lib import SupportedEncoding
from max.profiler import traced

from .decode_step_flux2 import Flux2DecodeStep
from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLFlux2Config
from .vae import Decoder, DiagonalGaussianDistribution, Encoder


class AutoencoderKLFlux2(Module):
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
            internal_io_layout="nhwc",
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(
        self, z: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
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

    bn_running_mean: Buffer
    bn_running_var: Buffer

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        """Initialize AutoencoderKLFlux2Model.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
            **kwargs: Additional keyword arguments forwarded to ComponentModel.
        """
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLFlux2Config,
            autoencoder_class=AutoencoderKLFlux2,
            **kwargs,
        )

    @staticmethod
    def _materialize(weight_data: Any) -> Buffer:
        data = getattr(weight_data, "data", weight_data)
        if isinstance(data, Buffer):
            return data
        return Buffer.from_dlpack(data)

    @staticmethod
    def _extract_mean(moments: TensorValue) -> TensorValue:
        return ops.chunk(moments, chunks=2, axis=1)[0]

    @traced(message="AutoencoderKLFlux2Model.load_model")
    def load_model(self) -> Callable[..., Any]:
        """Load encoder and BatchNorm statistics (skip standalone decoder).

        The standalone decoder compiled by the base class is not used in the
        Flux2 pipeline, which decodes through build_fused_decode().

        Returns:
            Compiled encoder model callable.
        """
        bn_stats: dict[str, Buffer] = {}
        encoder_state_dict: dict[str, Any] = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)

            if key in ("bn.running_mean", "bn.running_var"):
                bn_stats[key] = self._materialize(weight_data).to(
                    self.devices[0]
                )
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = weight_data
            elif key.startswith("quant_conv."):
                encoder_state_dict[key] = weight_data

        bn_mean_data = bn_stats.get("bn.running_mean")
        bn_var_data = bn_stats.get("bn.running_var")
        if bn_mean_data is None or bn_var_data is None:
            raise ValueError(
                "BatchNorm statistics (running_mean, running_var) not loaded. "
                "Make sure the model weights contain 'bn.running_mean' and 'bn.running_var'."
            )

        self.bn_running_mean = bn_mean_data
        self.bn_running_var = bn_var_data

        autoencoder = AutoencoderKLFlux2(self.config)
        self.encoder_model = self._compile_module(
            autoencoder.encoder,
            autoencoder.encoder.input_types(),
            encoder_state_dict,
            "autoencoder_kl_flux2_encoder_v2",
        )

        with Graph(
            "autoencoder_kl_flux2_extract_mean",
            input_types=(
                TensorType(
                    self.config.dtype,
                    shape=[
                        "batch",
                        self.config.latent_channels * 2,
                        "latent_height",
                        "latent_width",
                    ],
                    device=self.config.device,
                ),
            ),
        ) as graph:
            graph.output(self._extract_mean(graph.inputs[0].tensor))
        self._extract_mean_model = self.session.load(graph).execute

        return self.encoder_model

    @traced(message="AutoencoderKLFlux2Model.build_fused_decode")
    def build_fused_decode(
        self, device: Device, num_channels: int
    ) -> Callable[..., Any]:
        """Build a fused postprocess + VAE decode compiled graph.

        Combines BN denormalization, unpatchify, and VAE decoding into a single
        compiled graph.

        Args:
            device: Target device for the compiled graph.
            num_channels: Number of latent channels after patchification.

        Returns:
            Compiled callable taking packed latents and shape carriers.
        """
        dtype = self.config.dtype
        fused_state_dict: dict[str, Any] = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != dtype:
                if weight_data.dtype.is_float() and dtype.is_float():
                    weight_data = weight_data.astype(dtype)
            if key.startswith("decoder."):
                fused_state_dict[f"decoder.{key.removeprefix('decoder.')}"] = (
                    weight_data
                )
            elif key.startswith("post_quant_conv."):
                fused_state_dict[f"decoder.{key}"] = weight_data

        # The decode step receives BN tensors as runtime inputs, so do not load
        # them as module weights.
        autoencoder = AutoencoderKLFlux2(self.config)
        decode_step = Flux2DecodeStep(
            decoder=autoencoder.decoder,
            batch_norm_eps=self.config.batch_norm_eps,
        )
        compiled = self._compile_module(
            decode_step,
            decode_step.input_types(),
            fused_state_dict,
            "autoencoder_kl_flux2_decode_step_v2",
        )

        return lambda latents_bsc, h_carrier, w_carrier: self._unwrap_single(
            compiled(
                latents_bsc,
                h_carrier,
                w_carrier,
                self.bn_running_mean,
                self.bn_running_var,
            )
        )

    def encode(
        self, sample: Buffer, return_dict: bool = True
    ) -> dict[str, DiagonalGaussianDistribution] | DiagonalGaussianDistribution:
        if self.encoder_model is None:
            raise ValueError(
                "Encoder not loaded. Check if encoder weights exist in the model."
            )
        moments = self._unwrap_single(self.encoder_model(sample))
        mean = self._unwrap_single(self._extract_mean_model(moments))
        posterior = DiagonalGaussianDistribution(mean, moments)
        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    @property
    def bn(self) -> SimpleNamespace:
        """Access BatchNorm statistics in a diffusers-compatible shape."""
        return SimpleNamespace(
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
        )
