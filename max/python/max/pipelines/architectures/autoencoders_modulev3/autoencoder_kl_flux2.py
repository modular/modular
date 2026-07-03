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
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.profiler import traced

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


class PostprocessAndDecode(Module[..., Tensor]):
    """Fused BN-denorm + unpatchify + VAE decode in a single compiled graph.

    Eliminates the inter-graph boundary and intermediate tensor materialization
    that previously existed between _postprocess_latents and vae.decode().

    Accepts packed latents in (B, S, C) shape where S = latent_h * latent_w.
    The spatial dimensions are conveyed via two 1-D shape-carrier tensors whose
    *lengths* (not values) encode latent_h and latent_w as symbolic graph Dims,
    so a single compiled graph handles any spatial size without recompilation.
    """

    def __init__(
        self,
        decoder: Decoder,
        bn_mean: Tensor,
        bn_var: Tensor,
        batch_norm_eps: float,
        num_channels: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.batch_norm_eps = batch_norm_eps
        self._num_channels = num_channels
        self._device = device
        self._dtype = dtype

    def forward(
        self,
        latents_bsc: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        # Extract spatial dims from carrier shapes (symbolic Dims, not runtime values)
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        # Assert seq == latent_h * latent_w so the reshape verifier accepts it,
        # then reshape packed (B, S, C) -> spatial (B, H, W, C).
        latents_bsc = F.rebind(latents_bsc, [batch, h * w, c])
        latents_bhwc = F.reshape(latents_bsc, (batch, h, w, c))

        # (B, H, W, C) -> (B, C, H, W)
        latents = F.permute(latents_bhwc, (0, 3, 1, 2))

        # BN denormalization
        bn_mean_r = F.reshape(self.bn_mean, (1, c, 1, 1))
        bn_var_r = F.reshape(self.bn_var, (1, c, 1, 1))
        bn_std = F.sqrt(bn_var_r + self.batch_norm_eps)
        latents = latents * bn_std + bn_mean_r

        # Unpatchify: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents = F.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(latents, (batch, c // 4, h * 2, w * 2))

        decoded = self.decoder(latents, None)
        decoded = F.permute(
            decoded, (0, 2, 3, 1)
        )  # (B, C, H, W) -> (B, H, W, C)
        # Denormalize [-1, 1] -> [0, 1], scale to [0, 255], cast to uint8.
        # Keeping in the decoder's native dtype (bfloat16/float16): all values
        # in [0, 255] are exactly representable, and avoiding the float32 upcast
        # reduces GPU compute and memory bandwidth for the post-processing ops.
        # Round before the uint8 cast so the truncating cast doesn't bias every
        # pixel down by ~0.5; diffusers' image processor does
        # `(x * 255).round().astype(uint8)`.
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0
        decoded = F.round(decoded)
        return F.transfer_to(F.cast(decoded, DType.uint8), DeviceRef.CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", "seq", self._num_channels],
                device=self._device,
            ),
            # Shape carriers: lengths encode latent_h / latent_w as symbolic dims.
            # Content is never read; only the shapes matter.
            TensorType(
                DType.float32, shape=["latent_h"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=["latent_w"], device=DeviceRef.CPU()
            ),
        )


class AutoencoderKLFlux2Model(BaseAutoencoderModel):
    """ComponentModel wrapper for AutoencoderKLFlux2.

    This class provides the ComponentModel interface for AutoencoderKLFlux2,
    handling configuration, weight loading, model compilation, and BatchNorm
    statistics for Flux2's latent patchification.
    """

    bn_running_mean: Tensor
    bn_running_var: Tensor

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

    @traced(message="AutoencoderKLFlux2Model.load_model")
    def load_model(self) -> None:
        """Load encoder and BatchNorm statistics (skip standalone decoder).

        The standalone decoder compiled by the base class is never used in
        the Flux2 pipeline — decoding goes through the fused
        ``PostprocessAndDecode`` graph built by ``build_fused_decode()``.
        This override avoids that redundant compilation, cutting startup
        time and GPU memory usage.
        """
        bn_stats: dict[str, Any] = {}
        encoder_state_dict: dict[str, Any] = {}
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != target_dtype:
                if weight_data.dtype.is_float() and target_dtype.is_float():
                    weight_data = weight_data.astype(target_dtype)

            if key in ("bn.running_mean", "bn.running_var"):
                bn_stats[key] = weight_data.data
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

        self.bn_running_mean = Tensor.from_dlpack(bn_mean_data).to(
            self.devices[0]
        )
        self.bn_running_var = Tensor.from_dlpack(bn_var_data).to(
            self.devices[0]
        )

        # Defer encoder compilation until first image-to-image request.
        # Text-to-image never invokes the encoder, so compiling it here
        # would waste startup time and GPU memory. The decoder is compiled
        # later via build_fused_decode() which wraps it in PostprocessAndDecode.
        self._encoder_state_dict = encoder_state_dict

    @traced(message="AutoencoderKLFlux2Model.ensure_encoder_compiled")
    def ensure_encoder_compiled(self) -> Callable[..., Any]:
        """Compile the VAE encoder on demand (idempotent).

        Called from the image-to-image path on first use. No-op if the
        encoder has already been compiled.

        Returns:
            Compiled encoder model callable.

        Raises:
            RuntimeError: If no encoder weights were found in the checkpoint.
        """
        if self.encoder_model is not None:
            return self.encoder_model

        if not self._encoder_state_dict:
            raise RuntimeError(
                "Cannot compile encoder — no encoder weights were found "
                "in the checkpoint."
            )

        with F.lazy():
            autoencoder = AutoencoderKLFlux2(self.config)
            autoencoder.encoder.to(self.devices[0])
            self.encoder_model = autoencoder.encoder.compile(
                *autoencoder.encoder.input_types(),
                weights=self._encoder_state_dict,
            )

        assert self.encoder_model is not None
        return self.encoder_model

    @traced(message="AutoencoderKLFlux2Model.build_fused_decode")
    def build_fused_decode(
        self, device: Device, num_channels: int
    ) -> Callable[..., Any]:
        """Build a fused postprocess + VAE decode compiled graph.

        Combines BN denormalization, unpatchify, and VAE decoding into a single
        compiled graph, eliminating the intermediate tensor and device sync
        between the two previously separate compiled graphs.  The reshape from
        packed (B, S, C) to spatial (B, H, W, C) is the first op inside the
        graph; spatial dims are conveyed via shape-carrier tensors so the same
        compiled graph handles any image size without recompilation.

        Args:
            device: Target device for the compiled graph.
            num_channels: Number of latent channels (bn.running_mean shape[0]).

        Returns:
            Compiled callable taking (latents_bsc, h_carrier, w_carrier)
            and returning the decoded image tensor.
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
            elif key == "bn.running_mean":
                fused_weights["bn_mean"] = weight_data
            elif key == "bn.running_var":
                fused_weights["bn_var"] = weight_data

        with F.lazy():
            autoencoder = AutoencoderKLFlux2(self.config)
            fused = PostprocessAndDecode(
                decoder=autoencoder.decoder,
                bn_mean=self.bn_running_mean,
                bn_var=self.bn_running_var,
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
        """
        return SimpleNamespace(
            running_mean=self.bn_running_mean, running_var=self.bn_running_var
        )
