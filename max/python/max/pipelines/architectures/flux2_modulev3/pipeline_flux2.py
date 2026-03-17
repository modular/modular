# ===----------------------------------------------------------------------=== #
from __future__ import annotations

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.graph.ops import rebind, shape_to_tensor
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import (
    DenoisingCacheState,
    DiffusionPipeline,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.profiler import Tracer, traced

from ..autoencoders import Flux2AutoencoderModel
from ..mistral3.text_encoder import Mistral3TextEncoderModel
from .model import Flux2TransformerModel

if TYPE_CHECKING:
    from ..autoencoders.vae import DiagonalGaussianDistribution


@dataclass(kw_only=True)
class Flux2ModelInputs:
    """Input container for Flux2 pipeline execution."""

    tokens: Tensor
    """Primary encoder token IDs on device."""

    latents: Tensor
    """Initial latent noise tensor on device."""

    latent_image_ids: Tensor
    """Latent image positional identifiers on device."""

    sigmas: Tensor
    """Precomputed sigma schedule for denoising, on device."""

    guidance: Tensor
    """Guidance scale broadcast tensor on device."""

    latent_h: int
    """Latent height in patches (height // vae_scale_factor)."""

    latent_w: int
    """Latent width in patches (width // vae_scale_factor)."""

    image_seq_len: int
    """Packed image sequence length ((latent_h // 2) * (latent_w // 2))."""

    h_carrier: Tensor
    """1-D shape-carrier of length latent_h // 2; encodes packed height as a
    symbolic Dim for the decode graph.  Content is never read."""

    w_carrier: Tensor
    """1-D shape-carrier of length latent_w // 2; encodes packed width as a
    symbolic Dim for the decode graph.  Content is never read."""

    height: int
    """Output image height in pixels."""

    width: int
    """Output image width in pixels."""

    num_inference_steps: int
    """Number of denoising steps to run."""

    num_images_per_prompt: int
    """Number of images to generate per prompt."""

    input_image: npt.NDArray[np.uint8] | None
    """Optional input image for image-to-image generation (HWC uint8)."""

    def __post_init__(self) -> None:
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(
                f"height must be a positive int. Got {self.height!r}"
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(
                f"width must be a positive int. Got {self.width!r}"
            )
        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                f"num_inference_steps must be a positive int. Got {self.num_inference_steps!r}"
            )
        if (
            not isinstance(self.num_images_per_prompt, int)
            or self.num_images_per_prompt <= 0
        ):
            raise ValueError(
                f"num_images_per_prompt must be > 0. Got {self.num_images_per_prompt!r}"
            )


@dataclass
class Flux2PipelineOutput:
    """Container for Flux2 pipeline results.

    Attributes:
        images:
            Either a list of decoded PIL images, a NumPy array, or a MAX Tensor.
            When a Tensor is returned, it may represent decoded image data or
            intermediate latents depending on the selected output mode.
    """

    images: np.ndarray | Tensor


class Flux2Pipeline(DiffusionPipeline):
    """Diffusion pipeline for Flux2 image generation.

    This pipeline wires together:
        - Mistral3 text encoder
        - Flux2 transformer denoiser
        - Flux2 VAE (with BatchNorm-based latent normalization)
    """

    unprefixed_weight_component = "transformer"
    default_num_inference_steps = 28
    default_residual_threshold = 0.06

    vae: Flux2AutoencoderModel
    text_encoder: Mistral3TextEncoderModel
    transformer: Flux2TransformerModel

    components = {
        "vae": Flux2AutoencoderModel,
        "text_encoder": Mistral3TextEncoderModel,
        "transformer": Flux2TransformerModel,
    }

    @traced(message="Flux2Pipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        self.vae_mode = getattr(self.vae, "vae_mode", "kl")
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if not block_out_channels:
            block_out_channels = getattr(
                self.vae.config,
                "encoder_block_out_channels",
                [64, 64, 64, 64],
            )
        # Store derived config/device references before unwrapping.
        self._transformer_config = self.transformer.config
        self._transformer_devices = self.transformer.devices
        self.vae_scale_factor = (
            2 ** (len(block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_prepare_image_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_concat_image_latents()
        self.build_decode_latents()

        self._init_cache_state(
            dtype=self.transformer.config.dtype,
            device=self.transformer.devices[0],
        )

        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}
        self._cached_shape_carriers: dict[int, Tensor] = {}

    @traced(message="Flux2Pipeline.prepare_inputs")
    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:  # type: ignore[override]
        """Convert a PixelContext into Flux2ModelInputs."""
        if context.latents.size == 0:
            raise ValueError(
                "Flux2Pipeline requires non-empty latents in PixelContext"
            )
        if context.latent_image_ids.size == 0:
            raise ValueError(
                "Flux2Pipeline requires non-empty latent_image_ids in PixelContext"
            )
        if context.sigmas.size == 0:
            raise ValueError(
                "Flux2Pipeline requires non-empty sigmas in PixelContext"
            )

        device = self.transformer.devices[0]

        # Retrieve cached sigmas, if possible.
        latent_h = context.height // self.vae_scale_factor
        latent_w = context.width // self.vae_scale_factor
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        sigmas_key = f"{context.num_inference_steps}_{image_seq_len}"
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor(
                storage=Buffer.from_dlpack(context.sigmas).to(device)
            )
            self._cached_sigmas[sigmas_key] = sigmas

        # Retrieve cached guidance, if possible.
        guidance_key = (
            f"{context.num_images_per_prompt}_{context.guidance_scale}"
        )
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = Tensor.full(
                [context.num_images_per_prompt],
                context.guidance_scale,
                device=device,
                dtype=self.transformer.config.dtype,
            )
            self._cached_guidance[guidance_key] = guidance

        # Retrieve cached shape carriers, if possible.
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                self._cached_shape_carriers[n] = Tensor.from_dlpack(
                    np.empty(n, dtype=np.float32)
                )
        h_carrier = self._cached_shape_carriers[packed_h]
        w_carrier = self._cached_shape_carriers[packed_w]

        return Flux2ModelInputs(
            tokens=Tensor(
                storage=Buffer.from_dlpack(context.tokens.array).to(
                    self.text_encoder.devices[0]
                )
            ),
            latents=Tensor(
                storage=Buffer.from_dlpack(context.latents).to(device)
            ),
            latent_image_ids=Tensor(
                storage=Buffer.from_dlpack(context.latent_image_ids).to(device)
            ),
            sigmas=sigmas,
            guidance=guidance,
            latent_h=latent_h,
            latent_w=latent_w,
            image_seq_len=image_seq_len,
            h_carrier=h_carrier,
            w_carrier=w_carrier,
            height=context.height,
            width=context.width,
            num_inference_steps=context.num_inference_steps,
            num_images_per_prompt=context.num_images_per_prompt,
            input_image=context.input_image,
        )

    @traced(message="Flux2Pipeline.build_preprocess_latents")
    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        if self.vae_mode == "kl":
            input_types = [
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", "width"],
                    device=device,
                ),
            ]
            self.__dict__["_patchify_and_pack"] = max_compile(
                self._patchify_and_pack,
                input_types=input_types,
            )
        else:
            input_types = [
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", "width"],
                    device=device,
                ),
            ]
            self.__dict__["_pack_latents_direct"] = max_compile(
                self._pack_latents_direct,
                input_types=input_types,
            )

    @traced(message="Flux2Pipeline.build_prepare_image_latents")
    def build_prepare_image_latents(self) -> None:
        if self.vae_mode != "kl":
            return
        dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        num_channels = int(self.vae.bn.running_mean.shape[0])

        c = num_channels // 4
        self.__dict__["_normalize_and_pack_image_latent"] = max_compile(
            self._normalize_and_pack_image_latent,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", c, "height", 2, "width", 2],
                    device=device,
                ),
                TensorType(dtype, shape=[num_channels], device=device),
                TensorType(dtype, shape=[num_channels], device=device),
            ],
        )

    @traced(message="Flux2Pipeline.build_prepare_scheduler")
    def build_prepare_scheduler(self) -> None:
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=input_types,
        )

    @traced(message="Flux2Pipeline.build_scheduler_step")
    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(
                dtype, shape=["batch", "pred_seq", "channels"], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    @traced(message="Flux2Pipeline.build_concat_image_latents")
    def build_concat_image_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(
                dtype, shape=["batch", "img_seq", "channels"], device=device
            ),
            TensorType(DType.int64, shape=["batch", "seq", 4], device=device),
            TensorType(
                DType.int64, shape=["batch", "img_seq", 4], device=device
            ),
        ]
        self.__dict__["concat_image_latents"] = max_compile(
            self.concat_image_latents,
            input_types=input_types,
        )

    @traced(message="Flux2Pipeline.build_decode_latents")
    def build_decode_latents(self) -> None:
        if self.vae_mode != "kl":
            return
        device = self.transformer.devices[0]
        self._bn_mean = self.vae.bn.running_mean
        self._bn_var = self.vae.bn.running_var
        num_channels = int(self._bn_mean.shape[0])
        self._postprocess_and_decode = self.vae.build_fused_decode(
            device, num_channels
        )

    def concat_image_latents(
        self,
        latents: Tensor,
        image_latents: Tensor,
        latent_image_ids: Tensor,
        image_latent_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        latents_concat = F.concat([latents, image_latents], axis=1)
        latent_image_ids_concat = F.concat(
            [latent_image_ids, image_latent_ids], axis=1
        )
        return latents_concat, latent_image_ids_concat

    @staticmethod
    def _prepare_image_ids(
        latent_shapes: list[tuple[int, int]],
        scale: int = 10,
        device: Device = CPU(),
    ) -> Tensor:
        all_coords = []
        for i, (height, width) in enumerate(latent_shapes):
            t_coord = scale + scale * i
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)

            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
            all_coords.append(coords.reshape(-1, 4))

        combined = np.concatenate(all_coords, axis=0)
        combined = np.expand_dims(combined, 0)  # (1, total_seq, 4)
        return Tensor.from_dlpack(np.ascontiguousarray(combined)).to(device)

    @staticmethod
    def retrieve_latents(
        encoder_output: "DiagonalGaussianDistribution" | dict[str, Tensor] | Tensor,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        if isinstance(encoder_output, dict):
            if "latent_dist" in encoder_output:
                encoder_output = encoder_output["latent_dist"]
            elif "latents" in encoder_output:
                latents = encoder_output["latents"]
                return latents
        if isinstance(encoder_output, Tensor):
            return encoder_output
        if hasattr(encoder_output, "mode") and sample_mode == "mode":
            return encoder_output.mode()
        elif hasattr(encoder_output, "sample") and sample_mode == "sample":
            return encoder_output.sample(generator=generator)
        else:
            raise AttributeError(
                f"Could not access latents from encoder_output. "
                f"Expected DiagonalGaussianDistribution with 'mode' or "
                f"'sample' method, got {type(encoder_output)}"
            )

    def _normalize_and_pack_image_latent(
        self,
        image_latents: Tensor,
        bn_mean: Tensor,
        bn_var: Tensor,
    ) -> Tensor:
        """Finish patchify + BN normalize + pack.

        Input: 6D tensor (B, C, H', 2, W', 2) from the eager first reshape.
        Output: packed (B, H'*W', C*4).
        """
        # 1. Finish patchify: (B, C, H', 2, W', 2) -> (B, C*4, H', W')
        batch = image_latents.shape[0]
        c = image_latents.shape[1]
        h = image_latents.shape[2]
        w = image_latents.shape[4]
        image_latents = F.permute(image_latents, (0, 1, 3, 5, 2, 4))
        image_latents = F.reshape(image_latents, (batch, c * 4, h, w))

        # 2. BN normalize
        num_channels = bn_mean.shape[0]
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - bn_mean) / bn_std

        # 3. Pack: (B, C*4, H', W') -> (B, H'*W', C*4)
        num_ch = image_latents.shape[1]
        image_latents = F.reshape(image_latents, (batch, num_ch, h * w))
        image_latents = F.permute(image_latents, (0, 2, 1))

        return image_latents

    @traced(message="Flux2Pipeline.prepare_image_latents")
    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: Device,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        if self.vae_mode != "kl":
            return self._prepare_image_latents_tiny(
                images=images,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
                sample_mode=sample_mode,
            )

        bn_mean = self._bn_mean
        bn_var = self._bn_var

        packed_latents = []
        latent_shapes = []

        for image in images:
            image = image.to(device).cast(dtype)

            encoder_output = self.vae.encode(image, return_dict=True)
            if isinstance(encoder_output, dict):
                encoder_output = encoder_output["latent_dist"]
            raw_latents = self.retrieve_latents(
                encoder_output,
                generator=generator,
                sample_mode=sample_mode,
            )

            b, c, raw_h, raw_w = map(int, raw_latents.shape)
            latent_shapes.append((raw_h // 2, raw_w // 2))

            latents_6d = F.reshape(
                raw_latents,
                (b, c, raw_h // 2, 2, raw_w // 2, 2),
            )
            packed = self._normalize_and_pack_image_latent(
                latents_6d, bn_mean, bn_var
            )
            packed_latents.append(packed)

        # Generate image IDs.
        image_latent_ids = self._prepare_image_ids(latent_shapes, device=device)

        # Assemble final tensors. Each packed is (B, seq, C*4).
        if len(packed_latents) == 1:
            image_latents = packed_latents[0]
        else:
            image_latents = F.concat(packed_latents, axis=1)

        if batch_size > 1:
            image_latents = F.tile(image_latents, (batch_size, 1, 1))
            image_latent_ids = F.tile(image_latent_ids, (batch_size, 1, 1))
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def _prepare_image_latents_tiny(
        self,
        images: list[Tensor],
        batch_size: int,
        device: Device,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        packed_latents = []
        latent_shapes = []

        for image in images:
            image = image.to(device).cast(dtype)
            encoder_output = self.vae.encode(image, return_dict=True)
            raw_latents = self.retrieve_latents(
                encoder_output,
                generator=generator,
                sample_mode=sample_mode,
            )

            b, c, raw_h, raw_w = map(int, raw_latents.shape)
            latent_shapes.append((raw_h, raw_w))

            packed = F.reshape(raw_latents, (b, c, raw_h * raw_w))
            packed = F.permute(packed, (0, 2, 1))
            packed_latents.append(packed)

        image_latent_ids = self._prepare_image_ids(latent_shapes, device=device)
        image_latents = (
            packed_latents[0]
            if len(packed_latents) == 1
            else F.concat(packed_latents, axis=1)
        )

        if batch_size > 1:
            image_latents = F.tile(image_latents, (batch_size, 1, 1))
            image_latent_ids = F.tile(image_latent_ids, (batch_size, 1, 1))
        image_latent_ids = image_latent_ids.to(device)
        return image_latents, image_latent_ids

    @traced(message="Flux2Pipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self,
        tokens: Tensor,
        num_images_per_prompt: int = 1,
        attention_mask: Tensor | npt.ArrayLike | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Create prompt embeddings and text position IDs for the transformer.

        The text encoder returns fused prompt embeddings directly, with hidden
        states from the configured layers already stacked and merged across the
        layer/hidden dimensions.

        Args:
            tokens: Token ID tensor of shape (S,) on the text encoder device.
            num_images_per_prompt: Number of image generations per prompt.

        Returns:
            A tuple of:
                - prompt_embeds: Tensor of shape (B', S, L*D)
                - text_ids: Tensor[int64] of shape (B', S, 4)
        """
        # Shape metadata is host-side; this does not trigger a GPU sync.
        seq_len = int(tokens.shape[0])
        # TODO: Generalize this if diffusion pipelines ever need batched
        # text-encoder inputs instead of the current batch_size=1 contract.
        batch_size = 1

        with Tracer("text_encoder"):
            if attention_mask is None:
                prompt_embeds = self.text_encoder(tokens)
            else:
                prompt_embeds = self.text_encoder(  # type: ignore[call-arg]
                    tokens,
                    attention_mask=attention_mask,
                )

        with Tracer("post_process"):
            if num_images_per_prompt != 1:
                prompt_embeds = F.tile(
                    prompt_embeds, (1, num_images_per_prompt, 1)
                )
                prompt_embeds = F.reshape(
                    prompt_embeds,
                    [batch_size * num_images_per_prompt, seq_len, -1],
                )

            batch_size_final = batch_size * num_images_per_prompt
            text_ids_key = f"{batch_size_final}_{seq_len}"
            if text_ids_key in self._cached_text_ids:
                text_ids = self._cached_text_ids[text_ids_key]
            else:
                text_ids = self._prepare_text_ids(
                    batch_size=batch_size_final,
                    seq_len=seq_len,
                    device=self.text_encoder.devices[0],
                )
                self._cached_text_ids[text_ids_key] = text_ids

        return prompt_embeds, text_ids

    @traced(message="Flux2Pipeline.decode_latents")
    def decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> np.ndarray | Tensor:
        """Decode Flux2 packed latents into CPU image data.

        Args:
            latents: Packed latents, shaped (B, S, C).
            h_carrier: 1-D shape carrier of length packed_h (content unused).
            w_carrier: 1-D shape carrier of length packed_w (content unused).

        Returns:
            CPU uint8 image data, usually shaped (B, H, W, C). The tiny VAE path
            may return a CPU Tensor to defer NumPy conversion out of the decode
            hot path.
        """
        if self.vae_mode != "kl":
            if hasattr(self.vae, "decode_packed_to_uint8_tensor"):
                return self.vae.decode_packed_to_uint8_tensor(
                    latents, h_carrier, w_carrier
                )
            batch = latents.shape[0]
            channels = latents.shape[2]
            height = h_carrier.shape[0]
            width = w_carrier.shape[0]
            latents = F.rebind(latents, [batch, height * width, channels])
            latents = F.reshape(latents, (batch, height, width, channels))
            latents = F.permute(latents, (0, 3, 1, 2))
            latents = latents.to(self.vae.devices[0]).cast(self.vae.config.dtype)
            if hasattr(self.vae, "decode_to_uint8_tensor"):
                return self.vae.decode_to_uint8_tensor(latents)
            if hasattr(self.vae, "decode_to_numpy"):
                return self.vae.decode_to_numpy(latents)
            decoded = self.vae.decode(latents)
            decoded = F.permute(decoded, (0, 2, 3, 1))
            decoded = decoded * 0.5 + 0.5
            decoded = F.max(decoded, 0.0)
            decoded = F.min(decoded, 1.0)
            decoded = decoded * 255.0
            decoded = F.transfer_to(F.cast(decoded, DType.uint8), CPU())
            return np.from_dlpack(decoded)

        decoded = self._postprocess_and_decode(latents, h_carrier, w_carrier)

        return np.from_dlpack(decoded)  # (B, H, W, C)

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: Device,
    ) -> Tensor:
        """Create 4D text position IDs in (T, H, W, L) format.

        For text tokens:
            T = 0, H = 0, W = 0, and L indexes the token position [0..seq_len-1].

        Returns:
            Tensor[int64] of shape (batch_size, seq_len, 4).
        """
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),  # T
                np.zeros(seq_len, dtype=np.int64),  # H
                np.zeros(seq_len, dtype=np.int64),  # W
                np.arange(seq_len, dtype=np.int64),  # L
            ],
            axis=-1,
        )  # (seq_len, 4)

        text_ids = np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        return Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(text_ids)).to(
                device
            )
        )

    @traced(message="Flux2Pipeline.preprocess_latents")
    def preprocess_latents(self, latents: Tensor) -> Tensor:
        if self.vae_mode != "kl":
            return self._pack_latents_direct(latents)
        return self._patchify_and_pack(latents)

    def _pack_latents_direct(self, latents: Tensor) -> Tensor:
        latents = latents.cast(self.transformer.config.dtype)
        batch = latents.shape[0]
        channels = latents.shape[1]
        height = latents.shape[2]
        width = latents.shape[3]
        latents = F.reshape(latents, (batch, channels, height * width))
        latents = F.permute(latents, (0, 2, 1))
        return latents

    def _patchify_and_pack(self, latents: Tensor) -> Tensor:
        """Patchify (B,C,H,W)->(B,C*4,H//2,W//2) then pack to (B,H//2*W//2,C*4)."""
        latents = latents.cast(self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h = latents.shape[2]
        w = latents.shape[3]
        latents = F.rebind(latents, [batch, c, (h // 2) * 2, (w // 2) * 2])
        latents = F.reshape(latents, (batch, c, h // 2, 2, w // 2, 2))
        h2 = latents.shape[2]
        w2 = latents.shape[4]

        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = F.reshape(latents, (batch, c * 4, h2, w2))

        # Pack: (B, C*4, H//2, W//2) -> (B, H//2*W//2, C*4)
        c4 = c * 4
        latents = F.reshape(latents, (batch, c4, h2 * w2))
        latents = F.permute(latents, (0, 2, 1))

        return latents

    def _numpy_image_to_tensor(
        self,
        image: npt.NDArray[np.uint8],
    ) -> Tensor:
        img_array = (image.astype(np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)
        return (
            Tensor.from_dlpack(img_array)
            .to(self.vae.devices[0])
            .cast(self.vae.config.dtype)
        )

    def scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """Apply a single Euler update step in sigma space.

        Slices ``noise_pred`` to ``latents.shape[1]`` tokens before applying
        the Euler update, which discards the image-latent predictions in the
        img2img case where ``noise_pred`` covers a concatenated sequence.
        """
        num_tokens = shape_to_tensor([latents.shape[1]])
        noise_pred_sliced = F.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents.dtype
        latents_f32 = latents.cast(DType.float32)
        noise_pred_sliced = rebind(noise_pred_sliced, latents_f32.shape)
        return (latents_f32 + dt * noise_pred_sliced).cast(latents_dtype)

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute timesteps and dt values from sigmas in a single fused graph.

        Returns:
            (all_timesteps, all_dts) where timesteps = sigmas[:-1] cast to
            model dtype, and dts = sigmas[1:] - sigmas[:-1] (float32).
        """
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        all_timesteps = sigmas_curr.cast(self.transformer.config.dtype)
        return all_timesteps, all_dt

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        return self.transformer(
            kwargs["latents"],
            kwargs["prompt_embeds"],
            kwargs["timestep"],
            kwargs["latent_image_ids"],
            kwargs["text_ids"],
            kwargs["guidance"],
            prev_residual=cache_state.prev_residual,
            prev_output=cache_state.prev_output,
        )

    @traced(message="Flux2Pipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2ModelInputs,
    ) -> Flux2PipelineOutput:
        """Run the Flux2 denoising loop and decode outputs.

        Args:
            model_inputs: Inputs containing tokens, latents, timesteps, sigmas, and IDs.

        Returns:
            Flux2PipelineOutput containing one output per batch element.
        """
        # 1) Encode prompts.
        prompt_embeds, text_ids = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            image_tensor = self._numpy_image_to_tensor(model_inputs.input_image)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=batch_size,
                device=self.vae.devices[0],
                dtype=self.vae.config.dtype,
            )

        # 2) Prepare latents and conditioning tensors.
        latents = self.preprocess_latents(model_inputs.latents)
        latent_image_ids = model_inputs.latent_image_ids

        # 3) Prepare scheduler tensors.
        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.prepare_scheduler(model_inputs.sigmas)
            guidance = model_inputs.guidance

            # For faster tensor slicing inside the denoising loop.
            timesteps_seq: Any = all_timesteps
            dts_seq: Any = all_dts
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        # 4) Denoising loop.
        is_img2img = image_latents is not None
        device = self.transformer.devices[0]

        seq_len_for_cache = model_inputs.image_seq_len
        if image_latents is not None:
            seq_len_for_cache += int(image_latents.shape[1])
        cache = self.create_cache_state(
            batch_size, seq_len_for_cache, self.transformer.config
        )

        with Tracer("denoising_loop"):
            for i in range(model_inputs.num_inference_steps):
                with Tracer(f"denoising_step_{i}"):
                    timestep = timesteps_seq[i : i + 1]
                    dt = dts_seq[i : i + 1]

                    if is_img2img:
                        assert image_latents is not None
                        assert image_latent_ids is not None
                        latents_concat, latent_image_ids_concat = (
                            self.concat_image_latents(
                                latents,
                                image_latents,
                                latent_image_ids,
                                image_latent_ids,
                            )
                        )
                    else:
                        latents_concat = latents
                        latent_image_ids_concat = latent_image_ids

                    noise_pred = self.run_denoising_step(
                        step=i,
                        cache_state=cache,
                        device=device,
                        latents=latents_concat,
                        prompt_embeds=prompt_embeds,
                        timestep=timestep,
                        latent_image_ids=latent_image_ids_concat,
                        text_ids=text_ids,
                        guidance=guidance,
                    )

                    with Tracer("scheduler_step"):
                        latents = self.scheduler_step(latents, noise_pred, dt)

        # 5) Decode final outputs for all batch elements in a single pass.
        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                model_inputs.latent_image_ids,
                model_inputs.h_carrier,
                model_inputs.w_carrier,
            )

        return Flux2PipelineOutput(images=images)
