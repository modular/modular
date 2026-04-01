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
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue, ops
from max.pipelines.core import PixelContext
from max.pipelines.lib import float32_array_to_buffer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    DiffusionPipelineOutput,
    max_compile,
)
from max.pipelines.lib.utils import BoundedCache
from max.profiler import Tracer, traced

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3.text_encoder import Mistral3TextEncoderModel
from .model import Flux2TransformerModel

if TYPE_CHECKING:
    from ..autoencoders_modulev3.vae import DiagonalGaussianDistribution


@dataclass(kw_only=True)
class Flux2ModelInputs:
    """Input container for Flux2 pipeline execution."""

    tokens: Buffer
    """Primary encoder token IDs on device."""

    latents: Buffer
    """Initial latent noise tensor on device."""

    latent_image_ids: Buffer
    """Latent image positional identifiers on device."""

    sigmas: Buffer
    """Precomputed sigma schedule for denoising, on device."""

    guidance: Buffer
    """Guidance scale broadcast tensor on device."""

    latent_h: int
    """Latent height in patches (height // vae_scale_factor)."""

    latent_w: int
    """Latent width in patches (width // vae_scale_factor)."""

    image_seq_len: int
    """Packed image sequence length ((latent_h // 2) * (latent_w // 2))."""

    h_carrier: Buffer
    """1-D shape-carrier of length latent_h // 2; encodes packed height as a
    symbolic Dim for the decode graph.  Content is never read."""

    w_carrier: Buffer
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


class Flux2Pipeline(DiffusionPipeline):
    """Diffusion pipeline for Flux2 image generation.

    This pipeline wires together:
        - Mistral3 text encoder
        - Flux2 transformer denoiser
        - Flux2 VAE (with BatchNorm-based latent normalization)
    """

    default_num_inference_steps = 28

    vae: AutoencoderKLFlux2Model
    text_encoder: Mistral3TextEncoderModel
    transformer: Flux2TransformerModel

    components = {
        "vae": AutoencoderKLFlux2Model,
        "text_encoder": Mistral3TextEncoderModel,
        "transformer": Flux2TransformerModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize derived attributes that depend on loaded components."""
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_prepare_image_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_concat_image_latents()
        self.build_decode_latents()
        self.build_concat_packed_latents()

        self._cached_guidance: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_text_ids: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_sigmas: BoundedCache[str, Buffer] = BoundedCache(32)
        self._cached_shape_carriers: BoundedCache[int, Buffer] = BoundedCache(
            32
        )
        self._repeat_prompt_embeddings_cache: dict[
            int, Callable[[Buffer], Buffer]
        ] = {}
        self._repeat_image_conditioning_cache: dict[
            int, Callable[[Buffer, Buffer], tuple[Buffer, Buffer]]
        ] = {}

    @traced
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
            sigmas = Buffer.from_dlpack(context.sigmas).to(device)
            self._cached_sigmas[sigmas_key] = sigmas

        # Retrieve cached guidance, if possible.
        guidance_key = (
            f"{context.num_images_per_prompt}_{context.guidance_scale}"
        )
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = float32_array_to_buffer(
                np.full(
                    [context.num_images_per_prompt],
                    context.guidance_scale,
                    dtype=np.float32,
                ),
                dtype=self.transformer.config.dtype,
                device=device,
            )
            self._cached_guidance[guidance_key] = guidance

        # Retrieve cached shape carriers, if possible.
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        for n in (packed_h, packed_w):
            if n not in self._cached_shape_carriers:
                self._cached_shape_carriers[n] = Buffer.from_numpy(
                    np.empty(n, dtype=np.float32)
                )
        h_carrier = self._cached_shape_carriers[packed_h]
        w_carrier = self._cached_shape_carriers[packed_w]

        return Flux2ModelInputs(
            tokens=Buffer.from_dlpack(context.tokens.array).to(
                self.text_encoder.devices[0]
            ),
            latents=Buffer.from_dlpack(context.latents).to(device),
            latent_image_ids=Buffer.from_dlpack(context.latent_image_ids).to(
                device
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

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                DType.float32,
                shape=["batch", "channels", "height", "width"],
                device=device,
            ),
        ]
        self._patchify_and_pack = cast(
            Callable[[Buffer], Buffer],
            max_compile(
                self._patchify_and_pack_graph,
                input_types=input_types,
            ),
        )

    def build_prepare_image_latents(self) -> None:
        dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        num_channels = int(self.vae.bn.running_mean.shape[0])

        c = num_channels // 4
        self._normalize_and_pack_image_latent = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                self._normalize_and_pack_image_latent_graph,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", c, "height", "width"],
                        device=device,
                    ),
                    TensorType(dtype, shape=[num_channels], device=device),
                    TensorType(dtype, shape=[num_channels], device=device),
                ],
            ),
        )

    def build_prepare_scheduler(self) -> None:
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.prepare_scheduler = cast(
            Callable[[Buffer], tuple[Buffer, Buffer]],
            max_compile(
                self._prepare_scheduler_graph,
                input_types=input_types,
            ),
        )

    def build_concat_packed_latents(self) -> None:
        self._concat_packed_latents = cast(
            Callable[[Buffer, Buffer], Buffer],
            max_compile(
                self._concat_packed_latents_graph,
                input_types=[
                    TensorType(
                        self.vae.config.dtype,
                        shape=["batch", "seq", "channels"],
                        device=self.vae.devices[0],
                    ),
                    TensorType(
                        self.vae.config.dtype,
                        shape=["batch", "img_seq", "channels"],
                        device=self.vae.devices[0],
                    ),
                ],
            ),
        )

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
        self._scheduler_step = cast(
            Callable[[Buffer, Buffer, Buffer], Buffer],
            max_compile(
                self._scheduler_step_graph,
                input_types=input_types,
            ),
        )

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
        self.concat_image_latents = cast(
            Callable[[Buffer, Buffer, Buffer, Buffer], tuple[Buffer, Buffer]],
            max_compile(
                self._concat_image_latents_graph,
                input_types=input_types,
            ),
        )

    def build_decode_latents(self) -> None:
        device = self.transformer.devices[0]
        self._bn_mean = self.vae.bn.running_mean
        self._bn_var = self.vae.bn.running_var
        num_channels = int(self._bn_mean.shape[0])
        self._postprocess_and_decode = self.vae.build_fused_decode(
            device, num_channels
        )

    def _concat_image_latents_graph(
        self,
        latents: TensorValue,
        image_latents: TensorValue,
        latent_image_ids: TensorValue,
        image_latent_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            ops.concat([latents, image_latents], axis=1),
            ops.concat([latent_image_ids, image_latent_ids], axis=1),
        )

    @staticmethod
    def _prepare_image_ids(
        latent_shapes: list[tuple[int, int]],
        scale: int = 10,
        device: Device = CPU(),
    ) -> Buffer:
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
        return Buffer.from_numpy(np.ascontiguousarray(combined)).to(device)

    @staticmethod
    def retrieve_latents(
        encoder_output: "DiagonalGaussianDistribution",
        generator: object | None = None,
        sample_mode: str = "mode",
    ) -> Buffer:
        if hasattr(encoder_output, "mode") and sample_mode == "mode":
            return cast(Buffer, encoder_output.mode())
        elif hasattr(encoder_output, "sample") and sample_mode == "sample":
            return cast(Buffer, encoder_output.sample(generator=generator))
        else:
            raise AttributeError(
                f"Could not access latents from encoder_output. "
                f"Expected DiagonalGaussianDistribution with 'mode' or "
                f"'sample' method, got {type(encoder_output)}"
            )

    def _normalize_and_pack_image_latent_graph(
        self,
        image_latents: TensorValue,
        bn_mean: TensorValue,
        bn_var: TensorValue,
    ) -> TensorValue:
        """Finish patchify + BN normalize + pack.

        Input: raw 4D latents (B, C, H, W).
        Output: packed (B, H'*W', C*4).
        """
        # 1. Patchify: (B, C, H, W) -> (B, C*4, H//2, W//2)
        batch = image_latents.shape[0]
        c = image_latents.shape[1]
        raw_h = image_latents.shape[2]
        raw_w = image_latents.shape[3]
        image_latents = ops.rebind(
            image_latents,
            [batch, c, (raw_h // 2) * 2, (raw_w // 2) * 2],
        )
        image_latents = ops.reshape(
            image_latents,
            (batch, c, raw_h // 2, 2, raw_w // 2, 2),
        )
        h = image_latents.shape[2]
        w = image_latents.shape[4]
        image_latents = ops.permute(image_latents, [0, 1, 3, 5, 2, 4])
        image_latents = ops.reshape(image_latents, (batch, c * 4, h, w))

        # 2. BN normalize
        num_channels = bn_mean.shape[0]
        bn_mean = ops.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = ops.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = ops.sqrt(bn_var + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - bn_mean) / bn_std

        # 3. Pack: (B, C*4, H', W') -> (B, H'*W', C*4)
        num_ch = image_latents.shape[1]
        image_latents = ops.reshape(image_latents, (batch, num_ch, h * w))
        image_latents = ops.permute(image_latents, [0, 2, 1])

        return image_latents

    def _get_repeat_prompt_embeddings(
        self, repeats: int
    ) -> Callable[[Buffer], Buffer]:
        if repeats not in self._repeat_prompt_embeddings_cache:

            def repeat_prompt_embeddings(
                prompt_embeds: TensorValue,
            ) -> TensorValue:
                batch = prompt_embeds.shape[0]
                seq = prompt_embeds.shape[1]
                hidden_dim = prompt_embeds.shape[2]
                prompt_embeds = ops.tile(prompt_embeds, [1, repeats, 1])
                return ops.reshape(
                    prompt_embeds,
                    (batch * repeats, seq, hidden_dim),
                )

            self._repeat_prompt_embeddings_cache[repeats] = cast(
                Callable[[Buffer], Buffer],
                max_compile(
                    repeat_prompt_embeddings,
                    input_types=[
                        TensorType(
                            self.text_encoder.config.dtype,
                            shape=["batch", "seq", "hidden_dim"],
                            device=self.text_encoder.devices[0],
                        ),
                    ],
                ),
            )
        return self._repeat_prompt_embeddings_cache[repeats]

    def _repeat_image_conditioning(
        self, repeats: int
    ) -> Callable[[Buffer, Buffer], tuple[Buffer, Buffer]]:
        if repeats not in self._repeat_image_conditioning_cache:

            def repeat_image_conditioning(
                image_latents: TensorValue,
                image_latent_ids: TensorValue,
            ) -> tuple[TensorValue, TensorValue]:
                return (
                    ops.tile(image_latents, [repeats, 1, 1]),
                    ops.tile(image_latent_ids, [repeats, 1, 1]),
                )

            self._repeat_image_conditioning_cache[repeats] = cast(
                Callable[[Buffer, Buffer], tuple[Buffer, Buffer]],
                max_compile(
                    repeat_image_conditioning,
                    input_types=[
                        TensorType(
                            self.vae.config.dtype,
                            shape=["batch", "seq", "channels"],
                            device=self.vae.devices[0],
                        ),
                        TensorType(
                            DType.int64,
                            shape=["batch", "seq", 4],
                            device=self.vae.devices[0],
                        ),
                    ],
                ),
            )
        return self._repeat_image_conditioning_cache[repeats]

    def _concat_packed_latents_graph(
        self, left: TensorValue, right: TensorValue
    ) -> TensorValue:
        return ops.concat([left, right], axis=1)

    @traced
    def prepare_image_latents(
        self,
        images: list[Buffer],
        batch_size: int,
        device: Device,
        dtype: DType,
        generator: object | None = None,
        sample_mode: str = "mode",
    ) -> tuple[Buffer, Buffer]:
        bn_mean = self._bn_mean
        bn_var = self._bn_var
        packed_latents: list[Buffer] = []
        latent_shapes = []

        for image in images:
            if image.dtype == dtype:
                image = image.to(device)
            else:
                image_np = image.to_numpy()
                if image_np.dtype != np.float32:
                    image_np = image_np.astype(np.float32)
                image = float32_array_to_buffer(
                    image_np,
                    dtype=dtype,
                    device=device,
                )

            encoder_output = self.vae.encode(
                Tensor(storage=image), return_dict=True
            )
            if isinstance(encoder_output, dict):
                encoder_output = encoder_output["latent_dist"]
            raw_latents = self.retrieve_latents(
                encoder_output,
                generator=generator,
                sample_mode=sample_mode,
            )

            _b, _c, raw_h, raw_w = map(int, raw_latents.shape)
            latent_shapes.append((raw_h // 2, raw_w // 2))

            packed = self._normalize_and_pack_image_latent(
                raw_latents, bn_mean, bn_var
            )
            packed_latents.append(packed)

        # Generate image IDs.
        image_latent_ids = self._prepare_image_ids(latent_shapes, device=device)

        # Assemble final tensors. Each packed is (B, seq, C*4).
        if len(packed_latents) == 1:
            image_latents = packed_latents[0]
        else:
            image_latents = packed_latents[0]
            for packed in packed_latents[1:]:
                image_latents = self._concat_packed_latents(
                    image_latents, packed
                )

        if batch_size > 1:
            image_latents, image_latent_ids = self._repeat_image_conditioning(
                batch_size
            )(
                image_latents,
                image_latent_ids,
            )
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    @traced
    def prepare_prompt_embeddings(
        self,
        tokens: Buffer,
        num_images_per_prompt: int = 1,
    ) -> tuple[Buffer, Buffer]:
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
        batch_size = 1  # text encoder always outputs a single batch

        with Tracer("text_encoder"):
            prompt_embeds = cast(
                Buffer,
                self.text_encoder(tokens),
            )

        with Tracer("post_process"):
            if num_images_per_prompt != 1:
                prompt_embeds = self._get_repeat_prompt_embeddings(
                    num_images_per_prompt
                )(prompt_embeds)

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

    @traced
    def decode_latents(
        self,
        latents: Buffer,
        h_carrier: Buffer,
        w_carrier: Buffer,
    ) -> npt.NDArray[np.uint8]:
        """Decode Flux2 packed latents into a (B, H, W, C) uint8 NumPy array.

        Args:
            latents: Packed latents, shaped (B, S, C).
            h_carrier: 1-D shape carrier of length packed_h (content unused).
            w_carrier: 1-D shape carrier of length packed_w (content unused).

        Returns:
            uint8 NumPy array of shape (B, H, W, C) with values in [0, 255].
        """
        decoded = self._postprocess_and_decode(latents, h_carrier, w_carrier)
        decoded_np: np.ndarray
        if hasattr(decoded, "__dlpack__"):
            decoded_np = np.from_dlpack(decoded)
        elif hasattr(decoded, "to_numpy"):
            decoded_np = decoded.to_numpy()
        else:
            decoded_np = decoded.driver_tensor.to_numpy()

        if decoded_np.dtype != np.uint8:
            decoded_np = decoded_np.astype(np.uint8, copy=False)
        return cast(npt.NDArray[np.uint8], decoded_np)

    @staticmethod
    def _prepare_text_ids(
        batch_size: int,
        seq_len: int,
        device: Device,
    ) -> Buffer:
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
        return Buffer.from_numpy(np.ascontiguousarray(text_ids)).to(device)

    @traced
    def preprocess_latents(self, latents: Buffer) -> Buffer:
        return self._patchify_and_pack(latents)

    def _patchify_and_pack_graph(self, latents: TensorValue) -> TensorValue:
        """Patchify (B,C,H,W)->(B,C*4,H//2,W//2) then pack to (B,H//2*W//2,C*4)."""
        latents = ops.cast(latents, self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h = latents.shape[2]
        w = latents.shape[3]
        latents = ops.rebind(latents, [batch, c, (h // 2) * 2, (w // 2) * 2])
        latents = ops.reshape(latents, (batch, c, h // 2, 2, w // 2, 2))
        h2 = latents.shape[2]
        w2 = latents.shape[4]

        latents = ops.permute(latents, [0, 1, 3, 5, 2, 4])
        latents = ops.reshape(latents, (batch, c * 4, h2, w2))

        # Pack: (B, C*4, H//2, W//2) -> (B, H//2*W//2, C*4)
        c4 = c * 4
        latents = ops.reshape(latents, (batch, c4, h2 * w2))
        latents = ops.permute(latents, [0, 2, 1])

        return latents

    def _numpy_image_to_buffer(
        self,
        image: npt.NDArray[np.uint8],
    ) -> Buffer:
        img_array = (image.astype(np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)
        return float32_array_to_buffer(
            img_array,
            dtype=self.vae.config.dtype,
            device=self.vae.devices[0],
        )

    def _scheduler_step_graph(
        self,
        latents: TensorValue,
        noise_pred: TensorValue,
        dt: TensorValue,
    ) -> TensorValue:
        """Apply a single Euler update step in sigma space.

        Slices ``noise_pred`` to ``latents.shape[1]`` tokens before applying
        the Euler update, which discards the image-latent predictions in the
        img2img case where ``noise_pred`` covers a concatenated sequence.
        """
        num_tokens = ops.shape_to_tensor([latents.shape[1]])
        noise_pred_sliced = ops.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents.dtype
        latents_f32 = ops.cast(latents, DType.float32)
        noise_pred_sliced = ops.rebind(noise_pred_sliced, latents_f32.shape)
        return ops.cast(latents_f32 + dt * noise_pred_sliced, latents_dtype)

    def _prepare_scheduler_graph(
        self, sigmas: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Precompute timesteps and dt values from sigmas in a single fused graph.

        Returns:
            (all_timesteps, all_dts) where timesteps = sigmas[:-1] cast to
            model dtype, and dts = sigmas[1:] - sigmas[:-1] (float32).
        """
        sigmas_curr = ops.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = ops.slice_tensor(sigmas, [slice(1, None)])
        all_dt = ops.sub(sigmas_next, sigmas_curr)
        all_timesteps = ops.cast(sigmas_curr, self.transformer.config.dtype)
        return all_timesteps, all_dt

    @traced
    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2ModelInputs,
    ) -> DiffusionPipelineOutput:
        """Run the Flux2 denoising loop and decode outputs.

        Args:
            model_inputs: Inputs containing tokens, latents, timesteps, sigmas, and IDs.

        Returns:
            DiffusionPipelineOutput containing one output per batch element.
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
            image_tensor = self._numpy_image_to_buffer(model_inputs.input_image)
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
            timesteps_seq = all_timesteps
            dts_seq = all_dts
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        # 4) Denoising loop.
        is_img2img = image_latents is not None
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

                    with Tracer("transformer"):
                        noise_pred = self.transformer(
                            latents_concat,
                            prompt_embeds,
                            timestep,
                            latent_image_ids_concat,
                            text_ids,
                            guidance,
                        )[0]

                    with Tracer("scheduler_step"):
                        latents = self._scheduler_step(latents, noise_pred, dt)

        # 5) Decode final outputs for all batch elements in a single pass.
        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                model_inputs.h_carrier,
                model_inputs.w_carrier,
            )

        return DiffusionPipelineOutput(images=images)
