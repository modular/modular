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
from queue import Queue
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.profiler import Tracer, traced
from PIL import Image
from tqdm import tqdm

from ..autoencoders import AutoencoderKLFlux2Model
from ..mistral3.text_encoder import Mistral3TextEncoderModel
from .model import Flux2TransformerModel

if TYPE_CHECKING:
    from ..autoencoders.vae import DiagonalGaussianDistribution


@dataclass(kw_only=True)
class Flux2ModelInputs(PixelModelInputs):
    """
    Flux2-specific PixelModelInputs.

    Defaults:
    - width: 1024
    - height: 1024
    - guidance_scale: 4.0
    - num_inference_steps: 50
    - num_images_per_prompt: 1
    - input_image: None (optional input image for image-to-image generation)

    """

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    input_image: Image.Image | None = None
    """Optional input image for image-to-image generation (PIL.Image.Image).

    This field is used for Flux2 image-to-image generation where an input image
    is provided as a condition for the generation process.
    """


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

        self.build_prepare_prompt_embeddings()
        self.build_preprocess_latents()
        self.build_prepare_image_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_concat_image_latents()
        self.build_decode_latents()

        # A workaround to remove overhead from `functional.wrapped`.
        if unwrapped_transformer := self.transformer.unwrap_model():
            self.transformer = cast(Any, unwrapped_transformer)

        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:  # type: ignore[override]
        """Convert a PixelContext into Flux2ModelInputs."""
        if context.input_image is not None and isinstance(
            context.input_image, np.ndarray
        ):
            context.input_image = Image.fromarray(  # type: ignore[assignment]
                context.input_image.astype(np.uint8)
            )
        return Flux2ModelInputs.from_context(context)

    def build_prepare_prompt_embeddings(self) -> None:
        input_types = [
            TensorType(
                self.text_encoder.config.dtype,
                shape=["seq_len", "hidden_dim"],
                device=self.text_encoder.devices[0],
            )
            for _ in range(3)
        ]

        self.__dict__["_prepare_prompt_embeddings"] = max_compile(
            self._prepare_prompt_embeddings,
            input_types=input_types,
        )

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                DType.float32,
                shape=["batch", "channels", "height", 2, "width", 2],
                device=device,
            ),
        ]
        self.__dict__["_patchify_and_pack"] = max_compile(
            self._patchify_and_pack,
            input_types=input_types,
        )

    def build_prepare_image_latents(self) -> None:
        dtype = self.vae.config.dtype
        device = self.vae.devices[0]
        num_channels = self.vae.bn.running_mean.shape[0].dim

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
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
        ]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
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
        self.__dict__["concat_image_latents"] = max_compile(
            self.concat_image_latents,
            input_types=input_types,
        )

    def build_decode_latents(self) -> None:
        dtype = self.vae.config.dtype
        device = self.transformer.devices[0]
        num_channels = self.vae.bn.running_mean.shape[0].dim

        input_types = [
            TensorType(
                dtype,
                shape=["batch", "height", "width", num_channels],
                device=device,
            ),
            TensorType(dtype, shape=[num_channels], device=device),
            TensorType(dtype, shape=[num_channels], device=device),
        ]

        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=input_types,
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
        encoder_output: "DiagonalGaussianDistribution",
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
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

    @traced
    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: Device,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

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

    @traced
    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Create prompt embeddings and text position IDs for the transformer.

        Flux2 uses multiple hidden-state layers from the text encoder. Selected
        layers are padded/trimmed to a common sequence length, stacked, and then
        flattened across the layer/hidden dimensions.

        Args:
            tokens: TokenBuffer produced by tokenization / chat templating.
            num_images_per_prompt: Number of image generations per prompt.
            hidden_states_layers: Optional indices of hidden-state layers to use.

        Returns:
            A tuple of:
                - prompt_embeds: Tensor of shape (B', S, L*D)
                - text_ids: Tensor[int64] of shape (B', S, 4)
        """
        layers = hidden_states_layers or [10, 20, 30]

        with Tracer("text_encoder"):
            text_input_ids = Tensor.constant(
                tokens.array,
                dtype=DType.int64,
                device=self.text_encoder.devices[0],
            )
            hidden_states_all = self.text_encoder(text_input_ids)
            hidden_states_selected = [hidden_states_all[i] for i in layers]

        with Tracer("prompt_embeddings"):
            prompt_embeds = self._prepare_prompt_embeddings(
                *hidden_states_selected
            )
            batch_size, seq_len, _ = map(int, prompt_embeds.shape)

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

    def _prepare_prompt_embeddings(self, *hidden_states: Tensor) -> Tensor:
        # [B, L, S, D] -> [B, S, L, D] -> [B, S, L*D]
        stacked = F.stack(hidden_states, axis=0)
        stacked = F.unsqueeze(stacked, axis=0)
        stacked = F.permute(stacked, [0, 2, 1, 3])
        batch_size, seq_len, num_layers, hidden_dim = stacked.shape
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )

        return prompt_embeds

    @traced
    def decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Tensor:
        """Decode Flux2 packed latents into an image array (or return latents).

        Args:
            latents: Packed latents, shaped (B, S, C).
            height: Output image height in pixels.
            width: Output image width in pixels.
            output_type: "latent" to return latents, otherwise decode to NumPy.

        Returns:
            If output_type == "latent", returns latents (first element if B > 1).
            Otherwise returns a float32 HWC NumPy array.
        """
        if output_type == "latent":
            return latents[0] if int(latents.shape[0]) > 1 else latents

        h_latent = height // (self.vae_scale_factor * 2)
        w_latent = width // (self.vae_scale_factor * 2)

        # Unpack: (B, S, C) -> (B, H, W, C)
        batch = int(latents.shape[0])
        c = int(latents.shape[2])
        latents_bhwc = F.reshape(latents, (batch, h_latent, w_latent, c))

        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var
        latents_decoded = self._postprocess_latents(
            latents_bhwc, bn_mean, bn_var
        )

        # Decode with the VAE and normalize layout to HWC.
        decoded = self.vae.decode(latents_decoded)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

    def _postprocess_latents(
        self,
        latents_bhwc: Tensor,
        bn_mean: Tensor,
        bn_var: Tensor,
    ) -> Tensor:
        """Denormalize and unpatchify latents for VAE decoding.

        Args:
            latents_bhwc: Packed latents of shape (B, H, W, C).
            bn_mean: BatchNorm running mean of shape (C,).
            bn_var: BatchNorm running variance of shape (C,).

        Returns:
            Unpatchified latents of shape (B, C//4, H*2, W*2).
        """
        batch = latents_bhwc.shape[0]
        h = latents_bhwc.shape[1]
        w = latents_bhwc.shape[2]
        c = latents_bhwc.shape[3]

        # Permute: (B, H, W, C) -> (B, C, H, W)
        latents = F.permute(latents_bhwc, (0, 3, 1, 2))

        # BN denormalization
        bn_mean_r = F.reshape(bn_mean, (1, c, 1, 1))
        bn_var_r = F.reshape(bn_var, (1, c, 1, 1))
        bn_std = F.sqrt(bn_var_r + self.vae.config.batch_norm_eps)
        latents = latents * bn_std + bn_mean_r

        # Unpatchify: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents = F.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = F.reshape(latents, (batch, c // 4, h * 2, w * 2))

        return latents

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        """Convert a MAX Tensor to a CPU NumPy array (float32)."""
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _image_to_flat_hwc(image: np.ndarray) -> np.ndarray:
        """Convert a tensor-like NumPy image to a flat HWC float32 array."""
        img = np.asarray(image)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.float32, copy=False)

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
        return Tensor.from_dlpack(text_ids).to(device)

    @traced
    def preprocess_latents(
        self,
        latents: npt.NDArray[np.float32],
        latent_image_ids: npt.NDArray[np.float32],
    ) -> tuple[Tensor, Tensor]:
        with Tracer("host_to_device_latents"):
            latents_tensor = Tensor.from_dlpack(latents).to(
                self.transformer.devices[0]
            )

        with Tracer("patchify_and_pack"):
            batch = latents_tensor.shape[0]
            c = latents_tensor.shape[1]
            h = latents_tensor.shape[2]
            w = latents_tensor.shape[3]
            latents_tensor = F.reshape(
                latents_tensor, (batch, c, h // 2, 2, w // 2, 2)
            )
            latents_tensor = self._patchify_and_pack(latents_tensor)

        with Tracer("host_to_device_ids"):
            latent_image_ids_tensor = Tensor.from_dlpack(
                latent_image_ids.astype(np.int64)
            ).to(self.transformer.devices[0])

        return latents_tensor, latent_image_ids_tensor

    def _patchify_and_pack(self, latents: Tensor) -> Tensor:
        """Patchify (B,C,H,W)->(B,C*4,H//2,W//2) then pack to (B,H//2*W//2,C*4)."""
        latents = latents.cast(self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]

        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = F.reshape(latents, (batch, c * 4, h2, w2))

        # Pack: (B, C*4, H//2, W//2) -> (B, H//2*W//2, C*4)
        c4 = c * 4
        latents = F.reshape(latents, (batch, c4, h2 * w2))
        latents = F.permute(latents, (0, 2, 1))

        return latents

    def _pil_image_to_tensor(
        self,
        image: Image.Image,
    ) -> Tensor:
        img_array = (np.array(image, dtype=np.float32) / 127.5) - 1.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)
        img_tensor = (
            Tensor.from_dlpack(img_array)
            .to(self.vae.devices[0])
            .cast(self.vae.config.dtype)
        )

        return img_tensor

    def scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
        num_noise_tokens: int,
    ) -> Tensor:
        """Apply a single Euler update step in sigma space."""
        latents_sliced = F.slice_tensor(
            latents,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        noise_pred_sliced = F.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_noise_tokens), "num_tokens"),
                slice(None),
            ],
        )
        latents_dtype = latents_sliced.dtype
        latents_sliced = latents_sliced.cast(DType.float32)
        latents_sliced = latents_sliced + dt * noise_pred_sliced
        return latents_sliced.cast(latents_dtype)

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

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2ModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2PipelineOutput:
        """Run the Flux2 denoising loop and decode outputs.

        Args:
            model_inputs: Inputs containing tokens, latents, timesteps, sigmas, and IDs.
            callback_queue: Optional queue for streaming intermediate decoded outputs.
            output_type: Output mode ("np", "latent")

        Returns:
            Flux2PipelineOutput containing one output per batch element.
        """
        # 1) Encode prompts.
        prompt_embeds, text_ids = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )
        batch_size = int(prompt_embeds.shape[0])
        dtype = prompt_embeds.dtype

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            image_tensor = self._pil_image_to_tensor(model_inputs.input_image)
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=[image_tensor],
                batch_size=batch_size,
                device=self.vae.devices[0],
                dtype=self.vae.config.dtype,
            )

        # 2) Prepare latents and conditioning tensors.
        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )

        # 3) Prepare scheduler tensors.
        with Tracer("prepare_scheduler"):
            device = self.transformer.devices[0]
            guidance_key = f"{batch_size}_{model_inputs.guidance_scale}"
            if guidance_key in self._cached_guidance:
                guidance = self._cached_guidance[guidance_key]
            else:
                guidance = Tensor.full(
                    [latents.shape[0]],
                    model_inputs.guidance_scale,
                    device=device,
                    dtype=dtype,
                )
                self._cached_guidance[guidance_key] = guidance

            image_seq_len = int(latents.shape[1])
            num_inference_steps = model_inputs.num_inference_steps
            sigmas_key = f"{num_inference_steps}_{image_seq_len}"
            if sigmas_key in self._cached_sigmas:
                sigmas = self._cached_sigmas[sigmas_key]
            else:
                sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(device)
                self._cached_sigmas[sigmas_key] = sigmas
            all_timesteps, all_dts = self.prepare_scheduler(sigmas)

            # For faster tensor slicing inside the denoising loop.
            timesteps_seq: Any = all_timesteps
            dts_seq: Any = all_dts
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        # 4) Denoising loop.
        num_noise_tokens = int(latents.shape[1])

        is_img2img = image_latents is not None
        with Tracer("denoising_loop"):
            for i in tqdm(range(num_inference_steps), desc="Denoising"):
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
                        noise_pred = Tensor.from_dlpack(noise_pred)

                    with Tracer("scheduler_step"):
                        latents = self.scheduler_step(
                            latents, noise_pred, dt, num_noise_tokens
                        )

                    if hasattr(device, "synchronize"):
                        device.synchronize()

                    if callback_queue is not None:
                        callback_queue.put_nowait(
                            cast(
                                np.ndarray,
                                self.decode_latents(
                                    latents,
                                    model_inputs.height,
                                    model_inputs.width,
                                    output_type=output_type,
                                ),
                            )
                        )

        # 5) Decode final outputs per batch element.
        image_list = []
        with Tracer("decode_outputs"):
            for b in range(batch_size):
                with Tracer("slice_batch"):
                    latents_b = latents[b : b + 1]
                image_list.append(
                    self.decode_latents(
                        latents_b,
                        model_inputs.height,
                        model_inputs.width,
                        output_type=output_type,
                    )
                )

        return Flux2PipelineOutput(images=image_list)  # type: ignore[arg-type]
