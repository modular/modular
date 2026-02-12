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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.tensor import Tensor
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

    def prepare_inputs(self, context: PixelContext) -> Flux2ModelInputs:  # type: ignore[override]
        """Convert a PixelContext into Flux2ModelInputs."""
        if context.input_image is not None and isinstance(
            context.input_image, np.ndarray
        ):
            context.input_image = Image.fromarray(
                context.input_image.astype(np.uint8)
            )
        return Flux2ModelInputs.from_context(context)

    @staticmethod
    def _prepare_image_ids(
        image_latents: list[Tensor],
        scale: int = 10,
        device: Device | None = None,
    ) -> Tensor:
        if not isinstance(image_latents, list):
            raise ValueError(
                f"Expected `image_latents` to be a list, got {type(image_latents)}."
            )

        if len(image_latents) == 0:
            raise ValueError("Expected at least one image latent in the list.")

        if device is None:
            device = image_latents[0].device

        image_latent_ids = []

        for i, latent in enumerate(image_latents):
            latent_squeezed = F.squeeze(latent, axis=0)
            _, height, width = map(int, latent_squeezed.shape)
            t_coord = scale + scale * i
            t_coords = np.full((height, width), t_coord, dtype=np.int64)
            h_coords, w_coords = np.meshgrid(
                np.arange(height, dtype=np.int64),
                np.arange(width, dtype=np.int64),
                indexing="ij",
            )
            l_coords = np.zeros((height, width), dtype=np.int64)

            coords = np.stack([t_coords, h_coords, w_coords, l_coords], axis=-1)
            coords = coords.reshape(-1, 4)
            coords_tensor = Tensor.from_dlpack(coords).to(device)
            image_latent_ids.append(coords_tensor)

        image_latent_ids = F.concat(image_latent_ids, axis=0)
        image_latent_ids = F.unsqueeze(image_latent_ids, 0)

        return image_latent_ids

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
                f"Expected DiagonalGaussianDistribution with 'mode' or 'sample' method, "
                f"got {type(encoder_output)}"
            )

    def _encode_vae_image(
        self,
        image: Tensor,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> Tensor:
        if len(image.shape) != 4:
            raise ValueError(f"Expected image dims 4, got {len(image.shape)}.")

        encoder_output = self.vae.encode(image, return_dict=True)

        if isinstance(encoder_output, dict):
            encoder_output = encoder_output["latent_dist"]

        image_latents = self.retrieve_latents(
            encoder_output, generator=generator, sample_mode=sample_mode
        )
        image_latents = self._patchify_latents(image_latents)

        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - bn_mean) / bn_std

        return image_latents

    def prepare_image_latents(
        self,
        images: list[Tensor],
        batch_size: int,
        device: Device,
        dtype: DType,
        generator: Any = None,
        sample_mode: str = "mode",
    ) -> tuple[Tensor, Tensor]:
        image_latents = []
        for image in images:
            image = image.to(device).cast(dtype)
            latent = self._encode_vae_image(
                image=image, generator=generator, sample_mode=sample_mode
            )
            image_latents.append(latent)

        image_latent_ids = self._prepare_image_ids(image_latents, device=device)

        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)
            packed = F.squeeze(packed, axis=0)
            packed_latents.append(packed)

        image_latents = F.concat(packed_latents, axis=0)
        image_latents = F.unsqueeze(image_latents, 0)
        image_latents = F.tile(image_latents, (batch_size, 1, 1))

        image_latent_ids = F.tile(image_latent_ids, (batch_size, 1, 1))
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def _prepare_prompt_embeddings(
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
        max_seq = int(tokens.array.shape[-1])

        text_input_ids = Tensor.constant(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        hs_all = self.text_encoder(text_input_ids)

        selected: list[Tensor] = []
        for i in layers:
            hs = hs_all[i]
            hs = hs if isinstance(hs, Tensor) else Tensor.from_dlpack(hs)

            # Ensure [B, S, D]
            if hs.rank == 2:
                hs = F.unsqueeze(hs, axis=0)

            _, seq_len, _ = map(int, hs.shape)
            if seq_len < max_seq:
                hs = F.pad(
                    hs, pad=((0, 0), (0, max_seq - seq_len), (0, 0))
                )  # [B, max_seq, D]
            elif seq_len > max_seq:
                hs = hs[:, :max_seq, :]

            selected.append(hs)

        # [B, L, S, D] -> [B, S, L, D] -> [B, S, L*D]
        stacked = F.stack(selected, axis=1)
        stacked = F.permute(stacked, [0, 2, 1, 3])
        batch_size, seq_len, num_layers, hidden_dim = map(int, stacked.shape)
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = F.reshape(
                prompt_embeds, [batch_size * num_images_per_prompt, seq_len, -1]
            )

        text_ids = self._prepare_text_ids(
            batch_size=batch_size * num_images_per_prompt,
            seq_len=seq_len,
            device=self.text_encoder.devices[0],
        )

        return prompt_embeds, text_ids

    def _decode_latents(
        self,
        latents: Tensor,
        latent_image_ids: Tensor,
        output_type: Literal["np", "latent"] = "np",
    ) -> np.ndarray | Tensor:
        """Decode Flux2 packed latents into an image array (or return latents).

        Args:
            latents: Packed latents, typically shaped (B, S, C).
            latent_image_ids: Position IDs used to unpack into (B, C, H, W).
            output_type: "latent" to return latents, otherwise decode to NumPy.

        Returns:
            If output_type == "latent", returns latents (first element if B > 1).
            Otherwise returns a float32 HWC NumPy array.
        """
        if output_type == "latent":
            return latents[0] if int(latents.shape[0]) > 1 else latents

        # 1) Unpack sequence latents back to (B, C, H, W) using position IDs.
        latents_unpacked = self._unpack_latents_with_ids(
            latents, latent_image_ids
        )

        # 2) Undo Flux2 BatchNorm latent normalization (BN stats are stored in the VAE).
        bn_mean = self.vae.bn.running_mean
        bn_var = self.vae.bn.running_var

        num_channels = bn_mean.shape[0].dim
        bn_mean = F.reshape(bn_mean, (1, num_channels, 1, 1))
        bn_var = F.reshape(bn_var, (1, num_channels, 1, 1))
        bn_std = F.sqrt(bn_var + self.vae.config.batch_norm_eps)

        latents_unpacked = latents_unpacked * bn_std + bn_mean

        # 3) Unpatchify latents: (B, C, H, W) -> (B, C//4, H*2, W*2)
        latents_unpacked = self._unpatchify_latents(latents_unpacked)

        # 4) Decode with the VAE and normalize layout to HWC.
        decoded = self.vae.decode(latents_unpacked)
        return self._image_to_flat_hwc(self._to_numpy(decoded))

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

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        """Pack spatial latents (B, C, H, W) into sequence latents (B, H*W, C)."""
        batch_size, num_channels, height, width = map(int, latents.shape)
        latents = F.reshape(latents, (batch_size, num_channels, height * width))
        return F.permute(latents, (0, 2, 1))

    @staticmethod
    def _unpack_latents_with_ids(
        x: Tensor,
        x_ids: Tensor,
    ) -> Tensor:
        batch_size, seq_len, ch = map(int, x.shape)
        h_ids = x_ids[:, :, 1].cast(DType.int64)
        w_ids = x_ids[:, :, 2].cast(DType.int64)

        h = int(h_ids.max().item()) + 1
        w = int(w_ids.max().item()) + 1

        flat_ids = h_ids * w + w_ids

        x_list = []
        for b in range(batch_size):
            data_b = x[b]
            flat_ids_b = flat_ids[b]

            out = Tensor.zeros([h * w, ch], dtype=x.dtype, device=x.device)
            indices = F.reshape(flat_ids_b, [seq_len, 1]).cast(DType.int64)
            out = F.scatter_nd(out, data_b, indices)

            out = F.reshape(out, [h, w, ch])
            out = F.permute(out, (2, 0, 1))
            x_list.append(out)

        result = F.stack(x_list, axis=0)
        return result

    @staticmethod
    def _patchify_latents(latents: Tensor) -> Tensor:
        """Patchify latents by folding 2x2 spatial blocks into the channel dimension.

        Converts:
            (B, C, H, W) -> (B, C*4, H//2, W//2)
        """
        batch_size, num_channels_latents, height, width = map(
            int, latents.shape
        )

        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents, height // 2, 2, width // 2, 2),
        )
        latents = F.permute(latents, (0, 1, 3, 5, 2, 4))
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents * 4, height // 2, width // 2),
        )
        return latents

    @staticmethod
    def _unpatchify_latents(latents: Tensor) -> Tensor:
        """Inverse of `_patchify_latents`.

        Converts:
            (B, C*4, H//2, W//2) -> (B, C, H, W)
        """
        batch_size, num_channels_latents, height, width = map(
            int, latents.shape
        )
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, 2, 2, height, width),
        )
        latents = F.permute(latents, (0, 1, 4, 2, 5, 3))
        return F.reshape(
            latents,
            (batch_size, num_channels_latents // 4, height * 2, width * 2),
        )

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

    def _scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        sigmas: Tensor,
        step_index: int,
    ) -> Tensor:
        """Apply a single Euler update step in sigma space."""
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred
        return latents.cast(latents_dtype)

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
        prompt_embeds, text_ids = self._prepare_prompt_embeddings(
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
        latents: Tensor = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        latents = self._patchify_latents(latents)
        latents = self._pack_latents(latents)

        latent_image_ids = Tensor.from_dlpack(
            model_inputs.latent_image_ids.astype(np.int64)
        ).to(self.transformer.devices[0])

        guidance = Tensor.full(
            [latents.shape[0]],
            model_inputs.guidance_scale,
            device=self.transformer.devices[0],
            dtype=dtype,
        )

        sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(
            self.transformer.devices[0]
        )

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        timesteps_np = np.broadcast_to(
            timesteps[:, None], (num_timesteps, batch_size)
        )
        timesteps_batched = (
            Tensor.from_dlpack(timesteps_np)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )

        num_noise_tokens = int(latents.shape[1])

        # 3) Denoising loop.
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = timesteps_batched[i]

            if image_latents is not None:
                latent_model_input = F.concat([latents, image_latents], axis=1)
                latent_model_ids = F.concat(
                    [latent_image_ids, image_latent_ids], axis=1
                )
            else:
                latent_model_input = latents
                latent_model_ids = latent_image_ids

            noise_pred = self.transformer(
                latent_model_input,
                prompt_embeds,
                timestep,
                latent_model_ids,
                text_ids,
                guidance,
            )[0]
            noise_pred = Tensor.from_dlpack(noise_pred)
            noise_pred = noise_pred[:, :num_noise_tokens, :]

            latents = self._scheduler_step(latents, noise_pred, sigmas, i)

            if callback_queue is not None and output_type == "np":
                decoded = self._decode_latents(
                    latents,
                    latent_image_ids,
                    output_type="np",
                )
                # Ensure it's a numpy array for the queue
                if isinstance(decoded, Tensor):
                    decoded = np.array(decoded)
                callback_queue.put_nowait(decoded)

        # 4) Decode final outputs per batch element.
        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            latent_image_ids_b = latent_image_ids[b : b + 1]
            image_list.append(
                self._decode_latents(
                    latents_b, latent_image_ids_b, output_type=output_type
                )
            )

        return Flux2PipelineOutput(images=image_list)  # type: ignore[arg-type]
