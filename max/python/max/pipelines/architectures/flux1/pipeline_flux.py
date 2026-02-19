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

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Literal

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.tensor import Tensor
from tqdm import tqdm

from ..autoencoders import AutoencoderKLModel
from ..clip import ClipModel
from ..t5 import T5Model
from .model import Flux1TransformerModel


@dataclass(kw_only=True)
class FluxModelInputs(PixelModelInputs):
    """
    Flux-specific PixelModelInputs.

    Defaults:
    - width: 1024
    - height: 1024
    - true_cfg_scale: 1.0
    - num_inference_steps: 50
    - guidance_scale: 3.5
    - num_images_per_prompt: 1

    """

    width: int = 1024
    height: int = 1024
    true_cfg_scale: float = 1.0
    guidance_scale: float = 3.5
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1

    @property
    def do_true_cfg(self) -> bool:
        return self.negative_tokens is not None


@dataclass
class FluxPipelineOutput:
    """Output class for Flux image generation pipelines.

    Args:
        images (`np.ndarray` or `Tensor`)
            Numpy array or Max tensor of shape `(batch_size, height, width, num_channels)`.
            The denoised images of the diffusion pipeline.
    """

    images: np.ndarray | Tensor


class FluxPipeline(DiffusionPipeline):
    vae: AutoencoderKLModel
    text_encoder: ClipModel
    text_encoder_2: T5Model
    transformer: Flux1TransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": ClipModel,
        "text_encoder_2": T5Model,
        "transformer": Flux1TransformerModel,
    }

    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> FluxModelInputs:
        return FluxModelInputs.from_context(context)

    @staticmethod
    def _pack_latents(
        latents: Tensor,
    ) -> Tensor:
        batch_size, num_channels_latents, height, width = map(
            int, latents.shape
        )
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents, height // 2, 2, width // 2, 2),
        )
        latents = F.permute(latents, (0, 2, 4, 1, 3, 5))
        latents = F.reshape(
            latents,
            (
                batch_size,
                (height // 2) * (width // 2),
                num_channels_latents * 4,
            ),
        )
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor, height: int, width: int, vae_scale_factor: int
    ) -> Tensor:
        # TODO: should compile this function for speed up.
        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        latents = F.reshape(
            latents,
            (batch_size, height // 2, width // 2, ch_size // 4, 2, 2),
        )
        latents = F.permute(latents, (0, 3, 1, 4, 2, 5))

        latents = F.reshape(
            latents,
            (batch_size, ch_size // (2 * 2), height, width),
        )

        return latents

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        tokens_2: TokenBuffer | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tokens_2 = tokens_2 or tokens

        # unsqueeze
        if tokens.array.ndim == 1:
            tokens.array = np.expand_dims(tokens.array, axis=0)
        if tokens_2.array.ndim == 1:
            tokens_2.array = np.expand_dims(tokens_2.array, axis=0)

        text_input_ids = Tensor.constant(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        text_input_ids_2 = Tensor.constant(
            tokens_2.array,
            dtype=DType.int64,
            device=self.text_encoder_2.devices[0],
        )

        # t5 embeddings
        prompt_embeds = self.text_encoder_2(text_input_ids_2)

        # clip embeddings
        clip_embeddings = self.text_encoder(text_input_ids)
        pooled_prompt_embeds = clip_embeddings[1]

        text_ids = Tensor.zeros(
            (prompt_embeds.shape[1], 3),
            device=self.text_encoder_2.devices[0],
            dtype=prompt_embeds.dtype,
        )

        bs_embed = int(prompt_embeds.shape[0])
        seq_len = prompt_embeds.shape[1]

        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed * num_images_per_prompt, seq_len, -1)
        )

        pooled_prompt_embeds = F.tile(
            pooled_prompt_embeds, (1, num_images_per_prompt)
        )
        pooled_prompt_embeds = pooled_prompt_embeds.reshape(
            (bs_embed * num_images_per_prompt, -1)
        )
        dtype = prompt_embeds.dtype
        device = prompt_embeds.device

        return (
            prompt_embeds,
            pooled_prompt_embeds.to(device).cast(dtype),
            text_ids.to(device).cast(dtype),
        )

    def _decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        if output_type == "latent":
            return latents
        latents = Tensor.from_dlpack(latents)
        latents = self._unpack_latents(
            latents, height, width, self.vae_scale_factor
        )
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        return self._to_numpy(self.vae.decode(latents))

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    def _scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        sigmas: Tensor,
        step_index: int,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    def execute(  # type: ignore[override]
        self,
        model_inputs: FluxModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> FluxPipelineOutput:
        """Execute the pipeline."""
        # 1. Encode prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            self._prepare_prompt_embeddings(
                tokens=model_inputs.tokens,
                tokens_2=model_inputs.tokens_2,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )
        )

        negative_prompt_embeds: Tensor | None = None
        negative_pooled_prompt_embeds: Tensor | None = None
        negative_text_ids: Tensor | None = None
        if model_inputs.do_true_cfg:
            assert model_inputs.negative_tokens is not None
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self._prepare_prompt_embeddings(
                tokens=model_inputs.negative_tokens,
                tokens_2=model_inputs.negative_tokens_2,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

        # 2. Denoise
        dtype = prompt_embeds.dtype
        latents: Tensor = (
            Tensor.from_dlpack(model_inputs.latents)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )
        latents = self._pack_latents(latents)

        latent_image_ids: Tensor = (
            Tensor.from_dlpack(model_inputs.latent_image_ids)
            .to(self.transformer.devices[0])
            .cast(dtype)
        )

        if self.transformer.config.guidance_embeds:
            guidance = Tensor.full(
                [latents.shape[0]],
                model_inputs.guidance_scale,
                device=self.transformer.devices[0],
                dtype=dtype,
            )
        else:
            guidance = Tensor.zeros(
                [latents.shape[0]],
                device=self.transformer.devices[0],
                dtype=dtype,
            )

        sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(
            self.transformer.devices[0]
        )
        batch_size = int(prompt_embeds.shape[0])

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        timesteps_np = np.broadcast_to(
            timesteps[:, None], (num_timesteps, batch_size)
        )
        timesteps_batched = Tensor.from_dlpack(timesteps_np).to(
            self.transformer.devices[0]
        )
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = timesteps_batched[i]

            noise_pred = self.transformer(
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                timestep,
                latent_image_ids,
                text_ids,
                guidance,
            )[0]

            if model_inputs.do_true_cfg:
                assert negative_prompt_embeds is not None
                assert negative_pooled_prompt_embeds is not None
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latents,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    timestep,
                    latent_image_ids,
                    negative_text_ids,
                    guidance,
                )[0]

                noise_pred = neg_noise_pred + model_inputs.true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            # scheduler step
            latents = self._scheduler_step(latents, noise_pred, sigmas, i)

            if callback_queue is not None:
                image = self._decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        # 3. Decode
        outputs = self._decode_latents(
            latents,
            model_inputs.height,
            model_inputs.width,
            output_type=output_type,
        )

        return FluxPipelineOutput(images=outputs)
