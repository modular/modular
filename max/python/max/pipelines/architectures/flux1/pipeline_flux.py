# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from typing import Literal

import numpy as np
import PIL.Image
from max.driver import Buffer
from max.dtype import DType
from max.experimental import Tensor
from max.experimental import functional as F
from max.graph import DeviceRef
from max.interfaces import PixelGenerationOutput, TokenBuffer
from max.pipelines.lib.diffusion_schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from max.pipelines.lib.image_processor import (
    VaeImageProcessor,
)
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from tqdm import tqdm

from ..autoencoders import AutoencoderKLModel
from ..clip import ClipModel
from ..t5 import T5Model
from .model import Flux1Model


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
        images (`list[PIL.Image.Image]` or `np.ndarray` or `Tensor`)
            List of denoised PIL images of length `batch_size` or numpy array or Max tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Max tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image] | np.ndarray | Tensor


class FluxPipeline(DiffusionPipeline):
    config_name = "model_index.json"

    components = {
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "vae": AutoencoderKLModel,
        "text_encoder": ClipModel,
        "text_encoder_2": T5Model,
        "transformer": Flux1Model,
    }

    def init_remaining_components(self) -> None:
        image_processor_class = self.components.get(
            "image_processor", VaeImageProcessor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        image_processor = image_processor_class(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.image_processor = image_processor

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> Tensor:
        # TODO: should compile this function for speed up.
        batch_size, _, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        latents = F.reshape(
            latents,
            (batch_size.dim, height // 2, width // 2, channels.dim // 4, 2, 2),
        )
        latents = F.permute(latents, (0, 3, 1, 4, 2, 5))

        latents = F.reshape(
            latents, (batch_size.dim, channels.dim // (2 * 2), height, width)
        )

        return latents

    def _prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        tokens_2: TokenBuffer | None = None,
        num_images_per_prompt: int = 1,
        device: DeviceRef | None = None,
    ) -> tuple[Buffer, Buffer, Tensor]:
        text_input_ids = Tensor.constant(
            tokens.active, dtype=DType.int64, device=device
        )
        if tokens_2 is not None:
            text_input_ids_2 = Tensor.constant(
                tokens_2.active, dtype=DType.int64, device=device
            )
        else:
            text_input_ids_2 = text_input_ids

        # t5 embeddings
        prompt_embeds = self.text_encoder_2(text_input_ids_2)

        # clip embeddings
        clip_embeddings = self.text_encoder(text_input_ids)
        pooled_prompt_embeds = clip_embeddings[1]

        text_ids = Tensor.zeros(
            (prompt_embeds.shape[1], 3),
            device=device,
            dtype=prompt_embeds.dtype,
        )

        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, seq_len, -1)
        )

        pooled_prompt_embeds = F.tile(
            pooled_prompt_embeds, (1, num_images_per_prompt)
        )
        pooled_prompt_embeds = pooled_prompt_embeds.reshape(
            (bs_embed.dim * num_images_per_prompt, -1)
        )

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _denoise_latents(
        self,
        latents: Tensor,
        prompt_embeds: Tensor,
        pooled_prompt_embeds: Tensor,
        text_ids: Tensor,
        guidance: Tensor,
        timestep: Tensor,
        latent_image_ids: Tensor,
    ) -> Tensor:
        noise_pred = self.transformer(
            latents,
            prompt_embeds,
            pooled_prompt_embeds,
            timestep,
            latent_image_ids,
            text_ids,
            guidance,
        )[0]
        return noise_pred

    def _decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor:
        if output_type == "latent":
            image = latents
        else:
            latents = Tensor.from_dlpack(latents)
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents)

            image = self.image_processor.postprocess(
                image, output_type=output_type
            )
        return image

    def execute(
        self,
        inputs: FluxModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> PixelGenerationOutput:
        """Execute the pipeline."""

        execute_device = self._execution_device()

        # 1. Encode prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            self._prepare_prompt_embeddings(
                tokens=inputs.tokens,
                tokens_2=inputs.tokens_2,
                num_images_per_prompt=inputs.num_images_per_prompt,
                device=execute_device,
            )
        )

        if inputs.do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self._prepare_prompt_embeddings(
                tokens=inputs.negative_tokens,
                tokens_2=inputs.negative_tokens_2,
                num_images_per_prompt=inputs.num_images_per_prompt,
                device=execute_device,
            )

        # 2. Denoise
        latents: Buffer = Buffer.from_numpy(inputs.latents).to(execute_device)
        latent_image_ids: Buffer = Buffer.from_numpy(
            inputs.latent_image_ids
        ).to(execute_device)
        timesteps: np.ndarray = inputs.timesteps
        num_timesteps = timesteps.shape[0]

        batch_size = latents.shape[0]

        if self.transformer.config.guidance_embeds:
            guidance = Tensor.full(
                [latents.shape[0].dim],
                inputs.guidance_scale,
                device=execute_device,
                dtype=prompt_embeds.dtype,
            )
        else:
            guidance = Tensor.zeros(
                [latents.shape[0].dim],
                device=execute_device,
                dtype=prompt_embeds.dtype,
            )

        self.scheduler.set_begin_index(0)
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            if self._interrupt:
                break
            self._current_timestep = i

            t = timesteps[i]

            timestep = np.full((batch_size,), t, dtype=np.float32)
            timestep = Tensor.from_dlpack(timestep).to(prompt_embeds.device)

            noise_pred = self._denoise_latents(
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                guidance,
                timestep,
                latent_image_ids,
            )

            if inputs.do_true_cfg:
                neg_noise_pred = self._denoise_latents(
                    latents,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                    guidance,
                    timestep,
                    latent_image_ids,
                )

                noise_pred = neg_noise_pred + inputs.true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_queue is not None:
                image = self._decode_latents(
                    latents,
                    inputs.height,
                    inputs.width,
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        # 3. Decode
        outputs = self._decode_latents(
            latents, inputs.height, inputs.width, output_type=output_type
        )

        return FluxPipelineOutput(images=outputs)
