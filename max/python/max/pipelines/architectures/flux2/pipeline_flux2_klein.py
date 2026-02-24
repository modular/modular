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

import logging
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
from max import functional as F
from max.dtype import DType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import PixelModelInputs
from max.tensor import Tensor
from PIL import Image
from tqdm import tqdm

from ..qwen3.text_encoder import Qwen3TextEncoderModel
from .pipeline_flux2 import Flux2Pipeline

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class Flux2KleinModelInputs(PixelModelInputs):
    """Flux2 Klein-specific model inputs."""

    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    mask: np.ndarray | None = None
    input_image: Image.Image | None = None

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.negative_tokens is not None and self.guidance_scale > 1.0


@dataclass
class Flux2KleinPipelineOutput:
    """Container for Flux2 Klein pipeline results."""

    images: np.ndarray | Tensor


class Flux2KleinPipeline(Flux2Pipeline):
    """Flux2 Klein diffusion pipeline with Qwen3 text encoder."""

    components = {
        "vae": Flux2Pipeline.components["vae"],
        "text_encoder": Qwen3TextEncoderModel,
        "transformer": Flux2Pipeline.components["transformer"],
    }

    def prepare_inputs(self, context: PixelContext) -> Flux2KleinModelInputs:  # type: ignore[override]
        if context.input_image is not None and isinstance(
            context.input_image, np.ndarray
        ):
            context.input_image = Image.fromarray(
                context.input_image.astype(np.uint8)
            )
        return Flux2KleinModelInputs.from_context(context)

    def prepare_prompt_embeddings(
        self,
        tokens: TokenBuffer,
        num_images_per_prompt: int = 1,
        attention_mask: np.ndarray | None = None,
        hidden_states_layers: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        layers = hidden_states_layers or [9, 18, 27]
        token_ids = tokens.array
        if token_ids.ndim == 2:
            token_ids = token_ids[0]
        max_seq = int(token_ids.shape[-1])

        # NOTE: Qwen3TextEncoderModel currently does not accept attention_mask as an input.
        # Keep the full padded sequence (max_seq=512) to avoid quality regression from
        # manual trimming/zero-padding approximations.
        _ = attention_mask

        text_input_ids = Tensor.constant(
            token_ids, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        hidden_states_all = self.text_encoder(text_input_ids)
        hidden_states_selected = []
        for i in layers:
            hs = hidden_states_all[i - 1]
            if hs.rank == 3:
                hs = hs[0]

            seq_len = int(hs.shape[0])
            hidden_dim = int(hs.shape[1])
            if seq_len < max_seq:
                hs = F.concat(
                    [
                        hs,
                        Tensor.zeros(
                            [max_seq - seq_len, hidden_dim],
                            dtype=hs.dtype,
                            device=hs.device,
                        ),
                    ],
                    axis=0,
                )
            elif seq_len > max_seq:
                hs = hs[:max_seq]
            hidden_states_selected.append(hs)

        prompt_embeds = self._prepare_prompt_embeddings(*hidden_states_selected)
        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])

        prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(
            (batch_size * num_images_per_prompt, seq_len, -1)
        )

        text_ids = self._prepare_text_ids(
            batch_size=batch_size * num_images_per_prompt,
            seq_len=seq_len,
            device=self.text_encoder.devices[0],
        )
        return prompt_embeds, text_ids

    def _prepare_prompt_embeddings(self, *hidden_states: Tensor) -> Tensor:
        # [L, S, D] -> [1, S, L, D] -> [1, S, L*D]
        stacked = F.stack(hidden_states, axis=0)
        stacked = F.unsqueeze(stacked, axis=0)
        stacked = F.permute(stacked, [0, 2, 1, 3])
        batch_size = stacked.shape[0]
        seq_len = stacked.shape[1]
        num_layers = stacked.shape[2]
        hidden_dim = stacked.shape[3]
        prompt_embeds = F.reshape(
            stacked, [batch_size, seq_len, num_layers * hidden_dim]
        )
        return prompt_embeds

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2KleinModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2KleinPipelineOutput:
        prompt_embeds, text_ids = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            attention_mask=model_inputs.mask,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
        )

        diff_cfg = self.pipeline_config.model.diffusers_config or {}
        is_distilled = bool(diff_cfg.get("is_distilled", False))
        if model_inputs.guidance_scale > 1.0 and is_distilled:
            logger.warning(
                "Guidance scale %s is ignored for distilled Klein models.",
                model_inputs.guidance_scale,
            )

        negative_prompt_embeds: Tensor | None = None
        negative_text_ids: Tensor | None = None
        do_cfg = model_inputs.do_classifier_free_guidance and not is_distilled
        if do_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds, negative_text_ids = (
                self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens,
                    attention_mask=None,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
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

        guidance = Tensor.zeros(
            [latents.shape[0]],
            device=self.transformer.devices[0],
            dtype=dtype,
        )

        sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(
            self.transformer.devices[0]
        )
        all_timesteps, dts = self.prepare_scheduler(sigmas)
        num_timesteps = int(all_timesteps.shape[0])

        num_noise_tokens = int(latents.shape[1])
        is_img2img = image_latents is not None
        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = all_timesteps[i : i + 1]
            dt = dts[i : i + 1]

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

            if do_cfg:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latent_model_input,
                    negative_prompt_embeds,
                    timestep,
                    latent_model_ids,
                    negative_text_ids,
                    guidance,
                )[0]
                neg_noise_pred = Tensor.from_dlpack(neg_noise_pred)
                neg_noise_pred = neg_noise_pred[:, :num_noise_tokens, :]
                noise_pred = neg_noise_pred + model_inputs.guidance_scale * (
                    noise_pred - neg_noise_pred
                )

            latents = self.scheduler_step(
                latents, noise_pred, dt, num_noise_tokens
            )

            if callback_queue is not None and output_type == "np":
                decoded = self.decode_latents(
                    latents,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type="np",
                    is_img2img=is_img2img,
                )
                if isinstance(decoded, Tensor):
                    decoded = np.array(decoded)
                callback_queue.put_nowait(decoded)

        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            latent_image_ids_b = latent_image_ids[b : b + 1]
            image_list.append(
                self.decode_latents(
                    latents_b,
                    latent_image_ids_b,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                    is_img2img=is_img2img,
                )
            )

        return Flux2KleinPipelineOutput(images=image_list)  # type: ignore[arg-type]
