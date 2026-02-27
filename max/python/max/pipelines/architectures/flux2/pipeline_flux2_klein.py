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
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from max.experimental import functional as F
from max.dtype import DType
from max.interfaces import TokenBuffer
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import PixelModelInputs
from max.experimental.tensor import Tensor
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
    mask: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
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
    prompt_embedding_hidden_states_layers: tuple[int, ...] = (9, 18, 27)

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
        mask: npt.NDArray[np.bool_],
        num_images_per_prompt: int = 1,
    ) -> tuple[Tensor, Tensor]:
        token_ids = np.asarray(tokens.array, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError(
                f"Flux2Klein expects 1D tokens, got shape {token_ids.shape}."
            )
        padded_seq_len = int(token_ids.shape[0])
        if mask.shape[0] != padded_seq_len:
            raise ValueError(
                "Prompt mask length must match token length. "
                f"Got mask={mask.shape[0]}, tokens={padded_seq_len}."
            )

        # NOTE: Qwen3TextEncoderModel currently does not support attention_mask input.
        # Keep the full padded sequence to avoid divergence from diffusers prompt geometry.
        target_seq_len = int(mask.shape[0])

        text_input_ids = Tensor.constant(
            token_ids,
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        hidden_states_all = self.text_encoder(text_input_ids)
        hidden_states_selected = []
        for i in self.prompt_embedding_hidden_states_layers:
            hs = hidden_states_all[i - 1]
            if hs.rank == 3:
                hs = hs[0]

            seq_len = int(hs.shape[0])
            hidden_dim = int(hs.shape[1])
            if seq_len < target_seq_len:
                hs = F.concat(
                    [
                        hs,
                        Tensor.zeros(
                            [target_seq_len - seq_len, hidden_dim],
                            dtype=hs.dtype,
                            device=hs.device,
                        ),
                    ],
                    axis=0,
                )
            elif seq_len > target_seq_len:
                hs = hs[:target_seq_len]
            hidden_states_selected.append(hs)

        prompt_embeds = self._prepare_prompt_embeddings(*hidden_states_selected)
        batch_size = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])

        if num_images_per_prompt != 1:
            prompt_embeds = F.tile(prompt_embeds, (1, num_images_per_prompt, 1))
            prompt_embeds = prompt_embeds.reshape(
                (batch_size * num_images_per_prompt, seq_len, -1)
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

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2KleinModelInputs,
        callback_queue: Queue[np.ndarray] | None = None,
        output_type: Literal["np", "latent"] = "np",
    ) -> Flux2KleinPipelineOutput:
        prompt_embeds, text_ids = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            mask=model_inputs.mask,
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
            neg_token_ids = np.asarray(model_inputs.negative_tokens.array)
            neg_seq_len = (
                int(neg_token_ids.shape[0])
                if neg_token_ids.ndim == 1
                else int(neg_token_ids.shape[-1])
            )
            negative_mask = np.ones(neg_seq_len, dtype=np.bool_)
            negative_prompt_embeds, negative_text_ids = (
                self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens,
                    mask=negative_mask,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
            )
        elif do_cfg:
            logger.warning(
                "CFG requested but negative prompt tokens are missing; "
                "running without CFG."
            )
            do_cfg = False

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

        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )

        device = self.transformer.devices[0]
        guidance_key = f"zero_{batch_size}"
        if guidance_key in self._cached_guidance:
            guidance = self._cached_guidance[guidance_key]
        else:
            guidance = Tensor.zeros(
                [latents.shape[0]],
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

        timesteps_seq: Any = all_timesteps
        dts_seq: Any = all_dts
        if hasattr(timesteps_seq, "driver_tensor"):
            timesteps_seq = timesteps_seq.driver_tensor
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        num_noise_tokens = int(latents.shape[1])
        is_img2img = image_latents is not None
        for i in tqdm(range(num_inference_steps), desc="Denoising"):
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

            noise_pred = self.transformer(
                latents_concat,
                prompt_embeds,
                timestep,
                latent_image_ids_concat,
                text_ids,
                guidance,
            )[0]
            noise_pred = Tensor.from_dlpack(noise_pred)

            if do_cfg:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latents_concat,
                    negative_prompt_embeds,
                    timestep,
                    latent_image_ids_concat,
                    negative_text_ids,
                    guidance,
                )[0]
                neg_noise_pred = Tensor.from_dlpack(neg_noise_pred)
                noise_pred = neg_noise_pred + model_inputs.guidance_scale * (
                    noise_pred - neg_noise_pred
                )

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
                            latent_image_ids,
                            model_inputs.height,
                            model_inputs.width,
                            output_type=output_type,
                        ),
                    )
                )

        image_list = []
        for b in range(batch_size):
            latents_b = latents[b : b + 1]
            image_list.append(
                self.decode_latents(
                    latents_b,
                    latent_image_ids,
                    model_inputs.height,
                    model_inputs.width,
                    output_type=output_type,
                )
            )

        return Flux2KleinPipelineOutput(images=image_list)  # type: ignore[arg-type]
