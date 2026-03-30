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
from typing import Any, cast

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    DiffusionPipelineOutput,
    max_compile,
)
from max.profiler import Tracer, traced

from ..qwen3.text_encoder import Qwen3TextEncoderKleinModel
from .pipeline_flux2 import Flux2ModelInputs, Flux2Pipeline

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class Flux2KleinModelInputs(Flux2ModelInputs):
    """Flux2 Klein-specific model inputs extending the base Flux2 inputs."""

    negative_tokens: Buffer | None = None
    """Negative prompt token IDs on device (for classifier-free guidance)."""

    attention_mask: np.ndarray | None = None
    """Tokenizer-generated mask for the padded positive prompt sequence."""

    negative_attention_mask: np.ndarray | None = None
    """Tokenizer-generated mask for the padded negative prompt sequence."""

    guidance_scale: float = 4.0
    """Guidance scale for classifier-free guidance."""

    is_distilled: bool = False
    """Whether the model is distilled (disables CFG)."""

    do_classifier_free_guidance: bool = False
    """Whether to run the negative-prompt branch for CFG."""


class Flux2KleinPipeline(Flux2Pipeline):
    """Flux2 Klein diffusion pipeline with a bridged Qwen3 text encoder."""

    components = {
        "vae": Flux2Pipeline.components["vae"],
        "text_encoder": Qwen3TextEncoderKleinModel,
        "transformer": Flux2Pipeline.components["transformer"],
    }

    @traced(message="Flux2KleinPipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        """Initialize derived attributes, including the compiled CFG combine."""
        super().init_remaining_components()
        self.build_cfg_combine()

    @traced(message="Flux2KleinPipeline.build_cfg_combine")
    def build_cfg_combine(self) -> None:
        """Compile the CFG combine formula with symbolic shapes."""
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.cfg_combine = cast(
            Any,
            max_compile(
                self.cfg_combine,
                input_types=[
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "channels"],
                        device=device,
                    ),
                    TensorType(
                        dtype,
                        shape=["batch", "seq", "channels"],
                        device=device,
                    ),
                    TensorType(DType.float32, shape=[], device=device),
                ],
            ),
        )

    def cfg_combine(
        self,
        noise_pred: TensorValue,
        neg_noise_pred: TensorValue,
        guidance_scale: TensorValue,
    ) -> TensorValue:
        """Apply CFG formula in f32 before casting back to the model dtype."""
        input_dtype = noise_pred.dtype
        noise_pred_f32 = ops.cast(noise_pred, DType.float32)
        neg_noise_pred_f32 = ops.cast(neg_noise_pred, DType.float32)
        diff = noise_pred_f32 - neg_noise_pred_f32
        scaled = guidance_scale * diff
        result = neg_noise_pred_f32 + scaled
        return ops.cast(result, input_dtype)

    @traced(message="Flux2KleinPipeline.prepare_inputs")
    def prepare_inputs(self, context: PixelContext) -> Flux2KleinModelInputs:  # type: ignore[override]
        """Convert a PixelContext into Flux2KleinModelInputs."""
        base_inputs = super().prepare_inputs(context)

        negative_tokens = None
        if context.negative_tokens is not None:
            negative_tokens = Buffer.from_dlpack(
                context.negative_tokens.array
            ).to(self.text_encoder.devices[0])

        diff_cfg = self.pipeline_config.model.diffusers_config or {}
        is_distilled = bool(diff_cfg.get("is_distilled", False))
        do_classifier_free_guidance = (
            negative_tokens is not None
            and context.guidance_scale > 1.0
            and not is_distilled
        )

        if context.guidance_scale > 1.0 and is_distilled:
            logger.warning(
                "Guidance scale %s is ignored for distilled Klein models.",
                context.guidance_scale,
            )

        return Flux2KleinModelInputs(
            tokens=base_inputs.tokens,
            latents=base_inputs.latents,
            latent_image_ids=base_inputs.latent_image_ids,
            sigmas=base_inputs.sigmas,
            guidance=base_inputs.guidance,
            latent_h=base_inputs.latent_h,
            latent_w=base_inputs.latent_w,
            image_seq_len=base_inputs.image_seq_len,
            h_carrier=base_inputs.h_carrier,
            w_carrier=base_inputs.w_carrier,
            height=base_inputs.height,
            width=base_inputs.width,
            num_inference_steps=base_inputs.num_inference_steps,
            num_images_per_prompt=base_inputs.num_images_per_prompt,
            input_image=base_inputs.input_image,
            negative_tokens=negative_tokens,
            attention_mask=context.mask,
            negative_attention_mask=context.negative_mask,
            guidance_scale=context.guidance_scale,
            is_distilled=is_distilled,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

    @traced(message="Flux2KleinPipeline.prepare_prompt_embeddings")
    def prepare_prompt_embeddings(
        self,
        tokens: Buffer,
        num_images_per_prompt: int = 1,
        attention_mask: np.ndarray | None = None,
    ) -> tuple[Buffer, Buffer]:
        """Create prompt embeddings and text IDs using the v2 Qwen3 encoder."""
        seq_len = int(tokens.shape[0])
        batch_size = 1

        with Tracer("text_encoder"):
            prompt_embeds = cast(
                Buffer,
                self.text_encoder(
                    tokens,
                    attention_mask=attention_mask,
                ),
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

    @traced(message="Flux2KleinPipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux2KleinModelInputs,
    ) -> DiffusionPipelineOutput:
        """Run the Flux2 Klein denoising loop with optional CFG."""
        prompt_embeds, text_ids = self.prepare_prompt_embeddings(
            tokens=model_inputs.tokens,
            num_images_per_prompt=model_inputs.num_images_per_prompt,
            attention_mask=model_inputs.attention_mask,
        )
        batch_size = int(prompt_embeds.shape[0])

        negative_prompt_embeds = None
        negative_text_ids = None
        do_cfg = model_inputs.do_classifier_free_guidance
        if do_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds, negative_text_ids = (
                self.prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                    attention_mask=model_inputs.negative_attention_mask,
                )
            )
        elif (
            model_inputs.negative_tokens is None
            and not model_inputs.is_distilled
            and model_inputs.guidance_scale > 1.0
        ):
            logger.warning(
                "CFG requested but negative prompt tokens are missing; "
                "running without CFG."
            )
            do_cfg = False

        image_latents = None
        image_latent_ids = None
        if model_inputs.input_image is not None:
            with Tracer("prepare_image_input"):
                image_buffer = self._numpy_image_to_buffer(
                    model_inputs.input_image
                )
                image_latents, image_latent_ids = self.prepare_image_latents(
                    images=[image_buffer],
                    batch_size=batch_size,
                    device=self.vae.devices[0],
                    dtype=self.vae.config.dtype,
                )

        with Tracer("preprocess_latents"):
            latents = self.preprocess_latents(model_inputs.latents)
            latent_image_ids = model_inputs.latent_image_ids

        with Tracer("prepare_scheduler"):
            all_timesteps, all_dts = self.prepare_scheduler(model_inputs.sigmas)
            guidance = model_inputs.guidance

            timesteps_seq: Any = all_timesteps
            dts_seq: Any = all_dts
            if hasattr(timesteps_seq, "driver_tensor"):
                timesteps_seq = timesteps_seq.driver_tensor
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

        with Tracer("prepare_cfg_scale"):
            guidance_scale_tensor = None
            if do_cfg:
                guidance_scale_tensor = Buffer.from_numpy(
                    np.array(model_inputs.guidance_scale, dtype=np.float32)
                ).to(self.transformer.devices[0])

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

                    if do_cfg:
                        assert negative_prompt_embeds is not None
                        assert negative_text_ids is not None
                        with Tracer("transformer_negative"):
                            neg_noise_pred = self.transformer(
                                latents_concat,
                                negative_prompt_embeds,
                                timestep,
                                latent_image_ids_concat,
                                negative_text_ids,
                                guidance,
                            )[0]
                        with Tracer("cfg_combine"):
                            assert guidance_scale_tensor is not None
                            noise_pred = self.cfg_combine(
                                noise_pred,
                                neg_noise_pred,
                                guidance_scale_tensor,
                            )

                    with Tracer("scheduler_step"):
                        latents = self._scheduler_step(latents, noise_pred, dt)

        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                model_inputs.h_carrier,
                model_inputs.w_carrier,
            )

        return DiffusionPipelineOutput(images=images)
