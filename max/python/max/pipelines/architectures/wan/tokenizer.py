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
"""Wan-specific pixel generation tokenizer."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.interfaces.request import OpenResponsesRequest
from max.pipelines.lib.pixel_tokenizer import PixelGenerationTokenizer

from .context import WanContext

logger = logging.getLogger("max.pipelines")


class WanTokenizer(PixelGenerationTokenizer):
    """Wan-specific tokenizer that produces WanContext with video/MoE fields."""

    def _select_wan_flow_shift(self, height: int, width: int) -> float:
        scheduler_cfg = (
            self.diffusers_config.get("components", {})
            .get("scheduler", {})
            .get("config_dict", {})
        )
        # Use explicit flow_shift from scheduler config if set (user override).
        cfg_shift = scheduler_cfg.get("flow_shift")
        if cfg_shift is not None and float(cfg_shift) != 1.0:
            return float(cfg_shift)
        # Default: interpolate based on pixel count.
        # 480p (480*832 = 399 360) → 3.0, 720p (720*1280 = 921 600) → 5.0
        pixels = height * width
        lo_px, hi_px = 399_360, 921_600
        lo_shift, hi_shift = 3.0, 5.0
        t = max(0.0, min(1.0, (pixels - lo_px) / (hi_px - lo_px)))
        return lo_shift + t * (hi_shift - lo_shift)

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> WanContext:
        base = await super().new_context(request, input_image=input_image)

        video_options = request.body.provider_options.video

        num_frames: int | None = (
            video_options.num_frames if video_options else None
        )
        guidance_scale_2: float | None = (
            video_options.guidance_scale_2 if video_options else None
        )

        height = base.height
        width = base.width
        timesteps: npt.NDArray[np.float32] = base.timesteps
        sigmas: npt.NDArray[np.float32] = base.sigmas

        if getattr(self._scheduler, "use_flow_sigmas", False):
            self._scheduler.flow_shift = self._select_wan_flow_shift(
                height, width
            )
            latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
            latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
            image_seq_len = (latent_height // 2) * (latent_width // 2)
            timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
                image_seq_len, base.num_inference_steps
            )

        boundary_timestep: float | None = None
        boundary_ratio = self.diffusers_config.get("boundary_ratio")
        if boundary_ratio is not None:
            boundary_timestep = float(boundary_ratio) * float(
                getattr(self._scheduler, "num_train_timesteps", 1000)
            )

        step_coefficients: npt.NDArray[np.float32] | None = None
        if hasattr(self._scheduler, "build_step_coefficients"):
            step_coefficients = self._scheduler.build_step_coefficients()

        latents = base.latents
        if num_frames is not None:
            vae_scale_factor_temporal = 4
            latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
            latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
            latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
            shape_5d = (
                base.num_images_per_prompt,
                self._num_channels_latents,
                latent_frames,
                latent_height,
                latent_width,
            )
            latents = self._randn_tensor(shape_5d, request.body.seed)

        return WanContext(
            request_id=base.request_id,
            model_name=base.model_name,
            tokens=base.tokens,
            mask=base.mask,
            tokens_2=base.tokens_2,
            negative_tokens=base.negative_tokens,
            negative_mask=base.negative_mask,
            negative_tokens_2=base.negative_tokens_2,
            explicit_negative_prompt=base.explicit_negative_prompt,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=latents,
            latent_image_ids=base.latent_image_ids,
            height=base.height,
            width=base.width,
            num_frames=num_frames,
            guidance_scale=base.guidance_scale,
            true_cfg_scale=base.true_cfg_scale,
            guidance_scale_2=guidance_scale_2,
            cfg_normalization=base.cfg_normalization,
            cfg_truncation=base.cfg_truncation,
            num_inference_steps=base.num_inference_steps,
            num_warmup_steps=base.num_warmup_steps,
            strength=base.strength,
            boundary_timestep=boundary_timestep,
            step_coefficients=step_coefficients,
            num_images_per_prompt=base.num_images_per_prompt,
            input_image=base.input_image,
            output_format=base.output_format,
            residual_threshold=base.residual_threshold,
            status=base.status,
        )
