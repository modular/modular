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

"""Wan-Animate pipeline for motion transfer and character replacement.

Extends WanI2VPipeline with:
- Pose conditioning via Conv3d injection
- Face motion encoding (StyleGAN2 bridge) + face encoder (MAX Graph)
- CLIP image conditioning (dual-path cross-attention)
- Multi-segment processing with temporal overlap
- Replace mode (background preservation via mask)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.profiler import Tracer, traced
from tqdm.auto import tqdm

from ..autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWanModel,
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
from ..clip import ClipModel
from ..umt5 import UMT5Model
from .model_animate import WanAnimateTransformerModel
from .pipeline_wan import (
    WanModelInputs,
    WanPipelineOutput,
    WanRuntimeCache,
    WanUniPCState,
)
from .pipeline_wan_i2v import WanI2VPipeline

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WanAnimateModelInputs(WanModelInputs):
    """Extended model inputs for Wan-Animate pipeline."""

    # Tokenizer-prepared pose video: [T, 3, H, W] float32 in [-1, 1]
    pose_video_np: npt.NDArray[np.float32] | None = None
    # Tokenizer-prepared face pixels: [T, 3, 512, 512] float32 in [-1, 1]
    face_pixels_np: npt.NDArray[np.float32] | None = None
    # Input image (reference character)
    input_image: npt.NDArray[np.uint8] | None = None
    # Animate mode: "animate" or "replace"
    animate_mode: str = "animate"
    # Replace mode inputs
    bg_video_np: npt.NDArray[np.float32] | None = None
    mask_video_np: npt.NDArray[np.float32] | None = None
    # Segment metadata derived by the tokenizer
    num_pose_frames: int | None = None
    num_segments: int | None = None
    effective_seg_len: int | None = None
    # Segment parameters
    segment_frame_length: int = 77
    prev_segment_conditioning_frames: int = 1
    # Random seed for deterministic multi-segment noise generation
    seed: int | None = None


class WanAnimatePipeline(WanI2VPipeline):
    """Wan-Animate pipeline — motion transfer with pose/face conditioning.

    Extends WanI2VPipeline with CLIP image encoding, motion/face encoding,
    and multi-segment processing.
    """

    transformer: WanAnimateTransformerModel  # type: ignore[assignment]

    components = {
        "vae": AutoencoderKLWanModel,
        "text_encoder": UMT5Model,
        "transformer": WanAnimateTransformerModel,
        "image_encoder": ClipModel,
    }

    def _get_component_config_dict(
        self, components_config: dict[str, Any], name: str
    ) -> dict[str, Any]:
        config_dict = dict(
            super()._get_component_config_dict(components_config, name)
        )
        # NOTE: For Wan-Animate pipeline, Clip vision encoder
        # returns the penultimate layer, not the last layer.
        if name == "image_encoder":
            config_dict["return_penultimate"] = True
        return config_dict

    def init_remaining_components(self) -> None:
        self.transformer_2 = None
        self._setup_vae_config()

        self.build_guidance()
        self.build_unipc_step()
        self.build_cast_f32_to_model_dtype()
        self.build_denorm(skip_first=True)
        self.build_standardize_latents()
        self.build_uncond_face_embedding()
        self.build_i2v_concat()
        self.cache: WanRuntimeCache = WanRuntimeCache()

    def build_standardize_latents(self) -> None:
        """Compile VAE latent standardization in model dtype."""
        device = self.transformer.devices[0]
        latent_dtype = self.vae.config.dtype
        z_dim = int(self.vae.config.z_dim)
        input_types = [
            TensorType(
                latent_dtype,
                ["batch", z_dim, "frames", "height", "width"],
                device=device,
            ),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
        ]
        with Graph(
            "wan_animate_standardize_latents", input_types=input_types
        ) as g:
            latents, mean, std = (value.tensor for value in g.inputs)
            standardized = ops.cast(
                (ops.cast(latents, DType.float32) - mean) / std,
                latent_dtype,
            )
            g.output(standardized)
        self.__dict__["_standardize_latents_model"] = self.session.load(g)

    def build_uncond_face_embedding(self) -> None:
        """Compile unconditional face embedding generation."""
        device = self.transformer.devices[0]
        face_dtype = self.transformer.config.dtype
        with Graph(
            "wan_animate_uncond_face_embedding",
            input_types=[
                TensorType(
                    face_dtype,
                    [1, "frames", "tokens", "channels"],
                    device=device,
                )
            ],
        ) as g:
            face_emb = g.inputs[0].tensor
            uncond = ops.cast(
                ops.cast(face_emb, DType.float32) * 0.0 - 1.0,
                face_dtype,
            )
            g.output(uncond)
        self.__dict__["_uncond_face_embedding_model"] = self.session.load(g)

    @traced(message="WanAnimatePipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanAnimateModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        device = self.transformer.devices[0]
        height = model_inputs.height
        width = model_inputs.width

        with Tracer("prepare_prompt_embeddings"):
            (
                prompt_embeds,
                negative_prompt_embeds,
                do_cfg,
            ) = self.prepare_prompt_embeddings(model_inputs)

        with Tracer("prepare_image_embeddings"):
            clip_features = self.image_encoder.encode(model_inputs.input_image)

        logger.info(
            "Animate: %d pose frames, %d segments (segment_len=%d, prev_cond=%d), animate_mode=%s",
            model_inputs.num_pose_frames,
            model_inputs.num_segments,
            model_inputs.segment_frame_length,
            model_inputs.prev_segment_conditioning_frames,
            model_inputs.animate_mode,
        )

        vae_t = self.vae_scale_factor_temporal
        height_latent = height // self.vae_scale_factor_spatial
        width_latent = width // self.vae_scale_factor_spatial
        z_dim = self.vae.config.z_dim
        y_ref = self.prepare_reference_image_condition(
            model_inputs.input_image,
            height,
            width,
            height_latent,
            width_latent,
            vae_t,
            device,
        )

        all_out_frames: list[npt.NDArray[np.float32]] = []
        prev_segment_cond_video: npt.NDArray[np.float32] | None = None

        for seg_idx in range(model_inputs.num_segments):
            seg_start = seg_idx * model_inputs.effective_seg_len
            seg_end = seg_start + model_inputs.segment_frame_length
            seg_pose_np = model_inputs.pose_video_np[seg_start:seg_end]
            seg_face_np = model_inputs.face_pixels_np[seg_start:seg_end]
            num_seg_frames = int(seg_pose_np.shape[0])

            logger.info(
                "Segment %d/%d: frames %d-%d (%d frames)",
                seg_idx + 1,
                model_inputs.num_segments,
                seg_start,
                seg_start + num_seg_frames - 1,
                num_seg_frames,
            )

            with Tracer(f"segment_{seg_idx}:vae_encode_pose"):
                pose_latents = self.encode_pose_segment(
                    seg_pose_np,
                    device,
                )

            with Tracer(f"segment_{seg_idx}:encode_face"):
                face_emb = self.encode_face_segment(
                    seg_face_np,
                    device,
                )

            with Tracer(f"segment_{seg_idx}:prepare_condition"):
                condition, total_latent_t = self.prepare_segment_condition(
                    seg_idx=seg_idx,
                    animate_mode=model_inputs.animate_mode,
                    bg_video_np=model_inputs.bg_video_np,
                    mask_video_np=model_inputs.mask_video_np,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    prev_segment_cond_video=prev_segment_cond_video,
                    num_seg_frames=num_seg_frames,
                    prev_cond_frames=model_inputs.prev_segment_conditioning_frames,
                    pose_latents=pose_latents,
                    y_ref=y_ref,
                    height=height,
                    width=width,
                    height_latent=height_latent,
                    width_latent=width_latent,
                    vae_t=vae_t,
                    device=device,
                )

            with Tracer(f"segment_{seg_idx}:prepare_latents"):
                noise_shape = (
                    1,
                    z_dim,
                    total_latent_t,
                    height_latent,
                    width_latent,
                )
                latents = self.prepare_segment_latents(
                    model_inputs,
                    seg_idx,
                    noise_shape,
                    device,
                )

            with Tracer(f"segment_{seg_idx}:prepare_scheduler"):
                batched_timesteps, coeff_buffers, guidance_scale_high = (
                    self.prepare_scheduler_state(
                        latents,
                        model_inputs,
                        prompt_embeds,
                        do_cfg,
                        device,
                    )
                )

            with Tracer(f"segment_{seg_idx}:denoising"):
                latents = self.run_animate_denoising(
                    latents=latents,
                    condition=condition,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    clip_features=clip_features,
                    pose_latents=pose_latents,
                    face_emb=face_emb,
                    total_latent_t=total_latent_t,
                    height_latent=height_latent,
                    width_latent=width_latent,
                    do_cfg=do_cfg,
                    batched_timesteps=batched_timesteps,
                    coeff_buffers=coeff_buffers,
                    device=device,
                    guidance_scale=guidance_scale_high,
                )

            with Tracer(f"segment_{seg_idx}:vae_decode"):
                decoded = self.decode_segment_latents(latents)

            if (
                seg_idx > 0
                and model_inputs.prev_segment_conditioning_frames > 0
            ):
                decoded = decoded[
                    :, :, model_inputs.prev_segment_conditioning_frames :, :, :
                ]

            if model_inputs.num_segments > 1:
                prev_segment_cond_video = decoded[
                    :, :, -model_inputs.prev_cond_frames :, :, :
                ]

            all_out_frames.append(decoded)

        video = np.concatenate(all_out_frames, axis=2)
        video = video[:, :, : model_inputs.num_pose_frames, :, :]

        return WanPipelineOutput(images=video)

    def _normalize_reference_image(
        self,
        ref_image: Any,
        height: int,
        width: int,
    ) -> npt.NDArray[np.float32]:
        """Convert the reference image to CHW float32 in [-1, 1]."""
        arr = np.asarray(ref_image)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected reference image with 3 dims, got shape {arr.shape}"
            )
        if arr.shape[-1] >= 3:
            rgb = arr[:, :, :3]
        elif arr.shape[0] >= 3:
            rgb = arr[:3].transpose(1, 2, 0)
        else:
            raise ValueError(
                f"Expected reference image with at least 3 channels, got shape {arr.shape}"
            )

        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb_pil = PIL.Image.fromarray(rgb.astype(np.uint8))
            rgb = np.array(
                rgb_pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
            )

        rgb_f32 = rgb.astype(np.float32)
        if rgb_f32.max() <= 1.0:
            rgb_f32 = rgb_f32 * 2.0 - 1.0
        else:
            rgb_f32 = rgb_f32 / 127.5 - 1.0
        return rgb_f32.transpose(2, 0, 1)

    def prepare_reference_image_condition(
        self,
        ref_image: npt.NDArray[np.uint8],
        height: int,
        width: int,
        height_latent: int,
        width_latent: int,
        vae_t: int,
        device: Device,
    ) -> npt.NDArray[np.float32]:
        """Encode the reference image into the fixed segment condition prefix."""
        ref_f32 = self._normalize_reference_image(ref_image, height, width)

        ref_video = ref_f32[np.newaxis, :, np.newaxis, :, :]
        ref_buf = _numpy_f32_to_buffer(ref_video, self.vae.config.dtype, device)
        ref_latent = self.vae.encode(ref_buf)
        ref_latent_std = _buffer_to_numpy_f32(
            self._standardize_latents_model.execute(
                ref_latent, self._vae_mean_buf, self._vae_std_buf
            )[0]
        )
        ref_mask = np.ones(
            (1, vae_t, 1, height_latent, width_latent), dtype=np.float32
        )
        return np.concatenate([ref_mask, ref_latent_std], axis=1)

    def prepare_segment_condition(
        self,
        *,
        seg_idx: int,
        animate_mode: str,
        bg_video_np: npt.NDArray[np.float32] | None,
        mask_video_np: npt.NDArray[np.float32] | None,
        seg_start: int,
        seg_end: int,
        prev_segment_cond_video: npt.NDArray[np.float32] | None,
        num_seg_frames: int,
        prev_cond_frames: int,
        pose_latents: Buffer,
        y_ref: npt.NDArray[np.float32],
        height: int,
        width: int,
        height_latent: int,
        width_latent: int,
        vae_t: int,
        device: Device,
    ) -> tuple[Buffer, int]:
        """Prepare the per-segment I2V conditioning tensor."""
        t_l = int(pose_latents.shape[2])
        total_latent_t = 1 + t_l

        if animate_mode == "replace":
            if bg_video_np is None or mask_video_np is None:
                raise ValueError(
                    "Replace mode requires bg_video_np and mask_video_np."
                )
            seg_bg_video_np = bg_video_np[seg_start:seg_end]
            seg_mask_video_np = mask_video_np[seg_start:seg_end]
            prev_video = self._prepare_replace_cond_video(
                seg_bg_video_np,
                prev_segment_cond_video,
                prev_cond_frames,
            )
            cond_lat_t = (
                0
                if seg_idx == 0
                else (prev_cond_frames - 1) // self.vae_scale_factor_temporal
                + 1
            )
            prev_mask = self._prepare_replace_mask(
                seg_mask_video_np,
                height_latent,
                width_latent,
                vae_t,
                t_l,
                cond_lat_t,
            )
        else:
            prev_video = self._pad_animate_cond_video(
                prev_segment_cond_video,
                num_seg_frames,
                height,
                width,
            )
            prev_mask = np.zeros(
                (1, vae_t, t_l, height_latent, width_latent),
                dtype=np.float32,
            )
            if seg_idx > 0 and prev_segment_cond_video is not None:
                cond_lat_t = (
                    prev_cond_frames - 1
                ) // self.vae_scale_factor_temporal + 1
                prev_mask[:, :, :cond_lat_t, :, :] = 1.0

        prev_buf = _numpy_f32_to_buffer(
            prev_video, self.vae.config.dtype, device
        )
        prev_latent = self.vae.encode(prev_buf)
        prev_latent_std = _buffer_to_numpy_f32(
            self._standardize_latents_model.execute(
                prev_latent, self._vae_mean_buf, self._vae_std_buf
            )[0]
        )
        y_prev = np.concatenate([prev_mask, prev_latent_std], axis=1).astype(
            np.float32
        )
        y_full = np.concatenate([y_ref, y_prev], axis=2).astype(np.float32)
        logger.info(
            "Condition shape: %s, noise T=%d, pose T=%d",
            y_full.shape,
            total_latent_t,
            t_l,
        )
        return (
            _numpy_f32_to_buffer(y_full, self.vae.config.dtype, device),
            total_latent_t,
        )

    def _pad_animate_cond_video(
        self,
        prev_segment_cond_video: npt.NDArray[np.float32] | None,
        num_seg_frames: int,
        height: int,
        width: int,
    ) -> npt.NDArray[np.float32]:
        """Prepare animate-mode conditioning video for the current segment."""
        if prev_segment_cond_video is None:
            return np.zeros(
                (1, 3, num_seg_frames, height, width), dtype=np.float32
            )

        prev_n = prev_segment_cond_video.shape[2]
        remaining = num_seg_frames - prev_n
        zero_pad = np.zeros((1, 3, remaining, height, width), dtype=np.float32)
        return np.concatenate([prev_segment_cond_video, zero_pad], axis=2)

    def prepare_segment_latents(
        self,
        model_inputs: WanAnimateModelInputs,
        seg_idx: int,
        noise_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Prepare per-segment noise latents."""
        if (
            seg_idx == 0
            and model_inputs.latents is not None
            and model_inputs.latents.shape == noise_shape
        ):
            latents_np = model_inputs.latents.astype(np.float32)
            logger.info("Using provided initial noise: %s", noise_shape)
        else:
            # Generate noise with seed for deterministic results
            if model_inputs.seed is not None:
                segment_seed = model_inputs.seed + seg_idx
                rng = np.random.RandomState(segment_seed)
                latents_np = rng.standard_normal(noise_shape).astype(np.float32)
            else:
                latents_np = np.random.randn(*noise_shape).astype(np.float32)
        return Buffer.from_numpy(latents_np).to(device)

    def _prepare_replace_cond_video(
        self,
        bg_video_np_seg: npt.NDArray[np.float32],
        prev_cond_video: npt.NDArray[np.float32] | None,
        prev_cond_frames: int,
    ) -> npt.NDArray[np.float32]:
        """Prepare full-segment conditioning video for replace mode.

        First segment (prev_cond_video=None): returns the full background segment.
        Subsequent segments: returns [prev_generated_frames | background_remainder].

        Returns:
            npt.NDArray[np.float32] with shape [1, 3, T, H, W] in [-1, 1].
        """
        bg_np = bg_video_np_seg[np.newaxis].transpose(0, 2, 1, 3, 4)

        if prev_cond_video is None:
            return bg_np

        # Concat [prev_cond_frames | background_remainder]
        bg_remainder = bg_np[:, :, prev_cond_frames:, :, :]
        return np.concatenate([prev_cond_video, bg_remainder], axis=2)

    def _prepare_replace_mask(
        self,
        mask_video_np_seg: npt.NDArray[np.float32],
        height_latent: int,
        width_latent: int,
        vae_t: int,
        t_l: int,
        cond_lat_t: int,
    ) -> npt.NDArray[np.float32]:
        """Prepare the I2V mask for replace mode.

        Inverts the pixel mask (background=1, character=0), downsamples to
        latent spatial dims, expands the first frame by vae_t (matching
        diffusers get_i2v_mask), then reshapes to [1, vae_t, T_l, H_l, W_l].

        Args:
            mask_video_np_seg: Pixel-space mask frames (1=foreground/character).
            height_latent, width_latent: Latent spatial dims.
            vae_t: VAE temporal scale factor (4).
            t_l: Number of latent temporal frames.
            cond_lat_t: Number of leading latent frames to force to 1
                (0 for first segment, cond frames count for subsequent).

        Returns:
            npt.NDArray[np.float32] with shape [1, vae_t, T_l, H_l, W_l].
        """
        mask_lat = np.zeros(
            (mask_video_np_seg.shape[0], height_latent, width_latent),
            dtype=np.float32,
        )
        for i, frame in enumerate(mask_video_np_seg):
            mask_uint8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            mask_lat[i] = (
                np.array(
                    PIL.Image.fromarray(mask_uint8).resize(
                        (width_latent, height_latent),
                        PIL.Image.Resampling.NEAREST,
                    ),
                    dtype=np.uint8,
                ).astype(np.float32)
                / 255.0
            )

        # Invert: background=1 (preserve), character=0 (generate)
        mask_lat = 1.0 - mask_lat  # [T, H_l, W_l]

        # Shape [1, 1, T, H_l, W_l] for get_i2v_mask equivalent logic
        mask_5d = mask_lat[np.newaxis, np.newaxis, :, :, :]

        # Force leading conditioning frames to 1 (from previous segment)
        if cond_lat_t > 0:
            mask_5d[:, :, :cond_lat_t, :, :] = 1.0

        # Expand first frame by vae_t (matching diffusers get_i2v_mask):
        # [1,1,T,H_l,W_l] → first frame ×vae_t → [1,1,vae_t+T-1,H_l,W_l]
        # = [1,1,t_l*vae_t,H_l,W_l] since T = (t_l-1)*vae_t+1
        first_frame = np.repeat(mask_5d[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_frame, mask_5d[:, :, 1:, :, :]], axis=2
        )

        # Reshape [1, 1, t_l*vae_t, H_l, W_l] → [1, t_l, vae_t, H_l, W_l]
        # then transpose → [1, vae_t, t_l, H_l, W_l]
        mask_expanded = mask_expanded.reshape(
            1, -1, vae_t, height_latent, width_latent
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)

        return mask_expanded.astype(np.float32)

    def encode_pose_segment(
        self,
        pose_video_np_seg: npt.NDArray[np.float32],
        device: Device,
    ) -> Buffer:
        """Encode pre-normalized pose frames via VAE to get pose latents.

        Returns:
            Buffer [B, 16, T_latent, H_l, W_l] pose latents.
        """
        video_np = pose_video_np_seg[np.newaxis].transpose(0, 2, 1, 3, 4)

        enc_buf = _numpy_f32_to_buffer(
            np.ascontiguousarray(video_np, dtype=np.float32),
            self.vae.config.dtype,
            device,
        )
        enc_latent = self.vae.encode(enc_buf)
        return self._standardize_latents_model.execute(
            enc_latent, self._vae_mean_buf, self._vae_std_buf
        )[0]

    def encode_face_segment(
        self,
        face_pixels_seg: npt.NDArray[np.float32],
        device: Device,
    ) -> Buffer:
        """Encode pre-normalized face frames via motion encoder + face encoder.

        Args:
            face_pixels_seg: Face frames [T, 3, 512, 512] in [-1, 1].
            device: Target device.

        Returns:
            face_emb Buffer [B, T//4+1, 5, 5120].
        """
        # MAX-native motion encoder — single batch for all frames.
        all_buf = _numpy_f32_to_buffer(
            np.ascontiguousarray(face_pixels_seg, dtype=np.float32),
            self.vae.config.dtype,
            device,
        )
        motion_buf = self.transformer.encode_motion(all_buf)

        # Face encoder (MAX Graph): [1, T, 512] -> [1, T//4+1, 5, 5120]
        # Unsqueeze motion_buf [T, 512] -> [1, T, 512] via zero-copy view.
        motion_shape = tuple(int(d) for d in motion_buf.shape)
        motion_buf_3d = motion_buf.view(
            motion_buf.dtype, (1, motion_shape[0], motion_shape[1])
        )
        return self.transformer.encode_face(motion_buf_3d)

    def prepare_scheduler_state(
        self,
        latents: Buffer,
        model_inputs: WanModelInputs,
        prompt_embeds: Buffer,
        do_cfg: bool,
        device: Device,
    ) -> tuple[list[Buffer], list[Buffer], Buffer | None]:
        """Compute scheduler buffers used by the animate denoising loop."""
        timesteps = np.ascontiguousarray(
            model_inputs.timesteps, dtype=np.float32
        )
        batched_timesteps = self._get_batched_timesteps(
            scheduler_timesteps=timesteps,
            batch_size=int(latents.shape[0]),
            device=device,
        )
        if model_inputs.step_coefficients is None:
            raise ValueError(
                "WanAnimatePipeline requires precomputed step_coefficients."
            )
        coeff_buffers = [
            Buffer.from_numpy(np.ascontiguousarray(row, dtype=np.float32)).to(
                device
            )
            for row in model_inputs.step_coefficients
        ]

        guidance_scale_high: Buffer | None = None
        if do_cfg:
            guidance_scale_high = self._get_guidance_scale(
                float(model_inputs.guidance_scale),
                dtype=prompt_embeds.dtype,
                device=device,
            )
        return batched_timesteps, coeff_buffers, guidance_scale_high

    def run_animate_denoising(
        self,
        latents: Buffer,
        condition: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        clip_features: Buffer,
        pose_latents: Buffer,
        face_emb: Buffer,
        total_latent_t: int,
        height_latent: int,
        width_latent: int,
        do_cfg: bool,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        device: Device,
        guidance_scale: Buffer | None,
    ) -> Buffer:
        """Run denoising loop with animate conditioning."""
        rope_cos, rope_sin = self.transformer.compute_rope(
            total_latent_t,
            height_latent,
            width_latent,
        )

        p_t, p_h, p_w = self.transformer.config.patch_size
        ppf = total_latent_t // p_t
        pph = height_latent // p_h
        ppw = width_latent // p_w
        spatial_shape = Buffer.from_numpy(
            np.zeros((ppf, pph, ppw), dtype=np.int8)
        ).to(device)
        num_temporal_frames = Buffer.from_numpy(
            np.array([ppf], dtype=np.int32)
        ).to(device)

        step_state: WanUniPCState = (None, None, None)
        progress = tqdm(  # type: ignore[call-arg]
            range(len(batched_timesteps)),
            desc="Denoising",
            leave=True,
            disable=not sys.stderr.isatty(),
        )

        # Pre-compute uncond face embedding once (constant across steps)
        uncond_face_emb: Buffer | None = None
        if do_cfg and negative_prompt_embeds is not None:
            uncond_face_emb = self._uncond_face_embedding_model.execute(
                face_emb
            )[0]

        for i in progress:  # type: ignore[attr-defined]
            with Tracer(f"denoise_step_{i}"):
                dit_timestep = batched_timesteps[i]
                latent_model_input = self._cast_f32_to_model_dtype.execute(
                    latents
                )[0]

                # Concat condition with latents -> 36 channels
                latent_model_input = self._concat_i2v_condition(
                    latent_model_input, condition
                )

                # Run animate transformer
                with Tracer("transformer"):
                    noise_pred_buf = self.transformer(
                        latent_model_input,
                        dit_timestep,
                        prompt_embeds,
                        clip_features,
                        pose_latents,
                        rope_cos,
                        rope_sin,
                        spatial_shape,
                        face_emb,
                        num_temporal_frames,
                    )
                    noise_pred_buf = getattr(
                        noise_pred_buf, "driver_tensor", noise_pred_buf
                    )

                # CFG (2-pass: positive then negative)
                if do_cfg and negative_prompt_embeds is not None:
                    assert guidance_scale is not None
                    assert uncond_face_emb is not None
                    noise_uncond = self.transformer(
                        latent_model_input,
                        dit_timestep,
                        negative_prompt_embeds,
                        clip_features,
                        pose_latents,
                        rope_cos,
                        rope_sin,
                        spatial_shape,
                        uncond_face_emb,
                        num_temporal_frames,
                    )
                    noise_uncond = getattr(
                        noise_uncond, "driver_tensor", noise_uncond
                    )
                    noise_pred_buf = self.guidance(
                        noise_pred_buf, noise_uncond, guidance_scale
                    )
                    noise_pred_buf = getattr(
                        noise_pred_buf, "driver_tensor", noise_pred_buf
                    )

                # Scheduler step
                with Tracer("scheduler_step"):
                    latents, step_state = self.scheduler_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )

        return latents

    def decode_segment_latents(
        self,
        latents: Buffer,
    ) -> npt.NDArray[np.float32]:
        """Decode latents to video frames, optionally skipping first frame.

        Matches diffusers: skip first LATENT frame (ref conditioning),
        then VAE decode the remaining frames.
        """
        # Denormalize latents
        denormed = self._denormalize_vae_latents(latents)

        # VAE decode
        decoded = self.vae.decode(denormed)
        return _buffer_to_numpy_f32(decoded[0])

    def prepare_inputs(self, context: Any) -> WanAnimateModelInputs:
        """Prepare animate model inputs from context."""
        base_inputs = super().prepare_inputs(context)
        animate_inputs = WanAnimateModelInputs(**base_inputs.__dict__)

        if getattr(context, "pose_video_np", None) is not None:
            animate_inputs.pose_video_np = np.asarray(
                context.pose_video_np, dtype=np.float32
            )
        if getattr(context, "face_pixels_np", None) is not None:
            animate_inputs.face_pixels_np = np.asarray(
                context.face_pixels_np, dtype=np.float32
            )
        if hasattr(context, "input_image"):
            animate_inputs.input_image = context.input_image
        if hasattr(context, "animate_mode"):
            animate_inputs.animate_mode = context.animate_mode
        if getattr(context, "bg_video_np", None) is not None:
            animate_inputs.bg_video_np = np.asarray(
                context.bg_video_np, dtype=np.float32
            )
        if getattr(context, "mask_video_np", None) is not None:
            animate_inputs.mask_video_np = np.asarray(
                context.mask_video_np, dtype=np.float32
            )
        if getattr(context, "num_pose_frames", None) is not None:
            animate_inputs.num_pose_frames = int(context.num_pose_frames)
        if getattr(context, "num_segments", None) is not None:
            animate_inputs.num_segments = int(context.num_segments)
        if getattr(context, "effective_seg_len", None) is not None:
            animate_inputs.effective_seg_len = int(context.effective_seg_len)
        if hasattr(context, "segment_frame_length"):
            animate_inputs.segment_frame_length = int(
                context.segment_frame_length
            )
        if hasattr(context, "prev_segment_conditioning_frames"):
            animate_inputs.prev_segment_conditioning_frames = int(
                context.prev_segment_conditioning_frames
            )
        if hasattr(context, "seed"):
            animate_inputs.seed = (
                None if context.seed is None else int(context.seed)
            )

        return animate_inputs
