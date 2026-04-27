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

"""Wan executor implementing the PipelineExecutor interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType, ops
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces import TensorStruct
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.profiler import Tracer, traced
from typing_extensions import Self

from .components import TextEncoder, VaeWrapper, WanTransformer
from .context import WanContext

logger = logging.getLogger("max.pipelines")

# ---------------------------------------------------------------------------
# Input / Output structs
# ---------------------------------------------------------------------------

# UniPC scheduler state: (last_sample, prev_model_output, older_model_output)
WanUniPCState = tuple[Buffer | None, Buffer | None, Buffer | None]


@dataclass(frozen=True)
class WanExecutorInputs(TensorStruct):
    """Structured inputs for Wan execution.

    Core fields are always present. Optional feature fields are ``None``
    when the feature is disabled for a given request.
    """

    # -- Core (always present) ------------------------------------------------

    tokens: Buffer
    """Positive prompt token IDs, shape ``(S,)`` int64."""

    latents: Buffer
    """Noise latents, shape ``(B, C, T, H, W)`` float32."""

    timesteps: Buffer
    """Scheduler timesteps, shape ``(num_steps,)`` float32."""

    step_coefficients: Buffer
    """Pre-computed UniPC coefficients, shape ``(num_steps, 9)`` float32."""

    rope_cos: Buffer
    """RoPE cosine, shape ``(seq_len, head_dim)`` float32."""

    rope_sin: Buffer
    """RoPE sine, shape ``(seq_len, head_dim)`` float32."""

    spatial_shape: Buffer
    """Shape carrier ``(ppf, pph, ppw)`` int8."""

    width: Buffer
    """Output width in pixels, 1-element int64."""

    height: Buffer
    """Output height in pixels, 1-element int64."""

    num_frames: Buffer
    """Number of video frames, 1-element int64."""

    num_inference_steps: Buffer
    """Number of denoising steps, 1-element int64."""

    num_images_per_prompt: Buffer
    """Number of videos per prompt, 1-element int64."""

    guidance_scale: Buffer
    """Guidance scale, shape ``(1,)`` in model dtype."""

    # -- Optional features ----------------------------------------------------

    negative_tokens: Buffer | None = None
    """Negative prompt token IDs, shape ``(S,)`` int64."""

    input_image: Buffer | None = None
    """Input image for I2V, shape ``(H, W, 3)`` float32."""

    guidance_scale_2: Buffer | None = None
    """Secondary guidance scale for MoE low-noise phase, shape ``(1,)``."""

    boundary_timestep: Buffer | None = None
    """MoE boundary timestep, shape ``(1,)`` float32."""

    # -- Device transfer -------------------------------------------------------

    _CPU_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "width",
            "height",
            "num_frames",
            "num_inference_steps",
            "num_images_per_prompt",
        }
    )

    def to(self, device: Device) -> Self:
        """Transfer GPU-bound tensors to *device*, keeping metadata on CPU."""
        updates: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._CPU_FIELDS:
                continue
            val = getattr(self, f.name)
            if isinstance(val, Buffer):
                updates[f.name] = val.to(device)
        return replace(self, **updates)


@dataclass(frozen=True)
class WanExecutorOutputs(TensorStruct):
    """Structured outputs from Wan execution."""

    images: Buffer
    """Decoded video frames as numpy-compatible buffer."""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class WanExecutor(
    PipelineExecutor[WanContext, WanExecutorInputs, WanExecutorOutputs]
):
    """Wan video diffusion pipeline executor.

    Implements the :class:`PipelineExecutor` interface for Wan video
    generation, wiring together the sub-components (text encoder,
    transformer, VAE) through the tensor-in/tensor-out executor contract.
    """

    default_num_inference_steps: int = 50

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        self._manifest = manifest
        self._session = session
        self._runtime_config = runtime_config

        # Extract model config.
        transformer_config = manifest["transformer"]
        encoding = transformer_config.quantization_encoding or "bfloat16"
        self._model_dtype: DType = supported_encoding_dtype(encoding)
        self._model_device: Device = load_devices(
            transformer_config.device_specs
        )[0]

        # Build components.
        self.text_encoder = TextEncoder(manifest, session)
        self.transformer = WanTransformer(manifest, session)
        self.vae = VaeWrapper(manifest, session)

        # Extract scheduler config from diffusers config.
        diffusers_config = transformer_config.huggingface_config.to_dict()
        components_cfg = diffusers_config.get("components", {})
        scheduler_cfg = components_cfg.get("scheduler", {}).get(
            "config_dict", {}
        )
        self._num_train_timesteps = int(
            scheduler_cfg.get("num_train_timesteps", 1000)
        )
        self._boundary_ratio: float | None = diffusers_config.get(
            "boundary_ratio"
        )

        transformer_cfg = components_cfg.get("transformer", {}).get(
            "config_dict", {}
        )
        self._expand_timesteps = bool(
            transformer_cfg.get("expand_timesteps", False)
        )

        # Compile helper graphs.
        self._guided_schedule_graph = self._compile_guided_schedule()

        # Runtime caches.
        self._guidance_scale_cache: dict[tuple[float, DType, str], Buffer] = {}
        self._batched_timesteps_cache: dict[str, list[Buffer]] = {}
        self._spatial_shape_cache: dict[str, Buffer] = {}

    # -- PipelineExecutor interface -------------------------------------------

    @traced(message="WanExecutor.prepare_inputs")
    def prepare_inputs(self, contexts: list[WanContext]) -> WanExecutorInputs:
        if len(contexts) != 1:
            raise ValueError(
                "WanExecutor currently supports batch_size=1. "
                f"Got {len(contexts)} contexts."
            )
        context = contexts[0]

        num_frames = 81
        if context.num_frames is not None:
            num_frames = int(context.num_frames)

        latents = np.asarray(context.latents, dtype=np.float32)
        if latents.ndim == 5:
            latent_frames = int(latents.shape[2])
            num_frames = self._normalize_num_frames(num_frames, latent_frames)

        timesteps = np.ascontiguousarray(context.timesteps, dtype=np.float32)

        if context.step_coefficients is None:
            raise ValueError(
                "WanExecutor requires precomputed step_coefficients."
            )
        step_coefficients = np.ascontiguousarray(
            context.step_coefficients, dtype=np.float32
        )

        # Compute RoPE for the latent resolution.
        with Tracer("compute_rope"):
            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]) if latents.ndim == 5 else 1,
                height=int(latents.shape[3]) if latents.ndim == 5 else 1,
                width=int(latents.shape[4]) if latents.ndim == 5 else 1,
            )

        # Spatial shape carrier.
        p_t, p_h, p_w = self.transformer.config.patch_size
        if latents.ndim == 5:
            ppf = int(latents.shape[2]) // p_t
            pph = int(latents.shape[3]) // p_h
            ppw = int(latents.shape[4]) // p_w
        else:
            ppf = pph = ppw = 1
        spatial_shape = Buffer.from_numpy(
            np.zeros((ppf, pph, ppw), dtype=np.int8)
        ).to(self._model_device)

        # Guidance scale buffer.
        guidance_scale = self._make_guidance_scale_buffer(
            float(context.guidance_scale),
            dtype=self._model_dtype,
            device=self._model_device,
        )

        # Token buffers.
        tokens = Buffer.from_dlpack(context.tokens.array)

        negative_tokens: Buffer | None = None
        if context.negative_tokens is not None:
            negative_tokens = Buffer.from_dlpack(context.negative_tokens.array)

        # Optional fields.
        input_image: Buffer | None = None
        if context.input_image is not None:
            input_image = Buffer.from_dlpack(
                np.ascontiguousarray(context.input_image, dtype=np.float32)
            )

        guidance_scale_2: Buffer | None = None
        if context.guidance_scale_2 is not None:
            guidance_scale_2 = self._make_guidance_scale_buffer(
                float(context.guidance_scale_2),
                dtype=self._model_dtype,
                device=self._model_device,
            )

        boundary_timestep: Buffer | None = None
        if context.boundary_timestep is not None:
            boundary_timestep = Buffer.from_dlpack(
                np.array([context.boundary_timestep], dtype=np.float32)
            )

        return WanExecutorInputs(
            tokens=tokens,
            latents=Buffer.from_numpy(
                np.ascontiguousarray(latents, dtype=np.float32)
            ),
            timesteps=Buffer.from_dlpack(timesteps),
            step_coefficients=Buffer.from_dlpack(step_coefficients),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            spatial_shape=spatial_shape,
            width=Buffer.from_dlpack(np.array([context.width], dtype=np.int64)),
            height=Buffer.from_dlpack(
                np.array([context.height], dtype=np.int64)
            ),
            num_frames=Buffer.from_dlpack(
                np.array([num_frames], dtype=np.int64)
            ),
            num_inference_steps=Buffer.from_dlpack(
                np.array([context.num_inference_steps], dtype=np.int64)
            ),
            num_images_per_prompt=Buffer.from_dlpack(
                np.array(
                    [context.num_images_per_prompt * num_frames],
                    dtype=np.int64,
                )
            ),
            guidance_scale=guidance_scale,
            negative_tokens=negative_tokens,
            input_image=input_image,
            guidance_scale_2=guidance_scale_2,
            boundary_timestep=boundary_timestep,
        )

    @traced(message="WanExecutor.execute")
    def execute(self, inputs: WanExecutorInputs) -> WanExecutorOutputs:
        device = self._model_device

        # Bulk device transfer.
        with Tracer("inputs_to_device"):
            inputs = inputs.to(device)

        # 1. Encode prompts.
        # num_images_per_prompt includes the frame count for
        # pixel_generation's output count check; use 1 video per prompt
        # for text encoding.
        with Tracer("encode_prompts"):
            num_frames_val = int(np.from_dlpack(inputs.num_frames).flat[0])
            num_images_per_prompt = int(
                np.from_dlpack(inputs.num_images_per_prompt).flat[0]
            )
            num_videos_per_prompt = num_images_per_prompt // max(
                num_frames_val, 1
            )
            prompt_embeds = self.text_encoder(
                inputs.tokens,
                num_videos_per_prompt=num_videos_per_prompt,
            )

            negative_prompt_embeds: Buffer | None = None
            if inputs.negative_tokens is not None:
                negative_prompt_embeds = self.text_encoder(
                    inputs.negative_tokens,
                    num_videos_per_prompt=num_videos_per_prompt,
                )

        # Determine CFG mode.
        guidance_scale_val = self._buffer_to_scalar_f32(inputs.guidance_scale)
        do_cfg = guidance_scale_val > 1.0 and negative_prompt_embeds is not None

        # 2. Prepare latents.
        latents = inputs.latents

        # 3. Prepare I2V condition if image provided.
        i2v_condition: Buffer | None = None
        if inputs.input_image is not None:
            with Tracer("prepare_i2v_condition"):
                latent_shape = tuple(int(d) for d in latents.shape)
                num_frames = int(np.from_dlpack(inputs.num_frames).flat[0])
                height = int(np.from_dlpack(inputs.height).flat[0])
                width = int(np.from_dlpack(inputs.width).flat[0])
                i2v_condition = self.vae.encode_i2v_condition(
                    np.from_dlpack(inputs.input_image.to(CPU())),
                    latent_shape,
                    num_frames,
                    height,
                    width,
                )

        # 4. Prepare scheduler state.
        with Tracer("prepare_scheduler"):
            timesteps_np = np.asarray(
                np.from_dlpack(inputs.timesteps.to(CPU())),
                dtype=np.float32,
            )
            num_steps = int(np.from_dlpack(inputs.num_inference_steps).flat[0])

            batched_timesteps = self._get_batched_timesteps(
                timesteps_np,
                batch_size=int(latents.shape[0]),
                device=device,
            )
            coeff_np = np.from_dlpack(inputs.step_coefficients.to(CPU()))
            coeff_buffers = [
                Buffer.from_numpy(
                    np.ascontiguousarray(coeff_np[i], dtype=np.float32)
                ).to(device)
                for i in range(num_steps)
            ]

            # MoE boundary.
            boundary_timestep_val: float | None = None
            if inputs.boundary_timestep is not None:
                boundary_timestep_val = float(
                    np.asarray(
                        np.from_dlpack(inputs.boundary_timestep.to(CPU())),
                        dtype=np.float32,
                    ).flat[0]
                )
            elif self._boundary_ratio is not None:
                boundary_timestep_val = (
                    self._boundary_ratio * self._num_train_timesteps
                )

            has_moe = (
                self.transformer.has_moe and boundary_timestep_val is not None
            )
            boundary_step_idx = num_steps
            if boundary_timestep_val is not None:
                for idx in range(num_steps):
                    if float(timesteps_np[idx]) < boundary_timestep_val:
                        boundary_step_idx = idx
                        break

            # Guidance scales.
            guidance_scale_high: Buffer | None = None
            guidance_scale_low: Buffer | None = None
            if do_cfg:
                guidance_scale_high = inputs.guidance_scale
                if inputs.guidance_scale_2 is not None:
                    guidance_scale_low = inputs.guidance_scale_2
                else:
                    guidance_scale_low = inputs.guidance_scale

        # 5. Denoising loop.
        with Tracer("denoising_loop"):
            step_state: WanUniPCState = (None, None, None)

            # High-noise phase (or full denoising if no MoE).
            high_phase_name = (
                "denoising_phase_high_noise" if has_moe else "denoising_phase"
            )
            with Tracer(high_phase_name):
                latents, step_state = self._run_denoising_phase(
                    latents=latents,
                    use_secondary_transformer=False,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    rope_cos=inputs.rope_cos,
                    rope_sin=inputs.rope_sin,
                    batched_timesteps=batched_timesteps,
                    coeff_buffers=coeff_buffers,
                    do_cfg=do_cfg,
                    guidance_scale=guidance_scale_high,
                    step_range=range(boundary_step_idx),
                    desc="Denoising (high-noise)" if has_moe else "Denoising",
                    spatial_shape=inputs.spatial_shape,
                    step_state=step_state,
                    i2v_condition=i2v_condition,
                )

            # Low-noise phase (MoE only).
            if has_moe and boundary_step_idx < num_steps:
                with Tracer("denoising_phase_low_noise"):
                    latents, _ = self._run_denoising_phase(
                        latents=latents,
                        use_secondary_transformer=True,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=inputs.rope_cos,
                        rope_sin=inputs.rope_sin,
                        batched_timesteps=batched_timesteps,
                        coeff_buffers=coeff_buffers,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale_low,
                        step_range=range(boundary_step_idx, num_steps),
                        desc="Denoising (low-noise)",
                        spatial_shape=inputs.spatial_shape,
                        step_state=step_state,
                        i2v_condition=i2v_condition,
                    )

        # 6. Decode.
        with Tracer("decode_outputs"):
            height_val = int(np.from_dlpack(inputs.height).flat[0])
            width_val = int(np.from_dlpack(inputs.width).flat[0])
            images = self.vae.decode(
                latents, num_frames_val, height_val, width_val
            )

        return WanExecutorOutputs(
            images=Buffer.from_numpy(np.ascontiguousarray(images))
        )

    # -- Denoising loop -------------------------------------------------------

    def _run_denoising_phase(
        self,
        latents: Buffer,
        use_secondary_transformer: bool,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        do_cfg: bool,
        guidance_scale: Buffer | None,
        step_range: range,
        desc: str,
        spatial_shape: Buffer,
        step_state: WanUniPCState,
        i2v_condition: Buffer | None = None,
    ) -> tuple[Buffer, WanUniPCState]:
        """Run a denoising phase (high-noise or low-noise)."""
        # Select transformer call.
        if use_secondary_transformer:
            transformer_call = self.transformer.call_secondary
        else:
            transformer_call = self.transformer.__call__

        # Non-CFG uses scale=1.0, which makes the guidance formula a no-op:
        # zeros + 1.0 * (cond - zeros) = cond.
        if do_cfg:
            assert guidance_scale is not None
            scale = guidance_scale
        else:
            scale = self._make_guidance_scale_buffer(
                1.0, dtype=self._model_dtype, device=self._model_device
            )

        # Lazily-created zero buffer for the unconditional prediction
        # when CFG is disabled.
        noise_uncond_buf: Buffer | None = None

        for i in step_range:
            with Tracer(f"{desc}:step_{i}"):
                dit_timestep = batched_timesteps[i]

                # Transformer forward — positive prompt.
                with Tracer("transformer"):
                    noise_pred_cond = transformer_call(
                        latents,
                        dit_timestep,
                        prompt_embeds,
                        rope_cos,
                        rope_sin,
                        spatial_shape,
                        i2v_condition=i2v_condition,
                    )

                    # Negative prompt (CFG) or zero buffer (no CFG).
                    if do_cfg and negative_prompt_embeds is not None:
                        noise_pred_uncond = transformer_call(
                            latents,
                            dit_timestep,
                            negative_prompt_embeds,
                            rope_cos,
                            rope_sin,
                            spatial_shape,
                            i2v_condition=i2v_condition,
                        )
                    else:
                        if noise_uncond_buf is None:
                            shape = tuple(int(d) for d in noise_pred_cond.shape)
                            noise_uncond_buf = self._make_zero_buffer(
                                shape,
                                dtype=self._model_dtype,
                                device=latents.device,
                            )
                        noise_pred_uncond = noise_uncond_buf

                # Fused guidance + UniPC scheduler step.
                with Tracer("guided_schedule"):
                    latents, step_state = self._guided_schedule_step(
                        noise_pred_cond,
                        noise_pred_uncond,
                        scale,
                        latents,
                        coeff_buffers[i],
                        step_state,
                    )

        return latents, step_state

    def _guided_schedule_step(
        self,
        noise_pred_cond: Buffer,
        noise_pred_uncond: Buffer,
        guidance_scale: Buffer,
        latents: Buffer,
        coeffs: Buffer,
        step_state: WanUniPCState,
    ) -> tuple[Buffer, WanUniPCState]:
        """Run fused CFG guidance + UniPC scheduler step."""
        last_sample, prev_model_output, older_model_output = step_state
        if last_sample is None:
            shape = tuple(int(d) for d in latents.shape)
            zero = self._make_zero_buffer(
                shape, dtype=DType.float32, device=latents.device
            )
            last_sample = zero
            prev_model_output = zero
            older_model_output = zero

        assert prev_model_output is not None
        assert older_model_output is not None
        assert last_sample is not None

        result = self._guided_schedule_graph.execute(
            noise_pred_cond,
            noise_pred_uncond,
            guidance_scale,
            latents,
            last_sample,
            prev_model_output,
            older_model_output,
            coeffs,
        )
        previous_sample = result[0]
        converted = result[1]
        corrected_sample = result[2]

        return previous_sample, (
            corrected_sample,
            converted,
            prev_model_output,
        )

    # -- Helper graph compilation ---------------------------------------------

    def _compile_guided_schedule(self) -> Model:
        """Compile fused CFG guidance + UniPC scheduler step.

        Internal computation:
        1. ``guided = uncond + scale * (cond - uncond)``
           (when scale=1.0 and uncond=zeros this is a no-op)
        2. ``model_output = cast(guided, f32)``
        3. UniPC corrector + predictor update
        """
        device = self._model_device
        model_dtype = self._model_dtype
        latent_type_model = TensorType(
            model_dtype,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        latent_type_f32 = TensorType(
            DType.float32,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        coeff_type = TensorType(DType.float32, shape=[9], device=device)
        input_types = [
            latent_type_model,  # noise_pred_cond
            latent_type_model,  # noise_pred_uncond
            TensorType(model_dtype, shape=[1], device=device),  # scale
            latent_type_f32,  # sample (latents)
            latent_type_f32,  # last_sample
            latent_type_f32,  # prev_model_output
            latent_type_f32,  # older_model_output
            coeff_type,  # coefficients
        ]

        with Graph("wan_guided_schedule", input_types=input_types) as g:
            noise_pred_cond = g.inputs[0].tensor
            noise_pred_uncond = g.inputs[1].tensor
            scale = g.inputs[2].tensor
            sample = g.inputs[3].tensor
            last_sample = g.inputs[4].tensor
            prev_model_output = g.inputs[5].tensor
            older_model_output = g.inputs[6].tensor
            coeffs = g.inputs[7].tensor

            # CFG guidance.
            guided = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )
            model_output = ops.cast(guided, DType.float32)

            # UniPC scheduler step.
            sigma = coeffs[0:1]
            corrected_input_scale = coeffs[1:2]
            corrector_sample_scale = coeffs[2:3]
            corrector_m0_scale = coeffs[3:4]
            corrector_m1_scale = coeffs[4:5]
            corrector_mt_scale = coeffs[5:6]
            predictor_sample_scale = coeffs[6:7]
            predictor_m0_scale = coeffs[7:8]
            predictor_m1_scale = coeffs[8:9]

            converted = sample - sigma * model_output
            corrected_sample = (
                corrected_input_scale * sample
                + corrector_sample_scale * last_sample
                + corrector_m0_scale * prev_model_output
                + corrector_m1_scale * older_model_output
                + corrector_mt_scale * converted
            )
            previous_sample = (
                predictor_sample_scale * corrected_sample
                + predictor_m0_scale * converted
                + predictor_m1_scale * prev_model_output
            )
            g.output(previous_sample, converted, corrected_sample)
        return self._session.load(g)

    # -- Utilities ------------------------------------------------------------

    def _make_guidance_scale_buffer(
        self, value: float, *, dtype: DType, device: Device
    ) -> Buffer:
        """Create a guidance scale buffer, caching by value."""
        key = (float(value), dtype, str(device.id))
        cached = self._guidance_scale_cache.get(key)
        if cached is not None:
            return cached
        if dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(
                np.array([float(value)], dtype=np.float32)
            )
            scale = (
                Buffer.from_numpy(u16)
                .to(device)
                .view(dtype=DType.bfloat16, shape=[1])
            )
        else:
            scale = Buffer.from_numpy(
                np.array([float(value)], dtype=np.float32)
            ).to(device)
        self._guidance_scale_cache[key] = scale
        return scale

    def _get_batched_timesteps(
        self,
        scheduler_timesteps: npt.NDArray[np.float32],
        batch_size: int,
        device: Device,
    ) -> list[Buffer]:
        """Create per-step batched timestep buffers, with caching."""
        key = (
            f"{batch_size}_{len(scheduler_timesteps)}_"
            f"{float(scheduler_timesteps[0]):.4f}_"
            f"{float(scheduler_timesteps[-1]):.4f}_{device.id}"
        )
        cached = self._batched_timesteps_cache.get(key)
        if cached is not None:
            return cached

        batched = [
            Buffer.from_numpy(
                np.full([batch_size], float(step_value), dtype=np.float32)
            ).to(device)
            for step_value in scheduler_timesteps
        ]
        self._batched_timesteps_cache[key] = batched
        return batched

    def _normalize_num_frames(self, requested: int, latent_frames: int) -> int:
        """Adjust num_frames to be compatible with latent temporal shape."""
        vae_t = self.vae.scale_factor_temporal
        compatible = max(1, (max(latent_frames, 1) - 1) * vae_t + 1)
        if requested <= compatible:
            return requested
        logger.warning(
            "Requested num_frames=%d incompatible with latent temporal "
            "shape (%d). Auto-adjusting to %d.",
            requested,
            latent_frames,
            compatible,
        )
        return compatible

    @staticmethod
    def _make_zero_buffer(
        shape: tuple[int, ...], *, dtype: DType, device: Device
    ) -> Buffer:
        """Create a zero-filled buffer with the given shape and dtype."""
        if dtype == DType.bfloat16:
            return (
                Buffer.from_numpy(np.zeros(shape, dtype=np.uint16))
                .to(device)
                .view(dtype=DType.bfloat16, shape=list(shape))
            )
        np_dtype = np.float32
        return Buffer.from_numpy(np.zeros(shape, dtype=np_dtype)).to(device)

    @staticmethod
    def _buffer_to_scalar_f32(buf: Buffer) -> float:
        """Extract a scalar float32 from a 1-element Buffer."""
        cpu_buf = buf.to(CPU())
        if cpu_buf.dtype == DType.bfloat16:
            u16 = np.from_dlpack(
                cpu_buf.view(dtype=DType.uint16, shape=cpu_buf.shape)
            )
            f32_arr = (u16.astype(np.uint32) << 16).view(np.float32)
            return float(np.float32(f32_arr.flat[0]))
        return float(np.float32(np.from_dlpack(cpu_buf).flat[0]))
