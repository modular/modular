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
        self._guidance_graph = self._compile_guidance()
        self._unipc_step_graph = self._compile_unipc_step()
        self._duplicate_cfg_latents_graph = (
            self._compile_duplicate_cfg_latents()
        )
        self._duplicate_cfg_timesteps_graph = (
            self._compile_duplicate_cfg_timesteps()
        )
        self._concat_cfg_embeddings_graph = (
            self._compile_concat_cfg_embeddings()
        )
        self._split_cfg_predictions_graph = (
            self._compile_split_cfg_predictions()
        )
        self._cast_f32_to_model_dtype_graph = (
            self._compile_cast_f32_to_model_dtype()
        )

        # I2V concat graph compiled lazily on first use.
        self._i2v_concat_graph: Model | None = None

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
                attention_mask=None,
                num_videos_per_prompt=num_videos_per_prompt,
            )

            negative_prompt_embeds: Buffer | None = None
            if inputs.negative_tokens is not None:
                negative_prompt_embeds = self.text_encoder(
                    inputs.negative_tokens,
                    attention_mask=None,
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
        for i in step_range:
            with Tracer(f"{desc}:step_{i}"):
                dit_timestep = batched_timesteps[i]

                # Cast latents f32 -> model dtype.
                latent_model_input = (
                    self._cast_f32_to_model_dtype_graph.execute(latents)[0]
                )

                # Optional: I2V concat.
                if i2v_condition is not None:
                    latent_model_input = self._i2v_concat(
                        latent_model_input, i2v_condition
                    )

                # Transformer forward with optional CFG.
                with Tracer("transformer"):
                    noise_pred_buf = self._run_transformer_forward(
                        use_secondary=use_secondary_transformer,
                        latent_model_input=latent_model_input,
                        dit_timestep=dit_timestep,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        spatial_shape=spatial_shape,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                    )

                # UniPC scheduler step.
                with Tracer("scheduler_step"):
                    latents, step_state = self._unipc_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )

        return latents, step_state

    def _run_transformer_forward(
        self,
        *,
        use_secondary: bool,
        latent_model_input: Buffer,
        dit_timestep: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        do_cfg: bool,
        guidance_scale: Buffer | None,
    ) -> Buffer:
        """Run transformer + optional CFG guidance."""
        # Select transformer call.
        if use_secondary:
            transformer_call = self.transformer.call_secondary
        else:
            transformer_call = self.transformer.__call__

        # Two separate forward passes for CFG (unbatched path).
        noise_pred_buf = transformer_call(
            latent_model_input,
            dit_timestep,
            prompt_embeds,
            rope_cos,
            rope_sin,
            spatial_shape,
        )

        if do_cfg and negative_prompt_embeds is not None:
            assert guidance_scale is not None
            noise_uncond_buf = transformer_call(
                latent_model_input,
                dit_timestep,
                negative_prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
            guided = self._guidance_graph.execute(
                noise_pred_buf,
                noise_uncond_buf,
                guidance_scale,
            )
            return guided[0]

        return noise_pred_buf

    def _unipc_step(
        self,
        latents: Buffer,
        noise_pred: Buffer,
        coeffs: Buffer,
        step_state: WanUniPCState,
    ) -> tuple[Buffer, WanUniPCState]:
        """Run a single UniPC scheduler step."""
        last_sample, prev_model_output, older_model_output = step_state
        if last_sample is None:
            shape = tuple(int(d) for d in latents.shape)
            zero = Buffer.from_numpy(np.zeros(shape, dtype=np.float32)).to(
                latents.device
            )
            last_sample = zero
            prev_model_output = zero
            older_model_output = zero

        assert prev_model_output is not None
        assert older_model_output is not None
        assert last_sample is not None

        result = self._unipc_step_graph.execute(
            latents,
            noise_pred,
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

    def _i2v_concat(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Buffer:
        """Concat latents with I2V condition along channel axis."""
        if self._i2v_concat_graph is None:
            self._i2v_concat_graph = self._compile_i2v_concat(
                latent_model_input, condition
            )
        return self._i2v_concat_graph.execute(latent_model_input, condition)[0]

    # -- Helper graph compilation ---------------------------------------------

    def _compile_guidance(self) -> Model:
        """Compile CFG guidance: uncond + scale * (cond - uncond)."""
        device = self._model_device
        dtype = self._model_dtype
        latent_type = TensorType(
            dtype,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        input_types = [
            latent_type,  # noise_pred
            latent_type,  # noise_uncond
            TensorType(dtype, shape=[1], device=device),  # guidance_scale
        ]

        with Graph("wan_guidance", input_types=input_types) as g:
            noise_pred = g.inputs[0].tensor
            noise_uncond = g.inputs[1].tensor
            scale = g.inputs[2].tensor
            g.output(noise_uncond + scale * (noise_pred - noise_uncond))
        return self._session.load(g)

    def _compile_unipc_step(self) -> Model:
        """Compile UniPC scheduler step."""
        device = self._model_device
        model_dtype = self._model_dtype
        latent_type_f32 = TensorType(
            DType.float32,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        latent_type_model = TensorType(
            model_dtype,
            shape=["batch", "channels", "frames", "height", "width"],
            device=device,
        )
        coeff_type = TensorType(DType.float32, shape=[9], device=device)
        input_types = [
            latent_type_f32,  # sample (f32)
            latent_type_model,  # model_output (model dtype)
            latent_type_f32,  # last_sample
            latent_type_f32,  # prev_model_output
            latent_type_f32,  # older_model_output
            coeff_type,
        ]

        with Graph("wan_unipc_step", input_types=input_types) as g:
            sample = g.inputs[0].tensor
            model_output = ops.cast(g.inputs[1].tensor, DType.float32)
            last_sample = g.inputs[2].tensor
            prev_model_output = g.inputs[3].tensor
            older_model_output = g.inputs[4].tensor
            coeffs = g.inputs[5].tensor

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

    def _compile_duplicate_cfg_latents(self) -> Model:
        """Compile CFG latent duplication: concat([x, x], axis=0)."""
        device = self._model_device
        dtype = self._model_dtype
        in_channels = self.transformer.config.in_channels

        with Graph(
            "wan_dup_cfg_latents",
            input_types=[
                TensorType(
                    dtype,
                    shape=[1, in_channels, "frames", "height", "width"],
                    device=device,
                )
            ],
        ) as g:
            g.output(
                ops.concat([g.inputs[0].tensor, g.inputs[0].tensor], axis=0)
            )
        return self._session.load(g)

    def _compile_duplicate_cfg_timesteps(self) -> Model:
        """Compile CFG timestep duplication."""
        device = self._model_device
        with Graph(
            "wan_dup_cfg_timesteps",
            input_types=[TensorType(DType.float32, shape=[1], device=device)],
        ) as g:
            g.output(
                ops.concat([g.inputs[0].tensor, g.inputs[0].tensor], axis=0)
            )
        return self._session.load(g)

    def _compile_concat_cfg_embeddings(self) -> Model:
        """Compile CFG prompt embedding concatenation."""
        device = self._model_device
        text_dim = self.transformer.config.text_dim
        embed_dtype = self._model_dtype

        with Graph(
            "wan_concat_cfg_embeddings",
            input_types=[
                TensorType(
                    embed_dtype,
                    shape=[1, "seq_text", text_dim],
                    device=device,
                ),
                TensorType(
                    embed_dtype,
                    shape=[1, "seq_text", text_dim],
                    device=device,
                ),
            ],
        ) as g:
            g.output(
                ops.concat([g.inputs[0].tensor, g.inputs[1].tensor], axis=0)
            )
        return self._session.load(g)

    def _compile_split_cfg_predictions(self) -> Model:
        """Compile CFG prediction splitting."""
        device = self._model_device
        dtype = self._model_dtype
        out_channels = self.transformer.config.out_channels

        with Graph(
            "wan_split_cfg_predictions",
            input_types=[
                TensorType(
                    dtype,
                    shape=[2, out_channels, "frames", "height", "width"],
                    device=device,
                )
            ],
        ) as g:
            batched = g.inputs[0].tensor
            positive = ops.slice_tensor(
                batched,
                [
                    slice(0, 1),
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                ],
            )
            negative = ops.slice_tensor(
                batched,
                [
                    slice(1, 2),
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                ],
            )
            g.output(positive, negative)
        return self._session.load(g)

    def _compile_cast_f32_to_model_dtype(self) -> Model:
        """Compile float32 -> model dtype cast graph."""
        device = self._model_device
        model_dtype = self._model_dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_f32_to_mdtype",
            input_types=[TensorType(DType.float32, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, model_dtype))
        return self._session.load(g)

    def _compile_i2v_concat(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Model:
        """Compile I2V condition concat graph (lazy, on first use)."""
        device = self._model_device
        dtype = latent_model_input.dtype
        lat_shape: list[Any] = [
            int(latent_model_input.shape[0]),
            int(latent_model_input.shape[1]),
            "T",
            "H",
            "W",
        ]
        cond_shape: list[Any] = [
            int(condition.shape[0]),
            int(condition.shape[1]),
            "T",
            "H",
            "W",
        ]

        with Graph(
            "wan_i2v_concat",
            input_types=[
                TensorType(dtype, lat_shape, device=device),
                TensorType(dtype, cond_shape, device=device),
            ],
        ) as g:
            g.output(
                ops.concat([g.inputs[0].tensor, g.inputs[1].tensor], axis=1)
            )
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
