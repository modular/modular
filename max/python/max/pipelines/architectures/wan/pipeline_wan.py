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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    CompileWrapper,
    max_compile,
)
from max.profiler import Tracer, traced
from tqdm.auto import tqdm

from ..autoencoders import AutoencoderKLWanModel
from ..umt5 import UMT5Model
from .model import WanTransformerModel

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WanModelInputs:
    """Input container for Wan pipeline execution."""

    tokens: TokenBuffer
    negative_tokens: TokenBuffer | None = None
    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    mask: npt.NDArray[np.bool_] | None = None
    negative_mask: npt.NDArray[np.bool_] | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    guidance_scale_2: float | None = None
    boundary_timestep: float | None = None
    expand_timesteps: bool = False
    num_images_per_prompt: int = 1
    step_coefficients: npt.NDArray[np.float32] | None = None
    input_image: npt.NDArray[np.float32] | None = None


@dataclass
class WanPipelineOutput:
    images: np.ndarray | Buffer


@dataclass
class WanRuntimeCache:
    """Runtime cache for reusable Wan buffers and helper tensors."""

    spatial_shapes: dict[str, Buffer] = field(default_factory=dict)
    batched_timesteps: dict[str, list[Buffer]] = field(default_factory=dict)
    guidance_scales: dict[tuple[float, DType, str], Buffer] = field(
        default_factory=dict
    )


WanUniPCState = tuple[Buffer | None, Buffer | None, Buffer | None]


class WanPipeline(DiffusionPipeline):
    """Wan diffusion pipeline with MAX-native DiT/VAE interfaces.

    Supports Wan 2.2 MoE models with dual transformers (high-noise and
    low-noise experts) when ``transformer_2`` weights are present.
    """

    vae: AutoencoderKLWanModel
    text_encoder: UMT5Model
    transformer: WanTransformerModel
    transformer_2: WanTransformerModel | None

    # Compiled helpers — populated by build_*() methods in init_remaining_components
    guidance: CompileWrapper
    unipc_step: CompileWrapper
    duplicate_cfg_latents: CompileWrapper
    duplicate_cfg_timesteps: CompileWrapper
    concat_cfg_prompt_embeddings: CompileWrapper
    split_cfg_predictions: CompileWrapper
    _cast_f32_to_model_dtype: CompileWrapper
    _denorm: CompileWrapper

    components = {
        "vae": AutoencoderKLWanModel,
        "text_encoder": UMT5Model,
        "transformer": WanTransformerModel,
    }

    def _load_sub_models(
        self, weight_paths: list[Path]
    ) -> dict[str, ComponentModel]:
        """Load sub-models with LoRA injection for transformer."""
        import inspect

        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        components_config = diffusers_config.get("components", {})
        relative_paths = self._resolve_relative_component_paths()
        pipeline_encoding = self.pipeline_config.model.quantization_encoding

        # Resolve LoRA
        lora_files = self._resolve_lora_files()
        lora_path = lora_files.get("high_noise_model")
        if lora_path is None and lora_files:
            lora_path = next(iter(lora_files.values()))
        lora_scale = self._lora_config.get("scale", 1.0)

        loaded: dict[str, ComponentModel] = {}
        for name, cls in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            if not issubclass(cls, ComponentModel):
                continue
            config_dict = self._get_component_config_dict(
                components_config, name
            )
            if name in relative_paths:
                abs_paths = self._resolve_absolute_paths(
                    weight_paths, relative_paths[name]
                )
                encoding = pipeline_encoding
            else:
                abs_paths = self._download_component_weights(name)
                encoding = "bfloat16"

            init_params = inspect.signature(cls.__init__).parameters
            init_kwargs: dict[str, Any] = {
                "config": config_dict,
                "encoding": encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if "session" in init_params:
                init_kwargs["session"] = self.session
            if "lora_path" in init_params and lora_path is not None:
                init_kwargs["lora_path"] = lora_path
                init_kwargs["lora_scale"] = lora_scale

            with Tracer(f"load_component:{name}"):
                loaded[name] = cls(**init_kwargs)

        return loaded

    def _resolve_lora_files(self) -> dict[str, Path]:
        """Download and cache LoRA files from diffusers_config (lazy)."""
        if hasattr(self, "_lora_files"):
            return self._lora_files

        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        self._lora_config = diffusers_config.get("lora", {})
        lora_repo_id = self._lora_config.get("repo_id")
        lora_subfolder = self._lora_config.get("subfolder")

        self._lora_files: dict[str, Path] = {}
        if lora_repo_id and lora_subfolder is not None:
            from .lora_utils import download_wan_lora

            self._lora_files = download_wan_lora(
                lora_repo_id,
                lora_subfolder,
                self._lora_config.get("filenames"),
            )
        return self._lora_files

    def _setup_vae_config(self) -> None:
        """Extract VAE scale factors, scheduler config, and denorm constants."""
        self.vae_scale_factor_temporal = int(
            getattr(self.vae.config, "scale_factor_temporal", 4) or 4
        )
        self.vae_scale_factor_spatial = int(
            getattr(self.vae.config, "scale_factor_spatial", 8) or 8
        )
        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        self.boundary_ratio = diffusers_config.get("boundary_ratio")
        components_cfg = diffusers_config.get("components", {})
        scheduler_cfg = components_cfg.get("scheduler", {}).get(
            "config_dict", {}
        )
        self.num_train_timesteps = int(
            scheduler_cfg.get("num_train_timesteps", 1000)
        )
        transformer_cfg = components_cfg.get("transformer", {}).get(
            "config_dict", {}
        )
        self.expand_timesteps = bool(
            transformer_cfg.get("expand_timesteps", False)
        )

        device = self.transformer.devices[0]
        z_dim = int(self.vae.config.z_dim)
        mean_arr = np.asarray(
            self.vae.config.latents_mean, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        std_arr = np.asarray(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        self._vae_mean_buf = Buffer.from_numpy(mean_arr).to(device)
        self._vae_std_buf = Buffer.from_numpy(std_arr).to(device)

    def _setup_moe(self) -> None:
        """Load optional transformer_2 and configure MoE mode."""
        self._moe_dual_loaded = False
        self._active_transformer_weights = "primary"

        relative_paths = self._resolve_relative_component_paths()
        if "transformer_2" not in relative_paths:
            self.transformer_2 = None
            return

        # Load transformer_2 weights (lazy — not compiled yet).
        diffusers_config = self.pipeline_config.model.diffusers_config or {}
        components_cfg = diffusers_config.get("components", {})
        config_dict = self._get_component_config_dict(
            components_cfg, "transformer_2"
        )
        abs_paths = self._resolve_absolute_paths(
            self._weight_paths, relative_paths["transformer_2"]
        )
        lora_files = self._resolve_lora_files()
        lora_scale_2 = self._lora_config.get(
            "scale_2", self._lora_config.get("scale", 1.0)
        )
        t2_kwargs: dict[str, Any] = {
            "config": config_dict,
            "encoding": self.pipeline_config.model.quantization_encoding,
            "devices": self.devices,
            "weights": load_weights(abs_paths),
            "session": self.session,
            "eager_load": False,
        }
        lora_path_2 = lora_files.get("low_noise_model")
        if lora_path_2 is not None:
            t2_kwargs["lora_path"] = lora_path_2
            t2_kwargs["lora_scale"] = lora_scale_2
        with Tracer("load_component:transformer_2"):
            self.transformer_2 = WanTransformerModel(**t2_kwargs)

        # Try dual-load (both models on GPU), fall back to weight-swap.
        self.transformer_2.prepare_state_dict()
        if self._try_dual_load_transformer2():
            self._moe_dual_loaded = True
            logger.info(
                "MoE dual-load enabled: transformer_2 will stay resident "
                "without weight swapping"
            )
        else:
            # Weight-swap: secondary weights loaded lazily on first swap
            # to keep init VRAM usage low for symbolic block graphs.
            logger.info("MoE swap mode: transformer_2 will use weight swap")

    def init_remaining_components(self) -> None:
        """Initialize VAE config, MoE, and compile runtime graphs."""
        self._setup_vae_config()
        self._setup_moe()

        # Compile transformer for the default resolution.
        # TODO(compiler): use symbolic seq_len once engine OOM is fixed.
        h, w, nf = self.default_resolution
        seq_len = self._compute_seq_len(h, w, nf)
        self.transformer.load_model(
            seq_text_len=self.embed_seq_len,
            seq_len=seq_len,
        )

        self.build_guidance()
        self.build_unipc_step()
        self.build_duplicate_cfg_latents()
        self.build_duplicate_cfg_timesteps()
        self.build_concat_cfg_prompt_embeddings()
        self.build_split_cfg_predictions()
        self.build_cast_f32_to_model_dtype()
        self.build_cast_model_dtype_to_f32()
        self.build_denorm()

        self.cache: WanRuntimeCache = WanRuntimeCache()

    def build_guidance(self) -> None:
        """Compile classifier-free guidance: uncond + scale * (cond - uncond)."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
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

        self.__dict__["guidance"] = max_compile(
            self._guidance_model,
            input_types=input_types,
        )

    def build_unipc_step(self) -> None:
        """Compile a single on-device UniPC update step for Wan."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
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
            latent_type_model,  # model_output (model dtype, e.g. bf16)
            latent_type_f32,  # last_sample
            latent_type_f32,  # prev_model_output
            latent_type_f32,  # older_model_output
            coeff_type,
        ]
        self.__dict__["unipc_step"] = max_compile(
            self._tensor_unipc_step_model,
            input_types=input_types,
        )

    def build_duplicate_cfg_latents(self) -> None:
        """Compile CFG latent duplication helper."""
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        self.__dict__["duplicate_cfg_latents"] = max_compile(
            self._duplicate_batch,
            input_types=[
                TensorType(
                    dtype,
                    shape=[
                        1,
                        self.transformer.config.in_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=device,
                )
            ],
        )

    def build_duplicate_cfg_timesteps(self) -> None:
        device = self.transformer.devices[0]
        self.__dict__["duplicate_cfg_timesteps"] = max_compile(
            self._duplicate_batch,
            input_types=[TensorType(DType.float32, shape=[1], device=device)],
        )

    def build_concat_cfg_prompt_embeddings(self) -> None:
        device = self.transformer.devices[0]
        self.__dict__["concat_cfg_prompt_embeddings"] = max_compile(
            self._concat_batch_pair,
            input_types=[
                TensorType(
                    self.text_encoder.config.dtype,
                    shape=[1, "seq_text", self.transformer.config.text_dim],
                    device=device,
                ),
                TensorType(
                    self.text_encoder.config.dtype,
                    shape=[1, "seq_text", self.transformer.config.text_dim],
                    device=device,
                ),
            ],
        )

    def build_split_cfg_predictions(self) -> None:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        self.__dict__["split_cfg_predictions"] = max_compile(
            self._split_cfg_predictions,
            input_types=[
                TensorType(
                    dtype,
                    shape=[
                        2,
                        self.transformer.config.out_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=device,
                )
            ],
        )

    def build_cast_f32_to_model_dtype(self) -> None:
        """Compile float32 -> model dtype cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_f32_to_mdtype",
            input_types=[TensorType(DType.float32, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, model_dtype))
        self.__dict__["_cast_f32_to_model_dtype"] = self.session.load(g)

    def build_cast_model_dtype_to_f32(self) -> None:
        """Compile model dtype -> float32 cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        latent_5d = ["batch", "channels", "frames", "height", "width"]

        with Graph(
            "wan_cast_mdtype_to_f32",
            input_types=[TensorType(model_dtype, latent_5d, device=device)],
        ) as g:
            g.output(ops.cast(g.inputs[0].tensor, DType.float32))
        self.__dict__["_cast_model_dtype_to_f32"] = self.session.load(g)

    def build_denorm(self) -> None:
        """Compile VAE latent denormalization + dtype cast graph."""
        device = self.transformer.devices[0]
        model_dtype = self.transformer.config.dtype
        z_dim = int(self.vae.config.z_dim)
        input_types = [
            TensorType(
                DType.float32,
                ["batch", z_dim, "f", "h", "w"],
                device=device,
            ),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
            TensorType(DType.float32, [1, z_dim, 1, 1, 1], device=device),
        ]
        with Graph("wan_denorm", input_types=input_types) as g:
            latents, std, mean = (v.tensor for v in g.inputs)
            result = ops.cast(latents * std + mean, model_dtype)
            g.output(result)
        self.__dict__["_denorm"] = self.session.load(g)

    @staticmethod
    def _duplicate_batch(value: Any) -> Any:
        return ops.concat([value, value], axis=0)

    @staticmethod
    def _concat_batch_pair(first_value: Any, second_value: Any) -> Any:
        return ops.concat([first_value, second_value], axis=0)

    @staticmethod
    def _split_cfg_predictions(
        batched_predictions: Any,
    ) -> tuple[Any, Any]:
        positive_prediction = ops.slice_tensor(
            batched_predictions,
            [
                slice(0, 1),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
            ],
        )
        negative_prediction = ops.slice_tensor(
            batched_predictions,
            [
                slice(1, 2),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
            ],
        )
        return positive_prediction, negative_prediction

    def _guidance_model(
        self, noise_pred: Any, noise_uncond: Any, scale: Any
    ) -> Any:
        return noise_uncond + scale * (noise_pred - noise_uncond)

    def _get_guidance_scale(
        self,
        value: float,
        *,
        dtype: DType,
        device: Device,
    ) -> Buffer:
        key = (float(value), dtype, str(device.id))
        cached = self.cache.guidance_scales.get(key)
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
        self.cache.guidance_scales[key] = scale
        return scale

    def _tensor_unipc_step_model(
        self,
        sample: Any,
        model_output: Any,
        last_sample: Any,
        prev_model_output: Any,
        older_model_output: Any,
        coeffs: Any,
    ) -> tuple[Any, Any, Any]:
        # Cast model_output from model dtype (bf16) to float32
        model_output = ops.cast(model_output, DType.float32)

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
        return previous_sample, converted, corrected_sample

    def prepare_inputs(self, context: PixelGenerationContext) -> WanModelInputs:
        num_frames = 81
        if hasattr(context, "num_frames") and context.num_frames is not None:
            num_frames = int(context.num_frames)

        model_inputs = WanModelInputs(
            tokens=context.tokens,
            negative_tokens=getattr(context, "negative_tokens", None),
            timesteps=np.asarray(
                getattr(context, "timesteps", []), dtype=np.float32
            ),
            latents=np.asarray(
                getattr(context, "latents", []), dtype=np.float32
            ),
            width=getattr(context, "width", 832),
            height=getattr(context, "height", 480),
            num_frames=num_frames,
            num_inference_steps=getattr(context, "num_inference_steps", 50),
            guidance_scale=getattr(context, "guidance_scale", 5.0),
            num_images_per_prompt=getattr(context, "num_images_per_prompt", 1),
            step_coefficients=getattr(context, "step_coefficients", None),
            boundary_timestep=getattr(context, "boundary_timestep", None),
            input_image=getattr(context, "input_image", None),
        )

        if model_inputs.latents.ndim == 5:
            latent_frames = int(model_inputs.latents.shape[2])
            model_inputs.num_frames = self._normalize_num_frames_for_wan(
                requested_num_frames=num_frames,
                latent_frames=latent_frames,
            )

        if (
            hasattr(context, "guidance_scale_2")
            and context.guidance_scale_2 is not None
        ):
            model_inputs.guidance_scale_2 = context.guidance_scale_2

        return model_inputs

    def scheduler_step(
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
                latents.device.to_device()
                if hasattr(latents.device, "to_device")
                else latents.device
            )
            last_sample = zero
            prev_model_output = zero
            older_model_output = zero

        assert prev_model_output is not None
        assert older_model_output is not None
        assert last_sample is not None
        previous_sample, converted, corrected_sample = self.unipc_step(
            latents,
            noise_pred,
            last_sample,
            prev_model_output,
            older_model_output,
            coeffs,
        )
        next_prev_model_output = getattr(converted, "driver_tensor", converted)
        next_last_sample = getattr(
            corrected_sample, "driver_tensor", corrected_sample
        )
        next_latents = getattr(
            previous_sample, "driver_tensor", previous_sample
        )
        return next_latents, (
            next_last_sample,
            next_prev_model_output,
            prev_model_output,
        )

    def decode_latents(
        self,
        latents: Buffer,
        num_frames: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Denormalize latents and decode through VAE."""
        logger.info("Decoding Wan output")
        denorm_latents = self._denormalize_vae_latents(latents)
        decoded_video = self.vae.decode_5d(denorm_latents)
        decoded_np = self._buffer_to_numpy_f32(
            decoded_video, dtype=decoded_video.dtype
        )
        target_num_frames = min(decoded_np.shape[2], num_frames)
        return decoded_np[:, :, :target_num_frames, :height, :width]

    # Diffusers pads tokens to 512 but trims final embeddings to 226 for
    # cross-attention.  Subclasses can override for different models.
    embed_seq_len: int = 226

    # Default resolution for block graph compilation (height, width, frames).
    # TODO(compiler): remove once symbolic seq_len is supported.
    default_resolution: tuple[int, int, int] = (720, 1280, 81)

    def _compute_seq_len(self, height: int, width: int, num_frames: int) -> int:
        """Compute the latent sequence length for a given resolution."""
        p_t, p_h, p_w = self.transformer.config.patch_size
        ls = self.compute_video_latent_shape(
            batch_size=1,
            z_dim=int(self.vae.config.z_dim),
            num_frames=num_frames,
            height=height,
            width=width,
            scale_factor_temporal=self.vae_scale_factor_temporal,
            scale_factor_spatial=self.vae_scale_factor_spatial,
        )
        return (ls[2] // p_t) * (ls[3] // p_h) * (ls[4] // p_w)

    # Standard resolutions to pre-compile block graphs for.
    def prepare_prompt_embeddings(
        self,
        model_inputs: WanModelInputs,
    ) -> tuple[Buffer, Buffer | None, bool]:
        """Encode positive and optional negative prompts via T5."""
        logger.info("Preparing Wan prompt embeddings")
        max_seq_len = self.embed_seq_len
        prompt_embeds = self._get_t5_prompt_embeds(
            tokens=model_inputs.tokens,
            attention_mask=model_inputs.mask,
            num_videos_per_prompt=model_inputs.num_images_per_prompt,
            max_sequence_length=max_seq_len,
        )
        do_cfg = (
            model_inputs.guidance_scale > 1.0
            and model_inputs.negative_tokens is not None
        )
        negative_prompt_embeds: Buffer | None = None
        if do_cfg and model_inputs.negative_tokens is not None:
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                tokens=model_inputs.negative_tokens,
                attention_mask=model_inputs.negative_mask,
                num_videos_per_prompt=model_inputs.num_images_per_prompt,
                max_sequence_length=max_seq_len,
            )
        return prompt_embeds, negative_prompt_embeds, do_cfg

    @traced(message="WanPipeline.execute")
    def execute(  # type: ignore[override]
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        """Run the full Wan diffusion pipeline."""
        del kwargs
        device = self.transformer.devices[0]

        # 1. Encode prompts.
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds, negative_prompt_embeds, do_cfg = (
                self.prepare_prompt_embeddings(model_inputs)
            )

        # 2. Prepare latents.
        with Tracer("preprocess_latents"):
            logger.info("Preparing Wan latents")
            latents = Buffer.from_numpy(
                np.ascontiguousarray(model_inputs.latents, dtype=np.float32)
            ).to(device)

        # 3. Prepare scheduler state.
        with Tracer("prepare_scheduler"):
            if model_inputs.step_coefficients is None:
                raise ValueError(
                    "WanPipeline requires precomputed step_coefficients."
                )
            timesteps = np.ascontiguousarray(
                model_inputs.timesteps, dtype=np.float32
            )
            boundary_timestep = model_inputs.boundary_timestep
            if boundary_timestep is None and self.boundary_ratio is not None:
                boundary_timestep = (
                    self.boundary_ratio * self.num_train_timesteps
                )

            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]),
                height=int(latents.shape[3]),
                width=int(latents.shape[4]),
            )
            batched_timesteps = self._get_batched_timesteps(
                scheduler_timesteps=timesteps,
                batch_size=int(latents.shape[0]),
                device=device,
            )
            coeff_buffers = [
                Buffer.from_numpy(
                    np.ascontiguousarray(row, dtype=np.float32)
                ).to(device)
                for row in model_inputs.step_coefficients
            ]

            # Guidance scales.
            guidance_scale_high: Buffer | None = None
            guidance_scale_low: Buffer | None = None
            if do_cfg:
                guidance_scale_high = self._get_guidance_scale(
                    float(model_inputs.guidance_scale),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                guidance_scale_low = self._get_guidance_scale(
                    float(
                        model_inputs.guidance_scale_2
                        if model_inputs.guidance_scale_2 is not None
                        else model_inputs.guidance_scale
                    ),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )

            # MoE boundary.
            has_moe = (
                self.transformer_2 is not None and boundary_timestep is not None
            )
            boundary_step_idx = len(timesteps)
            if boundary_timestep is not None:
                for idx, t in enumerate(timesteps):
                    if float(t) < boundary_timestep:
                        boundary_step_idx = idx
                        break

            p_t, p_h, p_w = self.transformer.config.patch_size
            spatial_shape = self._get_spatial_shape(
                int(latents.shape[2]) // p_t,
                int(latents.shape[3]) // p_h,
                int(latents.shape[4]) // p_w,
                device,
            )

        # 4. Denoising loop.
        with Tracer("denoising_loop"):
            step_state: WanUniPCState = (None, None, None)
            if not self._moe_dual_loaded:
                self._activate_transformer_weights(use_secondary=False)

            # High-noise phase (or full denoising if no MoE).
            latents, step_state = self._run_denoising_phase(
                latents=latents,
                transformer_model=self.transformer,
                prompt_embeds=prompt_embeds,
                batched_prompt_embeds=None,
                negative_prompt_embeds=negative_prompt_embeds,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                batched_timesteps=batched_timesteps,
                coeff_buffers=coeff_buffers,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale_high,
                step_range=range(boundary_step_idx),
                desc="Denoising (high-noise)" if has_moe else "Denoising",
                spatial_shape=spatial_shape,
                step_state=step_state,
            )

            # Low-noise phase (MoE only).
            if has_moe and boundary_step_idx < len(batched_timesteps):
                if self._moe_dual_loaded:
                    low_noise_model = self.transformer_2
                else:
                    self._activate_transformer_weights(use_secondary=True)
                    low_noise_model = self.transformer
                latents, _ = self._run_denoising_phase(
                    latents=latents,
                    transformer_model=low_noise_model,
                    prompt_embeds=prompt_embeds,
                    batched_prompt_embeds=None,
                    negative_prompt_embeds=negative_prompt_embeds,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    batched_timesteps=batched_timesteps,
                    coeff_buffers=coeff_buffers,
                    do_cfg=do_cfg,
                    guidance_scale=guidance_scale_low,
                    step_range=range(boundary_step_idx, len(batched_timesteps)),
                    desc="Denoising (low-noise)",
                    spatial_shape=spatial_shape,
                    step_state=step_state,
                )

        # 5. Decode.
        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                int(model_inputs.num_frames),
                model_inputs.height,
                model_inputs.width,
            )
        return WanPipelineOutput(images=images)

    def _run_denoising_phase(
        self,
        latents: Buffer,
        transformer_model: Any,
        prompt_embeds: Buffer,
        batched_prompt_embeds: Buffer | None,
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
    ) -> tuple[Buffer, WanUniPCState]:
        progress = tqdm(  # type: ignore[call-arg]
            step_range,
            desc=desc,
            leave=True,
            disable=not sys.stderr.isatty(),
        )
        for i in progress:  # type: ignore[attr-defined]
            with Tracer(f"{desc}:step_{i}"):
                dit_timestep = batched_timesteps[i]
                latent_model_input = self._cast_f32_to_model_dtype.execute(
                    latents
                )[0]
                with Tracer("transformer"):
                    noise_pred_buf = self._run_transformer_forward(
                        transformer_model=transformer_model,
                        latent_model_input=latent_model_input,
                        dit_timestep=dit_timestep,
                        prompt_embeds=prompt_embeds,
                        batched_prompt_embeds=batched_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        spatial_shape=spatial_shape,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                    )
                with Tracer("scheduler_step"):
                    latents, step_state = self.scheduler_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )
        return latents, step_state

    def _run_transformer_forward(
        self,
        *,
        transformer_model: Any,
        latent_model_input: Buffer,
        dit_timestep: Buffer,
        prompt_embeds: Buffer,
        batched_prompt_embeds: Buffer | None,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        do_cfg: bool,
        guidance_scale: Buffer | None,
    ) -> Buffer:
        """Run transformer + optional CFG guidance, return noise prediction."""
        if (
            do_cfg
            and batched_prompt_embeds is not None
            and negative_prompt_embeds is not None
        ):
            duplicated_latents = self.duplicate_cfg_latents(latent_model_input)
            duplicated_timesteps = self.duplicate_cfg_timesteps(dit_timestep)
            batched_predictions = transformer_model(
                duplicated_latents,
                duplicated_timesteps,
                batched_prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
            batched_predictions = getattr(
                batched_predictions, "driver_tensor", batched_predictions
            )
            positive, negative = self.split_cfg_predictions(batched_predictions)
            assert guidance_scale is not None
            guided = self.guidance(positive, negative, guidance_scale)
            return getattr(guided, "driver_tensor", guided)

        noise_pred_buf = transformer_model(
            latent_model_input,
            dit_timestep,
            prompt_embeds,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
        noise_pred_buf = getattr(
            noise_pred_buf, "driver_tensor", noise_pred_buf
        )

        if (
            do_cfg
            and batched_prompt_embeds is None
            and negative_prompt_embeds is not None
        ):
            assert guidance_scale is not None
            noise_uncond_buf = transformer_model(
                latent_model_input,
                dit_timestep,
                negative_prompt_embeds,
                rope_cos,
                rope_sin,
                spatial_shape,
            )
            noise_uncond_buf = getattr(
                noise_uncond_buf, "driver_tensor", noise_uncond_buf
            )
            guided = self.guidance(
                noise_pred_buf,
                noise_uncond_buf,
                guidance_scale,
            )
            return getattr(guided, "driver_tensor", guided)

        return noise_pred_buf

    def _get_spatial_shape(
        self, ppf: int, pph: int, ppw: int, device: Device
    ) -> Buffer:
        key = f"{ppf}_{pph}_{ppw}_{device.id}"
        cached = self.cache.spatial_shapes.get(key)
        if cached is not None:
            return cached
        spatial_np = np.zeros((ppf, pph, ppw), dtype=np.int8)
        spatial_shape = Buffer.from_numpy(spatial_np).to(device)
        self.cache.spatial_shapes[key] = spatial_shape
        return spatial_shape

    def _get_batched_timesteps(
        self,
        scheduler_timesteps: np.ndarray,
        batch_size: int,
        device: Device,
    ) -> list[Buffer]:
        key = (
            f"{batch_size}_{len(scheduler_timesteps)}_"
            f"{float(scheduler_timesteps[0]):.4f}_{float(scheduler_timesteps[-1]):.4f}_"
            f"{device.id}"
        )
        cached = self.cache.batched_timesteps.get(key)
        if cached is not None:
            return cached

        batched_timesteps = [
            Buffer.from_numpy(
                np.full([batch_size], float(step_value), dtype=np.float32)
            ).to(device)
            for step_value in scheduler_timesteps
        ]
        self.cache.batched_timesteps[key] = batched_timesteps
        return batched_timesteps

    def _get_t5_prompt_embeds(
        self,
        tokens: TokenBuffer,
        attention_mask: npt.NDArray[np.bool_] | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> Buffer:
        """Run T5 encoder and post-process hidden states into padded embeddings."""
        token_ids = tokens.array
        if token_ids.ndim == 1:
            token_ids = np.expand_dims(token_ids, axis=0)
        if attention_mask is None:
            attention_mask = token_ids != 0
        if attention_mask.ndim == 1:
            attention_mask = np.expand_dims(attention_mask, axis=0)

        device = self.text_encoder.devices[0]
        text_input_ids = Buffer.from_dlpack(
            np.ascontiguousarray(token_ids, dtype=np.int64)
        ).to(device)
        text_attention_mask = Buffer.from_dlpack(
            np.ascontiguousarray(attention_mask.astype(np.int64, copy=False))
        ).to(device)
        raw = self.text_encoder(text_input_ids, text_attention_mask)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        hidden_states = getattr(raw, "driver_tensor", raw)

        # Post-process: pad/truncate to max_sequence_length, repeat for batch.
        batch_size = int(hidden_states.shape[0])
        hidden_dim = int(hidden_states.shape[2])
        hidden_np = self._buffer_to_numpy_f32(
            hidden_states, dtype=hidden_states.dtype
        ).reshape(batch_size, int(hidden_states.shape[1]), hidden_dim)
        mask_np = np.from_dlpack(text_attention_mask.to(CPU())).reshape(
            batch_size, int(text_attention_mask.shape[1])
        )

        embeds_np = np.zeros(
            (batch_size, max_sequence_length, hidden_dim), dtype=np.float32
        )
        for b in range(batch_size):
            seq_len = min(
                int(mask_np[b].sum()), hidden_np.shape[1], max_sequence_length
            )
            embeds_np[b, :seq_len, :] = hidden_np[b, :seq_len, :]

        if num_videos_per_prompt > 1:
            embeds_np = np.repeat(embeds_np, num_videos_per_prompt, axis=0)

        out_device = (
            hidden_states.device.to_device()
            if hasattr(hidden_states.device, "to_device")
            else hidden_states.device
        )
        if hidden_states.dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(np.ascontiguousarray(embeds_np))
            return (
                Buffer.from_numpy(u16)
                .to(out_device)
                .view(dtype=DType.bfloat16, shape=embeds_np.shape)
            )
        return Buffer.from_numpy(np.ascontiguousarray(embeds_np)).to(out_device)

    @staticmethod
    def _buffer_to_numpy_f32(
        value: Buffer,
        *,
        dtype: DType,
    ) -> np.ndarray:
        """Convert a Buffer to float32 numpy array (handles bfloat16)."""
        cpu_value = value.to(CPU())
        if dtype == DType.bfloat16:
            cpu_u16 = np.from_dlpack(
                cpu_value.view(dtype=DType.uint16, shape=cpu_value.shape)
            )
            return (cpu_u16.astype(np.uint32) << 16).view(np.float32)
        return np.from_dlpack(cpu_value).astype(np.float32, copy=False)

    def _denormalize_vae_latents(self, latents: Buffer) -> Buffer:
        """Denormalize latents using compiled denorm model (f32 in, model_dtype out)."""
        result = self._denorm.execute(
            latents, self._vae_std_buf, self._vae_mean_buf
        )
        return result[0]

    def _normalize_num_frames_for_wan(
        self,
        requested_num_frames: int,
        latent_frames: int,
    ) -> int:
        compatible_num_frames = max(
            1,
            (max(latent_frames, 1) - 1) * self.vae_scale_factor_temporal + 1,
        )
        if requested_num_frames <= compatible_num_frames:
            return requested_num_frames

        logger.warning(
            "Requested Wan num_frames=%d is incompatible with latent temporal "
            "shape (%d latent frames). Auto-adjusting output frame count to %d.",
            requested_num_frames,
            latent_frames,
            compatible_num_frames,
        )
        return compatible_num_frames

    @staticmethod
    def compute_video_latent_shape(
        *,
        batch_size: int,
        z_dim: int,
        num_frames: int,
        height: int,
        width: int,
        scale_factor_temporal: int,
        scale_factor_spatial: int,
    ) -> tuple[int, int, int, int, int]:
        adjusted_num_frames = max(1, int(num_frames))
        if adjusted_num_frames > 1:
            remainder = (adjusted_num_frames - 1) % scale_factor_temporal
            if remainder != 0:
                adjusted_num_frames += scale_factor_temporal - remainder

        latent_frames = (adjusted_num_frames - 1) // scale_factor_temporal + 1
        latent_height = 2 * (int(height) // (scale_factor_spatial * 2))
        latent_width = 2 * (int(width) // (scale_factor_spatial * 2))
        return (
            int(batch_size),
            int(z_dim),
            int(latent_frames),
            int(latent_height),
            int(latent_width),
        )

    @staticmethod
    def _get_free_vram_bytes() -> int | None:
        """Query free GPU VRAM in bytes via nvidia-smi."""
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            # First GPU, value in MiB
            return int(out.strip().split("\n")[0]) * 1024 * 1024
        except Exception:
            return None

    def _try_dual_load_transformer2(self) -> bool:
        """Try to compile transformer_2 on GPU if VRAM is sufficient.

        Estimates required VRAM as the primary transformer's weight size
        (both models have identical architecture). Returns True if
        transformer_2 was successfully compiled on GPU.
        """
        if self.transformer_2 is None:
            return False

        free_vram = self._get_free_vram_bytes()
        if free_vram is None:
            return False

        # Estimate required VRAM as the primary transformer's weight size.
        primary_sd = self.transformer.prepare_state_dict()
        estimated_bytes = 0
        for v in primary_sd.values():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                num_elements = 1
                for d in v.shape:
                    num_elements *= d
                estimated_bytes += num_elements * 2  # bfloat16

        margin = 1.2  # 20% headroom
        if free_vram < estimated_bytes * margin:
            logger.info(
                "Insufficient VRAM for dual load: need %.1f GB, free %.1f GB",
                estimated_bytes * margin / 1e9,
                free_vram / 1e9,
            )
            return False

        # Load transformer_2 using the same compiled block graphs.
        self.transformer_2.load_model(
            seq_text_len=self.embed_seq_len,
            seq_len=self._compute_seq_len(*self.default_resolution),
        )
        return True

    def _activate_transformer_weights(self, *, use_secondary: bool) -> None:
        if not use_secondary:
            if self._active_transformer_weights != "primary":
                self.transformer.reload_model_weights()
                self._active_transformer_weights = "primary"
            return

        if self.transformer_2 is None:
            return
        if self._active_transformer_weights != "secondary":
            self.transformer.reload_model_weights(
                self.transformer_2.prepare_state_dict()
            )
            self._active_transformer_weights = "secondary"

    @staticmethod
    def denormalize_vae_latents(
        latents_np: np.ndarray,
        latents_mean: list[float],
        latents_std: list[float],
        z_dim: int,
    ) -> np.ndarray:
        """Denormalize VAE latents in numpy (used by external callers)."""
        mean = np.asarray(latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        std = np.asarray(latents_std, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        return latents_np * std + mean

    @staticmethod
    def _to_numpy(image: Buffer) -> np.ndarray:
        return WanPipeline._buffer_to_numpy_f32(image, dtype=image.dtype)
