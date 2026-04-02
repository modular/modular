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

from dataclasses import MISSING, dataclass, field, fields
from queue import Queue
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.interfaces import (
    DenoisingCacheState,
    DiffusionPipeline,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    DiffusionPipelineOutput,
    max_compile,
)
from PIL import Image
from typing_extensions import Self

from ..autoencoders_modulev3 import AutoencoderKLModel
from ..clip import ClipModel
from ..t5 import T5Model
from .model import Flux1TransformerModel


@dataclass(kw_only=True)
class Flux1ModelInputs:
    """Input container for Flux1 pipeline execution."""

    tokens: TokenBuffer
    """Primary encoder token buffer."""

    tokens_2: TokenBuffer | None = None
    """Secondary encoder token buffer (for dual-encoder models)."""

    negative_tokens: TokenBuffer | None = None
    """Negative prompt tokens for the primary encoder."""

    negative_tokens_2: TokenBuffer | None = None
    """Negative prompt tokens for the secondary encoder."""

    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed denoising timestep schedule."""

    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed sigma schedule for denoising."""

    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Initial latent noise tensor."""

    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Latent image positional identifiers."""

    height: int = 1024
    """Output height in pixels."""

    width: int = 1024
    """Output width in pixels."""

    num_inference_steps: int = 50
    """Number of denoising steps."""

    guidance_scale: float = 3.5
    """Guidance scale for classifier-free guidance."""

    guidance: npt.NDArray[np.float32] | None = None
    """Optional precomputed guidance tensor."""

    true_cfg_scale: float = 1.0
    """True CFG scale (enabled when > 1.0 with negative prompt)."""

    num_warmup_steps: int = 0
    """Number of scheduler warmup steps."""

    num_images_per_prompt: int = 1
    """Number of images to generate per prompt."""

    input_image: Image.Image | None = None
    """Optional input image for image-to-image generation."""

    residual_threshold: Tensor | None = None
    """Scalar float32 tensor for FBCache residual threshold, on device.
    None when FBCache is not enabled."""

    @property
    def do_true_cfg(self) -> bool:
        return self.negative_tokens is not None

    def __post_init__(self) -> None:
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(
                f"height must be a positive int. Got {self.height!r}"
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(
                f"width must be a positive int. Got {self.width!r}"
            )
        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                f"num_inference_steps must be a positive int. Got {self.num_inference_steps!r}"
            )
        if (
            not isinstance(self.num_warmup_steps, int)
            or self.num_warmup_steps < 0
        ):
            raise ValueError(
                f"num_warmup_steps must be >= 0. Got {self.num_warmup_steps!r}"
            )

    @classmethod
    def from_context(cls, context: PixelGenerationContext) -> Self:
        """Build an instance from a PixelGenerationContext.

        If a context attribute is None, the dataclass default is used.
        """
        kwargs: dict[str, Any] = {}

        for dataclass_field in fields(cls):
            name = dataclass_field.name
            if not hasattr(context, name):
                continue
            v = getattr(context, name)

            if v is None:
                if dataclass_field.default is not MISSING:
                    kwargs[name] = dataclass_field.default
                elif dataclass_field.default_factory is not MISSING:
                    kwargs[name] = dataclass_field.default_factory()
                else:
                    kwargs[name] = None
            else:
                kwargs[name] = v

        return cls(**kwargs)


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

        self.build_prepare_prompt_embeddings()
        self.build_preprocess_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_decode_latents()

        self._transformer_device: Device = self.transformer.devices[0]
        self._guidance_embeds: bool = self.transformer.config.guidance_embeds

        self._init_cache_state(
            dtype=self.transformer.config.dtype,
            device=self.transformer.devices[0],
        )

        # Tensor caches.
        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def _make_rdt_tensor(
        self, request_value: float | None, device: Device
    ) -> Tensor | None:
        """Create a scalar float32 threshold tensor if FBCache is enabled."""
        if not self.cache_config.first_block_caching:
            return None
        value = (
            request_value
            if request_value is not None
            else self.default_residual_threshold
        )
        return Tensor(
            storage=Buffer.from_dlpack(np.array(value, dtype=np.float32)).to(
                device
            )
        )

    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> Flux1ModelInputs:
        model_inputs = Flux1ModelInputs.from_context(context)
        device = self._transformer_device
        model_inputs.residual_threshold = self._make_rdt_tensor(
            getattr(context, "residual_threshold", None), device
        )
        return model_inputs

    # -------------------------------------------------------------------------
    # Build methods (compile eager ops into MAX graphs)
    # -------------------------------------------------------------------------

    def build_prepare_prompt_embeddings(self) -> None:
        input_types = [
            TensorType(
                self.text_encoder_2.config.dtype,
                shape=["batch", "seq_len", "hidden_dim"],
                device=self.text_encoder_2.devices[0],
            ),
            TensorType(
                self.text_encoder_2.config.dtype,
                shape=["batch", "pooled_dim"],
                device=self.text_encoder_2.devices[0],
            ),
        ]
        self.__dict__["_prepare_prompt_embeddings"] = max_compile(
            self._prepare_prompt_embeddings,
            input_types=input_types,
        )

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                DType.float32,
                shape=["batch", "channels", "height", 2, "width", 2],
                device=device,
            ),
        ]
        self.__dict__["_pack_latents"] = max_compile(
            self._pack_latents,
            input_types=input_types,
        )

    def build_prepare_scheduler(self) -> None:
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=input_types,
        )

    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(
                dtype, shape=["batch", "seq", "channels"], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_decode_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        input_types = [
            TensorType(
                dtype,
                shape=["batch", "half_h", "half_w", "ch_4", 2, 2],
                device=device,
            ),
        ]
        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=input_types,
        )

    # -------------------------------------------------------------------------
    # Compiled inner methods (run inside MAX graphs)
    # -------------------------------------------------------------------------

    def _prepare_prompt_embeddings(
        self, prompt_embeds: Tensor, pooled_prompt_embeds: Tensor
    ) -> tuple[Tensor, Tensor]:
        prompt_embeds = prompt_embeds.cast(prompt_embeds.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.cast(
            pooled_prompt_embeds.dtype
        )
        return prompt_embeds, pooled_prompt_embeds

    def _pack_latents(self, latents: Tensor) -> Tensor:
        """Pack 6D latents (B,C,H//2,2,W//2,2) into sequence (B,H//2*W//2,C*4)."""
        latents = latents.cast(self.transformer.config.dtype)
        batch = latents.shape[0]
        c = latents.shape[1]
        h2 = latents.shape[2]
        w2 = latents.shape[4]
        latents = F.permute(latents, (0, 2, 4, 1, 3, 5))
        latents = F.reshape(latents, (batch, h2 * w2, c * 4))
        return latents

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute timesteps and dt values from sigmas in a single fused graph."""
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        all_timesteps = sigmas_curr.cast(DType.float32)
        return all_timesteps, all_dt

    def scheduler_step(
        self, latents: Tensor, noise_pred: Tensor, dt: Tensor
    ) -> Tensor:
        """Apply a single Euler update step in sigma space."""
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        latents = latents + dt * noise_pred
        return latents.cast(latents_dtype)

    def _postprocess_latents(self, latents: Tensor) -> Tensor:
        """Unpack and denormalize 6D latents to (B, C//4, H, W)."""
        scaling_factor = self.vae.config.scaling_factor
        shift_factor = self.vae.config.shift_factor or 0.0
        batch = latents.shape[0]
        half_h = latents.shape[1]
        half_w = latents.shape[2]
        c_quarter = latents.shape[3]
        latents = F.permute(latents, (0, 3, 1, 4, 2, 5))
        latents = F.reshape(latents, (batch, c_quarter, half_h * 2, half_w * 2))
        latents = (latents / scaling_factor) + shift_factor
        return latents

    # -------------------------------------------------------------------------
    # Non-compiled wrappers and pipeline methods
    # -------------------------------------------------------------------------

    def prepare_prompt_embeddings(
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

        text_input_ids = Tensor(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        text_input_ids_2 = Tensor(
            tokens_2.array,
            dtype=DType.int64,
            device=self.text_encoder_2.devices[0],
        )

        # t5 embeddings
        prompt_embeds = self.text_encoder_2(text_input_ids_2)

        # clip embeddings
        clip_embeddings = self.text_encoder(text_input_ids)
        pooled_prompt_embeds = clip_embeddings[1]

        # Compiled dtype cast
        prompt_embeds, pooled_prompt_embeds = self._prepare_prompt_embeddings(
            prompt_embeds, pooled_prompt_embeds
        )

        bs_embed = int(prompt_embeds.shape[0])
        seq_len = int(prompt_embeds.shape[1])

        if num_images_per_prompt != 1:
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

        batch_size_final = bs_embed * num_images_per_prompt
        text_ids_key = f"{batch_size_final}_{seq_len}"
        if text_ids_key not in self._cached_text_ids:
            self._cached_text_ids[text_ids_key] = Tensor.zeros(
                (seq_len, 3),
                device=self.text_encoder_2.devices[0],
                dtype=prompt_embeds.dtype,
            )
        text_ids = self._cached_text_ids[text_ids_key]

        dtype = prompt_embeds.dtype
        device = prompt_embeds.device

        return (
            prompt_embeds,
            pooled_prompt_embeds.to(device).cast(dtype),
            text_ids.to(device).cast(dtype),
        )

    def preprocess_latents(
        self,
        latents_np: npt.NDArray[np.float32],
        latent_image_ids_np: npt.NDArray[np.float32],
    ) -> tuple[Tensor, Tensor]:
        device = self._transformer_device
        latents = Tensor.from_dlpack(latents_np).to(device)
        batch, c, h, w = map(int, latents.shape)
        latents = F.reshape(latents, (batch, c, h // 2, 2, w // 2, 2))
        latents = self._pack_latents(latents)
        latent_image_ids = Tensor.from_dlpack(latent_image_ids_np).to(device)
        return latents, latent_image_ids

    def decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
    ) -> npt.NDArray[np.uint8]:
        latents = Tensor.from_dlpack(latents)
        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])
        h = 2 * (height // (self.vae_scale_factor * 2))
        w = 2 * (width // (self.vae_scale_factor * 2))
        latents = F.reshape(
            latents, (batch_size, h // 2, w // 2, ch_size // 4, 2, 2)
        )
        latents = self._postprocess_latents(latents)
        return self._to_numpy(self.vae.decode(latents))

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        # Perform all post-processing on GPU before transfer:
        #   cast, denormalize [-1,1]->[0,1], clip, NCHW->NHWC, scale, uint8.
        # This shrinks the PCIe transfer 4x (float32 -> uint8).
        image = image.cast(DType.float32)
        image = image * 0.5 + 0.5
        image = image.clip(min=0.0, max=1.0)
        image = F.permute(image, (0, 2, 3, 1))  # NCHW -> NHWC
        image = image * 255.0
        image = image.cast(DType.uint8)
        cpu_image: Tensor = image.to(CPU())
        return np.from_dlpack(cpu_image)

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        base_args = (
            kwargs["latents"],
            kwargs["prompt_embeds"],
            kwargs["pooled_prompt_embeds"],
            kwargs["timestep"],
            kwargs["latent_image_ids"],
            kwargs["text_ids"],
            kwargs["guidance"],
        )
        if self.cache_config.teacache:
            return self.transformer(
                *base_args,
                teacache_prev_modulated_input=cache_state.teacache_prev_modulated_input,
                teacache_cached_residual=cache_state.teacache_cached_residual,
                teacache_accumulated_rel_l1=cache_state.teacache_accumulated_rel_l1,
                force_compute=kwargs["force_compute"],
            )
        if self.cache_config.first_block_caching:
            return self.transformer(
                *base_args,
                prev_residual=cache_state.prev_residual,
                prev_output=cache_state.prev_output,
                residual_threshold=kwargs.get("residual_threshold"),
            )
        return self.transformer(*base_args)

    def execute(  # type: ignore[override]
        self,
        model_inputs: Flux1ModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
    ) -> DiffusionPipelineOutput:
        """Execute the pipeline."""
        # 1. Encode prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            self.prepare_prompt_embeddings(
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
            ) = self.prepare_prompt_embeddings(
                tokens=model_inputs.negative_tokens,
                tokens_2=model_inputs.negative_tokens_2,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

        # 2. Prepare latents
        dtype = prompt_embeds.dtype
        latents, latent_image_ids = self.preprocess_latents(
            model_inputs.latents, model_inputs.latent_image_ids
        )
        latent_image_ids = latent_image_ids.cast(dtype)

        # 3. Guidance
        batch_size = int(latents.shape[0])
        guidance_key = f"{batch_size}_{model_inputs.guidance_scale}"
        if guidance_key not in self._cached_guidance:
            if self._guidance_embeds:
                self._cached_guidance[guidance_key] = Tensor.full(
                    [batch_size],
                    model_inputs.guidance_scale,
                    device=self._transformer_device,
                    dtype=dtype,
                )
            else:
                self._cached_guidance[guidance_key] = Tensor.zeros(
                    [batch_size],
                    device=self._transformer_device,
                    dtype=dtype,
                )
        guidance = self._cached_guidance[guidance_key]

        # 4. Scheduler
        sigmas_key = f"{model_inputs.num_inference_steps}"
        if sigmas_key not in self._cached_sigmas:
            self._cached_sigmas[sigmas_key] = Tensor.from_dlpack(
                model_inputs.sigmas
            ).to(self._transformer_device)
        sigmas = self._cached_sigmas[sigmas_key]
        all_timesteps, all_dts = self.prepare_scheduler(sigmas)

        # For faster tensor slicing inside the denoising loop.
        timesteps_seq: Any = all_timesteps
        dts_seq: Any = all_dts
        if hasattr(timesteps_seq, "driver_tensor"):
            timesteps_seq = timesteps_seq.driver_tensor
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        num_timesteps = int(model_inputs.sigmas.shape[0]) - 1

        timesteps: np.ndarray = model_inputs.timesteps
        num_timesteps = timesteps.shape[0]
        # Cache state initialization.
        dev = self.transformer.devices[0]
        image_seq_len = int(latents.shape[1])
        cache_pos = self.create_cache_state(
            batch_size, image_seq_len, self.transformer.config
        )
        cache_neg = (
            self.create_cache_state(
                batch_size, image_seq_len, self.transformer.config
            )
            if model_inputs.do_true_cfg
            else None
        )

        is_teacache = self.cache_config.teacache

        for i in range(num_timesteps):
            timestep = timesteps_seq[i : i + 1]
            dt = dts_seq[i : i + 1]

            step_kwargs: dict[str, Any] = dict(
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                timestep=timestep,
                latent_image_ids=latent_image_ids,
                text_ids=text_ids,
                guidance=guidance,
                residual_threshold=model_inputs.residual_threshold,
            )
            if is_teacache:
                step_kwargs["force_compute"] = Tensor(
                    storage=Buffer.from_dlpack(
                        np.array(
                            [i == 0 or i == num_timesteps - 1],
                            dtype=bool,
                        )
                    ).to(dev)
                )
                step_kwargs["num_inference_steps"] = num_timesteps

            noise_pred = self.run_denoising_step(
                step=i,
                cache_state=cache_pos,
                device=dev,
                **step_kwargs,
            )

            if model_inputs.do_true_cfg:
                assert negative_prompt_embeds is not None
                assert negative_pooled_prompt_embeds is not None
                assert negative_text_ids is not None
                assert cache_neg is not None

                neg_step_kwargs = dict(
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    timestep=timestep,
                    latent_image_ids=latent_image_ids,
                    text_ids=negative_text_ids,
                    guidance=guidance,
                    residual_threshold=model_inputs.residual_threshold,
                )
                if is_teacache:
                    neg_step_kwargs["force_compute"] = step_kwargs[
                        "force_compute"
                    ]
                    neg_step_kwargs["num_inference_steps"] = num_timesteps

                neg_noise_pred = self.run_denoising_step(
                    step=i,
                    cache_state=cache_neg,
                    device=dev,
                    **neg_step_kwargs,
                )
                noise_pred = neg_noise_pred + model_inputs.true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            # scheduler step
            latents = self.scheduler_step(latents, noise_pred, dt)

            if callback_queue is not None:
                image = self.decode_latents(
                    latents,
                    model_inputs.height,
                    model_inputs.width,
                )
                callback_queue.put_nowait(image)

        # 3. Decode
        outputs = self.decode_latents(
            latents,
            model_inputs.height,
            model_inputs.width,
        )

        return DiffusionPipelineOutput(images=outputs)
