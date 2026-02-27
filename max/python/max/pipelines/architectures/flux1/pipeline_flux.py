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

from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.interfaces import PixelGenerationContext, TokenBuffer
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from tqdm import tqdm

from ..autoencoders import AutoencoderKLModel
from ..clip import ClipModel
from ..t5 import T5Model
from .model import Flux1TransformerModel


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
        images (`np.ndarray` or `Tensor`)
            Numpy array or Max tensor of shape `(batch_size, height, width, num_channels)`.
            The denoised images of the diffusion pipeline.
    """

    images: np.ndarray | Tensor


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

        # Tensor caches.
        self._cached_guidance: dict[str, Tensor] = {}
        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}

    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> FluxModelInputs:
        return FluxModelInputs.from_context(context)

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

        text_input_ids = Tensor.constant(
            tokens.array, dtype=DType.int64, device=self.text_encoder.devices[0]
        )
        text_input_ids_2 = Tensor.constant(
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
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        if output_type == "latent":
            return latents
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
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    def execute(  # type: ignore[override]
        self,
        model_inputs: FluxModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> FluxPipelineOutput:
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

        for i in tqdm(range(num_timesteps), desc="Denoising"):
            timestep = timesteps_seq[i : i + 1]
            dt = dts_seq[i : i + 1]

            noise_pred = self.transformer(
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                timestep,
                latent_image_ids,
                text_ids,
                guidance,
            )[0]

            if model_inputs.do_true_cfg:
                assert negative_prompt_embeds is not None
                assert negative_pooled_prompt_embeds is not None
                assert negative_text_ids is not None
                neg_noise_pred = self.transformer(
                    latents,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    timestep,
                    latent_image_ids,
                    negative_text_ids,
                    guidance,
                )[0]

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
                    output_type=output_type,
                )
                callback_queue.put_nowait(image)

        # 3. Decode
        outputs = self.decode_latents(
            latents,
            model_inputs.height,
            model_inputs.width,
            output_type=output_type,
        )

        return FluxPipelineOutput(images=outputs)
