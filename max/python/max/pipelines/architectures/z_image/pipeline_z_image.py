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

import hashlib
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import DiffusionPipeline, PixelModelInputs
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from max.profiler import Tracer, traced
from tqdm import tqdm

from ..autoencoders import AutoencoderKLModel
from ..qwen3.text_encoder import Qwen3TextEncoderZImageModel
from .model import ZImageTransformerModel


@dataclass(kw_only=True)
class ZImageModelInputs(PixelModelInputs):
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    mask: np.ndarray | None = None
    negative_mask: np.ndarray | None = None


@dataclass
class ZImagePipelineOutput:
    images: np.ndarray | Tensor


class ZImagePipeline(DiffusionPipeline):
    vae: AutoencoderKLModel
    text_encoder: Qwen3TextEncoderZImageModel
    transformer: ZImageTransformerModel

    components = {
        "vae": AutoencoderKLModel,
        "text_encoder": Qwen3TextEncoderZImageModel,
        "transformer": ZImageTransformerModel,
    }

    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        self.build_preprocess_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step()
        self.build_decode_latents()

        self._cached_text_ids: dict[str, Tensor] = {}
        self._cached_sigmas: dict[str, Tensor] = {}
        self._cached_img_ids: dict[str, Tensor] = {}
        self._cached_timesteps_batched: dict[str, Tensor] = {}

    def prepare_inputs(self, context: PixelContext) -> ZImageModelInputs:  # type: ignore[override]
        return ZImageModelInputs.from_context(context)

    def build_preprocess_latents(self) -> None:
        device = self.transformer.devices[0]
        self.__dict__["_pack_latents_from_6d"] = max_compile(
            self._pack_latents_from_6d,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["batch", "channels", "height", 2, "width", 2],
                    device=device,
                ),
            ],
        )

    def build_prepare_scheduler(self) -> None:
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=[
                TensorType(
                    DType.float32,
                    shape=["num_sigmas"],
                    device=self.transformer.devices[0],
                ),
            ],
        )

    def build_scheduler_step(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["scheduler_step"] = max_compile(
            self.scheduler_step,
            input_types=[
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(
                    dtype, shape=["batch", "seq", "channels"], device=device
                ),
                TensorType(DType.float32, shape=[1], device=device),
            ],
        )

    def build_decode_latents(self) -> None:
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        self.__dict__["_postprocess_latents"] = max_compile(
            self._postprocess_latents,
            input_types=[
                TensorType(
                    dtype,
                    shape=["batch", "half_h", "half_w", 2, 2, "ch_4"],
                    device=device,
                ),
            ],
        )

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        batch_size, num_channels, height, width = map(int, latents.shape)
        latents = F.reshape(
            latents,
            (batch_size, num_channels, height // 2, 2, width // 2, 2),
        )
        # Match diffusers Z-Image patchify order: (pH, pW, C) inside each token.
        latents = F.permute(latents, (0, 2, 4, 3, 5, 1))
        latents = F.reshape(
            latents,
            (
                batch_size,
                (height // 2) * (width // 2),
                num_channels * 4,
            ),
        )
        return latents

    @staticmethod
    def _pack_latents_from_6d(latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        num_channels = latents.shape[1]
        height = latents.shape[2]
        width = latents.shape[4]
        # Match diffusers Z-Image patchify order: (pH, pW, C) inside each token.
        latents = F.permute(latents, (0, 2, 4, 3, 5, 1))
        latents = F.reshape(
            latents,
            (
                batch_size,
                height * width,
                num_channels * 4,
            ),
        )
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> Tensor:
        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])

        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        h2 = height // 2
        w2 = width // 2
        latents = F.reshape(
            latents,
            (batch_size, h2, w2, 2, 2, ch_size // 4),
        )
        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents,
            (batch_size, ch_size // 4, height, width),
        )
        return latents

    @traced
    def _prepare_prompt_embeddings(
        self,
        tokens: np.ndarray,
        mask: np.ndarray | None,
        num_images_per_prompt: int,
    ) -> Tensor:
        if tokens.ndim == 2:
            tokens = tokens[0]
        selected_tokens = tokens

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[0]
            if mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    "Z-Image mask length must match token length after tokenizer preprocessing. "
                    f"Got mask={mask.shape[0]}, tokens={tokens.shape[0]}."
                )
            selected_mask = mask.astype(np.bool_, copy=False)
            if not np.any(selected_mask):
                raise ValueError("Z-Image mask cannot mask out all tokens.")
            if not np.all(selected_mask):
                selected_tokens = tokens[selected_mask]

        text_input_ids = Tensor.constant(
            selected_tokens,
            dtype=DType.int64,
            device=self.text_encoder.devices[0],
        )
        prompt_embeds = self.text_encoder(text_input_ids)
        if prompt_embeds.rank == 2:
            prompt_embeds = F.unsqueeze(prompt_embeds, axis=0)
        elif prompt_embeds.rank != 3:
            raise ValueError(
                f"Unexpected prompt_embeds rank={prompt_embeds.rank}; expected 2 or 3."
            )
        if num_images_per_prompt > 1:
            prompt_embeds = F.tile(prompt_embeds, (num_images_per_prompt, 1, 1))

        return prompt_embeds

    @staticmethod
    def _align_prompt_seq_len(
        embeds: Tensor,
        target_seq_len: int,
    ) -> Tensor:
        cur_len = int(embeds.shape[1])
        if cur_len == target_seq_len:
            return embeds
        if cur_len > target_seq_len:
            return embeds[:, :target_seq_len, :]

        pad_len = target_seq_len - cur_len
        pad = Tensor.zeros(
            (int(embeds.shape[0]), pad_len, int(embeds.shape[2])),
            dtype=embeds.dtype,
            device=embeds.device,
        )
        return F.concat([embeds, pad], axis=1)

    @staticmethod
    def _prepare_text_ids(
        seq_len: int,
        device: Device,
    ) -> Tensor:
        text_ids = np.zeros((seq_len, 3), dtype=np.int64)
        text_ids[:, 0] = np.arange(1, seq_len + 1, dtype=np.int64)
        return Tensor.from_dlpack(text_ids).to(device)

    @traced
    def _decode_latents(
        self,
        latents: Tensor,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        if output_type == "latent":
            return latents

        batch_size = int(latents.shape[0])
        ch_size = int(latents.shape[2])
        h = 2 * (height // (self.vae_scale_factor * 2))
        w = 2 * (width // (self.vae_scale_factor * 2))
        latents = F.reshape(
            latents, (batch_size, h // 2, w // 2, 2, 2, ch_size // 4)
        )

        latents = self._postprocess_latents(latents)
        return self._to_numpy(self.vae.decode(latents))

    def _postprocess_latents(self, latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        half_h = latents.shape[1]
        half_w = latents.shape[2]
        c_quarter = latents.shape[5]

        latents = F.permute(latents, (0, 5, 1, 3, 2, 4))
        latents = F.reshape(
            latents, (batch_size, c_quarter, half_h * 2, half_w * 2)
        )
        latents = (latents / float(self.vae.config.scaling_factor)) + float(
            self.vae.config.shift_factor or 0.0
        )
        return latents

    @staticmethod
    def _to_numpy(image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    @staticmethod
    def _vector_norm_per_sample(x: Tensor) -> Tensor:
        squared = x * x
        # x shape: [B, S, C] -> norm shape: [B]
        squared = F.sum(squared, axis=2)
        squared = F.sum(squared, axis=1)
        return F.sqrt(squared + 1e-12)

    @classmethod
    def _apply_cfg_renormalization(
        cls,
        pos: Tensor,
        pred: Tensor,
        cfg_normalization: bool,
    ) -> Tensor:
        if not cfg_normalization:
            return pred

        ori_pos_norm = cls._vector_norm_per_sample(pos)
        new_pos_norm = cls._vector_norm_per_sample(pred)
        while ori_pos_norm.rank > 1:
            ori_pos_norm = F.squeeze(ori_pos_norm, axis=-1)
        while new_pos_norm.rank > 1:
            new_pos_norm = F.squeeze(new_pos_norm, axis=-1)
        max_new_norm = ori_pos_norm
        # Avoid divide-by-zero and clip only when required.
        safe_new_norm = F.where(new_pos_norm > 1e-12, new_pos_norm, 1e-12)
        ratio = max_new_norm / safe_new_norm
        ratio = F.where(new_pos_norm > max_new_norm, ratio, 1.0)
        ratio = F.unsqueeze(F.unsqueeze(ratio, 1), 2)
        return pred * ratio

    @staticmethod
    def scheduler_step(
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    @staticmethod
    def prepare_scheduler(sigmas: Tensor) -> tuple[Tensor, Tensor]:
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = sigmas_next - sigmas_curr
        all_timesteps = sigmas_curr.cast(DType.float32)
        return all_timesteps, all_dt

    @traced
    def preprocess_latents(self, latents: Tensor, dtype: DType) -> Tensor:
        # Compiled pack kernel expects fp32 input, then we cast to model dtype.
        with Tracer("host_to_device_latents"):
            latents = latents.to(self.transformer.devices[0]).cast(
                DType.float32
            )

        with Tracer("patchify_and_pack"):
            batch, channels, height, width = map(int, latents.shape)
            latents = F.reshape(
                latents, (batch, channels, height // 2, 2, width // 2, 2)
            )
            latents = self._pack_latents_from_6d(latents)

        return latents.cast(dtype)

    def _image_to_tensor(
        self,
        image: np.ndarray,
        batch_size: int,
        dtype: DType,
    ) -> Tensor:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected input image shape [H, W, 3], got {image.shape}."
            )

        height, width, _ = image.shape
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(
                f"Input image dimensions must be divisible by {vae_scale}, "
                f"got {height}x{width}."
            )

        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        image_chw = np.transpose(image_f32, (2, 0, 1))
        image_bchw = np.expand_dims(image_chw, axis=0)
        if batch_size > 1:
            image_bchw = np.repeat(image_bchw, batch_size, axis=0)
        image_bchw = np.ascontiguousarray(image_bchw)

        return (
            Tensor.from_dlpack(image_bchw).to(self.vae.devices[0]).cast(dtype)
        )

    def _prepare_img2img_latents(
        self,
        noise_latents: Tensor,
        image: np.ndarray,
        sigmas: Tensor,
    ) -> Tensor:
        noise_latents = noise_latents.to(self.transformer.devices[0])
        batch_size = int(noise_latents.shape[0])
        image_tensor = self._image_to_tensor(
            image=image,
            batch_size=batch_size,
            dtype=self.vae.config.dtype,
        )

        encoder_output = self.vae.encode(image_tensor, return_dict=True)
        posterior = (
            encoder_output["latent_dist"]
            if isinstance(encoder_output, dict)
            else encoder_output
        )
        if not hasattr(posterior, "mode"):
            raise ValueError("VAE encoder output does not expose `mode()`.")

        image_latents = posterior.mode()
        image_latents = (
            image_latents - float(self.vae.config.shift_factor or 0.0)
        ) * float(self.vae.config.scaling_factor)
        image_latents = image_latents.to(self.transformer.devices[0]).cast(
            noise_latents.dtype
        )

        sigma = sigmas[0]
        latents = sigma * noise_latents + (1.0 - sigma) * image_latents
        return latents.cast(noise_latents.dtype)

    def execute(  # type: ignore[override]
        self,
        model_inputs: ZImageModelInputs,
        callback_queue: Queue[np.ndarray | Tensor] | None = None,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> ZImagePipelineOutput:
        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds = self._prepare_prompt_embeddings(
                tokens=model_inputs.tokens.array,
                mask=model_inputs.mask,
                num_images_per_prompt=model_inputs.num_images_per_prompt,
            )

            negative_prompt_embeds: Tensor | None = None
            do_cfg = (
                model_inputs.guidance_scale > 1.0
                and model_inputs.negative_tokens is not None
            )
            if do_cfg and model_inputs.negative_tokens is not None:
                negative_prompt_embeds = self._prepare_prompt_embeddings(
                    tokens=model_inputs.negative_tokens.array,
                    mask=model_inputs.negative_mask,
                    num_images_per_prompt=model_inputs.num_images_per_prompt,
                )
                negative_prompt_embeds = self._align_prompt_seq_len(
                    negative_prompt_embeds,
                    int(prompt_embeds.shape[1]),
                )

        dtype = prompt_embeds.dtype

        timesteps: np.ndarray = model_inputs.timesteps
        batch_size = int(prompt_embeds.shape[0])
        num_timesteps = timesteps.shape[0]
        if num_timesteps < 1:
            raise ValueError("No timesteps available for denoising.")
        text_seq_len = int(prompt_embeds.shape[1])
        text_seq_len_padded = text_seq_len + (-text_seq_len % 32)

        with Tracer("prepare_scheduler"):
            device = self.transformer.devices[0]
            image_seq_len = int(model_inputs.latent_image_ids.shape[-2])
            img_ids_key = (
                f"{text_seq_len_padded}_{image_seq_len}_"
                f"{model_inputs.height}_{model_inputs.width}"
            )
            if img_ids_key in self._cached_img_ids:
                img_ids = self._cached_img_ids[img_ids_key]
            else:
                img_ids_np = model_inputs.latent_image_ids.astype(
                    np.int64, copy=True
                )
                if img_ids_np.ndim == 3:
                    img_ids_np = img_ids_np[0]
                img_ids_np[:, 0] = img_ids_np[:, 0] + text_seq_len_padded + 1
                img_ids = Tensor.from_dlpack(img_ids_np).to(device)
                self._cached_img_ids[img_ids_key] = img_ids
            text_ids_key = f"{text_seq_len}"
            if text_ids_key in self._cached_text_ids:
                txt_ids = self._cached_text_ids[text_ids_key]
            else:
                txt_ids = self._prepare_text_ids(text_seq_len, device)
                self._cached_text_ids[text_ids_key] = txt_ids

            latents = Tensor.from_dlpack(model_inputs.latents)
            sigmas_key = f"{model_inputs.num_inference_steps}_{model_inputs.latents.shape[-2]}_{model_inputs.latents.shape[-1]}"
            if sigmas_key in self._cached_sigmas:
                sigmas = self._cached_sigmas[sigmas_key]
            else:
                sigmas = Tensor.from_dlpack(model_inputs.sigmas).to(device)
                self._cached_sigmas[sigmas_key] = sigmas
            if model_inputs.input_image is not None:
                img_arr = np.array(model_inputs.input_image)
                latents = self._prepare_img2img_latents(
                    noise_latents=latents,
                    image=img_arr,
                    sigmas=sigmas,
                )
            latents = self.preprocess_latents(latents, dtype)
            _, all_dts = self.prepare_scheduler(sigmas)
            dts_seq: Any = all_dts
            if hasattr(dts_seq, "driver_tensor"):
                dts_seq = dts_seq.driver_tensor

            # Precompute transformed timesteps and CFG activity outside loop.
            transformed_timesteps = (1.0 - timesteps).astype(
                np.float32, copy=False
            )
            timesteps_digest = hashlib.sha1(
                transformed_timesteps.tobytes()
            ).hexdigest()
            timesteps_key = f"{num_timesteps}_{batch_size}_{timesteps_digest}"
            if timesteps_key in self._cached_timesteps_batched:
                timesteps_batched = self._cached_timesteps_batched[
                    timesteps_key
                ]
            else:
                timesteps_np = np.broadcast_to(
                    transformed_timesteps[:, None], (num_timesteps, batch_size)
                )
                timesteps_batched = Tensor.from_dlpack(
                    np.ascontiguousarray(timesteps_np)
                ).to(device)
                self._cached_timesteps_batched[timesteps_key] = (
                    timesteps_batched
                )

            # Keep Tensor indexing semantics for [step, batch] access.
            timesteps_seq: Any = timesteps_batched

        cfg_active: np.ndarray | None = None
        if do_cfg:
            if model_inputs.cfg_truncation <= 1.0:
                cfg_active = (
                    transformed_timesteps <= model_inputs.cfg_truncation
                )
            else:
                cfg_active = np.ones(num_timesteps, dtype=np.bool_)

        with Tracer("denoising_loop"):
            for i in tqdm(range(num_timesteps), desc="Denoising"):
                with Tracer(f"denoising_step_{i}"):
                    timestep = timesteps_seq[i]
                    apply_cfg = bool(
                        do_cfg and cfg_active is not None and cfg_active[i]
                    )
                    current_guidance_scale = (
                        model_inputs.guidance_scale if apply_cfg else 0.0
                    )

                    with Tracer("transformer"):
                        noise_pred = self.transformer(
                            latents,
                            prompt_embeds,
                            timestep,
                            img_ids,
                            txt_ids,
                        )[0]

                    if apply_cfg:
                        assert negative_prompt_embeds is not None
                        with Tracer("cfg_transformer"):
                            neg_noise_pred = self.transformer(
                                latents,
                                negative_prompt_embeds,
                                timestep,
                                img_ids,
                                txt_ids,
                            )[0]
                        pos_noise_pred = noise_pred
                        noise_delta = F.sub(noise_pred, neg_noise_pred)
                        noise_pred = F.add(
                            pos_noise_pred,
                            F.mul(noise_delta, current_guidance_scale),
                        )
                        noise_pred = self._apply_cfg_renormalization(
                            pos_noise_pred,
                            noise_pred,
                            model_inputs.cfg_normalization,
                        )

                    with Tracer("scheduler_step"):
                        noise_pred = F.mul(noise_pred, -1.0)
                        dt = dts_seq[i : i + 1]
                        latents = self.scheduler_step(latents, noise_pred, dt)

                    if callback_queue is not None:
                        if hasattr(device, "synchronize"):
                            device.synchronize()
                        with Tracer("decode_callback"):
                            image = self._decode_latents(
                                latents,
                                model_inputs.height,
                                model_inputs.width,
                                output_type=output_type,
                            )
                        callback_queue.put_nowait(image)

        with Tracer("decode_outputs"):
            outputs = self._decode_latents(
                latents,
                model_inputs.height,
                model_inputs.width,
                output_type=output_type,
            )

        return ZImagePipelineOutput(images=outputs)
