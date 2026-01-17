# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
from max.driver import Buffer as Tensor
from max.dtype import DType
from max.experimental import Tensor as Tensor_v3
from max.experimental import functional as F
from max.experimental import random
from max.graph import DeviceRef
from max.pipelines.lib.diffusion_schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from max.pipelines.lib.image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import (
    DiffusionPipeline,
)
from tqdm import tqdm
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)

from ..autoencoder_kl import AutoencoderKLModel
from ..clip import ClipModel
from ..t5 import T5Model
from .model import Flux1Model


def retrieve_timesteps(
    scheduler: Any,
    num_inference_steps: int | None = None,
    device: str | DeviceRef | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, int]:
    r"""Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.

    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `DeviceRef`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.
        **kwargs (`Any`, *optional*):
            Additional arguments to pass to the scheduler's `set_timesteps` method.

    Returns:
        `tuple[Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = int(timesteps.shape[0])
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = int(timesteps.shape[0])
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@dataclass
class FluxPipelineOutput:
    """Output class for Flux image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray` or `Tensor`)
            List of denoised PIL images of length `batch_size` or numpy array or Max tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Max tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image] | np.ndarray | Tensor


class FluxPipeline(DiffusionPipeline):
    config_name = "model_index.json"

    components = {
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "vae": AutoencoderKLModel,
        "text_encoder": ClipModel,
        "tokenizer": CLIPTokenizer,
        "text_encoder_2": T5Model,
        "tokenizer_2": T5TokenizerFast,
        "transformer": Flux1Model,
    }

    def init_remaining_components(self) -> None:
        image_processor_class = self.components.get(
            "image_processor", VaeImageProcessor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        image_processor = image_processor_class(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.image_processor = image_processor

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str],
        device: DeviceRef | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`DeviceRef`):
                Max device
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            prompt_embeds (`Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            max_sequence_length (`int`, defaults to 512): Maximum sequence length to use with the `prompt`.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if lora_scale is not None and isinstance(self, FluxPipeline):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and hasattr(
                self.text_encoder, "set_lora_scale"
            ):
                self.text_encoder.set_lora_scale(lora_scale)
            if self.text_encoder_2 is not None and hasattr(
                self.text_encoder_2, "set_lora_scale"
            ):
                self.text_encoder_2.set_lora_scale(lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=min(
                    max_sequence_length, self.tokenizer.model_max_length
                ),
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
            )
            text_input_ids = Tensor_v3.constant(
                text_inputs.input_ids, device=device, dtype=DType.int64
            )

            text_encoder_outputs = self.text_encoder(text_input_ids)
            prompt_embeds = text_encoder_outputs[0]
            pooled_prompt_embeds = text_encoder_outputs[1]

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        if self.text_encoder_2 is not None:
            text_inputs_2 = self.tokenizer_2(
                prompt_2,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
            )
            text_input_ids_2 = Tensor_v3.constant(
                text_inputs_2.input_ids, device=device, dtype=DType.int64
            )

            prompt_embeds_2 = self.text_encoder_2(text_input_ids_2)[0]
        else:
            prompt_embeds_2 = None

        if prompt_embeds_2 is not None:
            prompt_embeds = prompt_embeds_2

        text_ids = Tensor_v3.zeros(
            (prompt_embeds.shape[1], 3),
            device=device,
            dtype=prompt_embeds.dtype,
        )

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = Tensor_v3.from_dlpack(
            prompt_embeds
        )  # V2 Tensor to V3 Tensor
        pooled_prompt_embeds = Tensor_v3.from_dlpack(
            pooled_prompt_embeds
        )  # V2 Tensor to V3 Tensor

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

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: DeviceRef,
        dtype: DType,
    ) -> Tensor_v3:
        latent_image_ids = np.stack(
            [
                np.zeros((height, width)),
                np.broadcast_to(np.arange(height)[:, None], (height, width)),
                np.broadcast_to(np.arange(width)[None, :], (height, width)),
            ],
            axis=-1,
        )

        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape

        latent_image_ids = np.reshape(
            latent_image_ids,
            (
                latent_image_id_height * latent_image_id_width,
                latent_image_id_channels,
            ),
        )
        latent_image_ids = (
            Tensor_v3.from_dlpack(latent_image_ids).to(device).cast(dtype)
        )

        return latent_image_ids

    @staticmethod
    def _pack_latents(
        latents: Tensor_v3,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> Tensor_v3:
        latents = F.reshape(
            latents,
            (batch_size, num_channels_latents, height // 2, 2, width // 2, 2),
        )
        latents = F.permute(latents, (0, 2, 4, 1, 3, 5))
        latents = F.reshape(
            latents,
            (
                batch_size,
                (height // 2) * (width // 2),
                num_channels_latents * 4,
            ),
        )

        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor_v3,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> Tensor_v3:
        # TODO: should compile this function for speed up.
        batch_size, _, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))

        latents = F.reshape(
            latents,
            (batch_size.dim, height // 2, width // 2, channels.dim // 4, 2, 2),
        )
        latents = F.permute(latents, (0, 3, 1, 4, 2, 5))

        latents = F.reshape(
            latents, (batch_size.dim, channels.dim // (2 * 2), height, width)
        )

        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: DeviceRef,
        latents: Tensor_v3 | None = None,
    ) -> tuple[Tensor_v3, Tensor_v3]:
        """Prepare latents for the Flux pipeline.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of latent channels.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type for the latents.
            device: The device to run on.
            latents: Pre-generated latents.

        Returns:
            Tuple of latents and latent image ids.
        """
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device).cast(dtype), latent_image_ids

        latents = random.normal(shape, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        return latents, latent_image_ids

    def __call__(
        self,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        latents: Tensor | None = None,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[Tensor] | None = None,
        negative_ip_adapter_image: PipelineImageInput | None = None,
        negative_ip_adapter_image_embeds: list[Tensor] | None = None,
        negative_prompt_embeds: Tensor | None = None,
        negative_pooled_prompt_embeds: Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        max_sequence_length: int = 512,
    ):
        r"""Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            latents (`Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device()

        lora_scale = (
            self._joint_attention_kwargs.get("scale", None)
            if self._joint_attention_kwargs is not None
            else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler, "use_flow_sigmas")
            and self.scheduler.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1].dim
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self._num_timesteps = timesteps.shape[0]

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = Tensor_v3.full(
                [latents.shape[0].dim],
                guidance_scale,
                device=device,
                dtype=prompt_embeds.dtype,
            )
        else:
            guidance = Tensor_v3.zeros(
                [latents.shape[0].dim],
                device=device,
                dtype=prompt_embeds.dtype,
            )

        if (
            ip_adapter_image is not None
            or ip_adapter_image_embeds is not None
            or negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            raise NotImplementedError(
                "IP adapter is not supported for Max yet."
            )

        if self._joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        batch_size = latents.shape[0].dim
        for i in tqdm(range(self._num_timesteps), desc="Denoising"):
            if self._interrupt:
                continue

            t = timesteps[i]
            self._current_timestep = t
            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                    image_embeds
                )

            # NOTE: Convert timesteps to a Max Tensor before denoising loop,
            # as in the original implementation, results in a significant slow down.
            # As a workaround, we keep timesteps as a numpy array and convert it
            # to a Max Tensor here. This might require a more efficient way to handle this.
            # Converting to a Max module V3 Tensor also results in a significant slow down.
            timestep = np.full((batch_size,), t) / 1000.0
            timestep = Tensor.from_dlpack(timestep).to(prompt_embeds.device)

            noise_pred = self.transformer(
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                timestep,
                latent_image_ids,
                text_ids,
                guidance,
            )[0]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                        negative_image_embeds
                    )

                neg_noise_pred = self.transformer(
                    latents,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    timestep,
                    latent_image_ids,
                    negative_text_ids,
                    guidance,
                )[0]
                # TODO: negative prompt path is very slow, need to optimize.
                noise_pred = Tensor_v3.from_dlpack(noise_pred)
                neg_noise_pred = Tensor_v3.from_dlpack(neg_noise_pred)
                noise_pred = neg_noise_pred + true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(
                    self, i, t, callback_kwargs
                )

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop(
                    "prompt_embeds", prompt_embeds
                )

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = Tensor_v3.from_dlpack(latents)  # V2 Tensor to V3 Tensor
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents)[0]

            image = Tensor_v3.from_dlpack(image)  # V2 Tensor to V3 Tensor
            image = self.image_processor.postprocess(
                image, output_type=output_type
            )

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
