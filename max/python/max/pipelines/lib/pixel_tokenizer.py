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
# mypy: disable-error-code="import-not-found"
"""Pixel generation tokenizer implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    PipelineTokenizer,
    TokenBuffer,
)
from max.interfaces.request import OpenResponsesRequest
from max.pipelines.core import PixelContext
from transformers import AutoTokenizer

from .diffusion_schedulers import SchedulerFactory

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


async def run_with_default_executor(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)


class PixelGenerationTokenizer(
    PipelineTokenizer[
        PixelContext,
        tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]],
        OpenResponsesRequest,
    ]
):
    """Encapsulates creation of PixelContext and specific token encode/decode logic.

    Args:
        model_path: Path to the model/tokenizer.
        pipeline_config: Pipeline configuration (must include diffusers_config).
        subfolder: Subfolder within the model path for the primary tokenizer.
        subfolder_2: Optional subfolder for a second tokenizer (e.g. text encoder).
        revision: Git revision/branch to use.
        max_length: Maximum sequence length for the primary tokenizer.
        secondary_max_length: Maximum sequence length for the secondary tokenizer, if used.
        trust_remote_code: Whether to trust remote code from the model.
        context_validators: Optional list of validators to run on PixelContext.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        subfolder: str,
        *,
        subfolder_2: str | None = None,
        revision: str | None = None,
        max_length: int | None = None,
        secondary_max_length: int | None = None,
        trust_remote_code: bool = False,
        context_validators: list[Callable[[PixelContext], None]] | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        if max_length is None:
            raise ValueError(
                "diffusion models frequently have an unbounded max length. Please provide a max length"
            )

        self.max_length = max_length

        if secondary_max_length is None and subfolder_2 is not None:
            raise ValueError(
                "diffusion models frequently have an unbounded max length. Please provide a max length"
            )

        self.secondary_max_length = secondary_max_length

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                model_max_length=self.max_length,
                subfolder=subfolder,
            )

            if subfolder_2 is not None:
                self.delegate_2 = AutoTokenizer.from_pretrained(
                    model_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    model_max_length=self.secondary_max_length,
                    subfolder=subfolder_2,
                )
            else:
                self.delegate_2 = None
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust-remote-code' is needed but not set\n"
            ) from e

        self._context_validators = (
            context_validators if context_validators else []
        )

        # Extract diffusers_config
        if not pipeline_config or not hasattr(
            pipeline_config.model, "diffusers_config"
        ):
            raise ValueError(
                "pipeline_config.model.diffusers_config is required for PixelGenerationTokenizer. "
                "Please provide a pipeline_config with a valid diffusers_config."
            )
        if pipeline_config.model.diffusers_config is None:
            raise ValueError(
                "pipeline_config.model.diffusers_config cannot be None. "
                "Please provide a valid diffusers_config."
            )
        self.diffusers_config = pipeline_config.model.diffusers_config

        # Store the pipeline class name for model-specific behavior
        self._pipeline_class_name = self.diffusers_config.get(
            "_class_name", None
        )

        # Extract static config values once during initialization
        components = self.diffusers_config.get("components", {})
        vae_config = components.get("vae", {}).get("config_dict", {})
        transformer_config = components.get("transformer", {}).get(
            "config_dict", {}
        )
        scheduler_config = components.get("scheduler", {}).get(
            "config_dict", {}
        )

        # Compute static VAE scale factor
        block_out_channels = vae_config.get("block_out_channels", None)
        self._vae_scale_factor = (
            2 ** (len(block_out_channels) - 1) if block_out_channels else 8
        )

        # Store static model dimensions
        self._default_sample_size = 128
        self._num_channels_latents = transformer_config["in_channels"] // 4

        # Store static scheduler config for shift calculation
        self._base_image_seq_len = scheduler_config.get(
            "base_image_seq_len", 256
        )
        self._max_image_seq_len = scheduler_config.get(
            "max_image_seq_len", 4096
        )
        self._base_shift = scheduler_config.get("base_shift", 0.5)
        self._max_shift = scheduler_config.get("max_shift", 1.15)

        # Store guidance embeds flag
        self._use_guidance_embeds = transformer_config.get(
            "guidance_embeds", False
        )

        # Create scheduler
        scheduler_component = components.get("scheduler", {})
        self._scheduler = SchedulerFactory.create(
            scheduler_component.get("class_name"),
            scheduler_component.get("config_dict", {}),
        )
        self._scheduler_use_flow_sigmas = getattr(
            self._scheduler.config, "use_flow_sigmas", False
        )

    def _calculate_shift(
        self,
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

    @staticmethod
    def _prepare_latent_image_ids(
        height: int, width: int
    ) -> npt.NDArray[np.float32]:
        latent_image_ids = np.zeros((height, width, 3))
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + np.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + np.arange(width)[None, :]
        )
        return latent_image_ids.reshape(-1, latent_image_ids.shape[-1]).astype(
            np.float32
        )

    def _randn_tensor(
        self,
        shape: tuple[int, ...],
        seed: int | None,
    ) -> npt.NDArray[np.float32]:
        rng = np.random.RandomState(seed)
        return rng.standard_normal(shape).astype(np.float32)

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        seed: int | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        shape = (batch_size, num_channels_latents, latent_height, latent_width)

        latents = self._randn_tensor(shape, seed)
        latent_image_ids = self._prepare_latent_image_ids(
            latent_height // 2, latent_width // 2
        )

        return latents, latent_image_ids

    def _retrieve_timesteps(
        self,
        scheduler: Any,
        num_inference_steps: int,
        sigmas: npt.NDArray[np.float32] | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray[np.float32], int]:
        r"""Calls the scheduler's set_timesteps and returns timesteps.

        Handles custom timesteps. Any ``**kwargs`` are passed to
        ``scheduler.set_timesteps``.

        Args:
            scheduler: The scheduler to get timesteps from.
            num_inference_steps: The number of diffusion steps when generating
                samples with a pre-trained model.
            sigmas: Optional custom sigmas to override the timestep spacing
                strategy. If provided, ``num_inference_steps`` must be None.
            **kwargs: Passed through to ``scheduler.set_timesteps``.

        Returns:
            A tuple of (timestep schedule array, number of inference steps).
        """
        if sigmas is not None:
            try:
                scheduler.set_timesteps(sigmas=sigmas, **kwargs)
            except TypeError as e:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                ) from e
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps.astype(np.float32), num_inference_steps

    async def _generate_tokens_ids(
        self,
        prompt: str,
        prompt_2: str | None = None,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
        do_true_cfg: bool = False,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.int64] | None,
    ]:
        """Tokenize prompt(s) with encoder model(s).

        Returns:
            Tuple of (token_ids, attn_mask, token_ids_2, negative_token_ids, negative_token_ids_2).
            token_ids_2 and negative_token_ids_2 are None if no secondary tokenizer is configured.
        """
        token_ids, attn_mask = await self.encode(prompt)

        token_ids_2: npt.NDArray[np.int64] | None = None
        if self.delegate_2 is not None:
            token_ids_2, _attn_mask_2 = await self.encode(
                prompt_2 or prompt,
                use_secondary=True,
            )

        negative_token_ids: npt.NDArray[np.int64] | None = None
        negative_token_ids_2: npt.NDArray[np.int64] | None = None
        if do_true_cfg:
            negative_token_ids, _attn_mask_neg = await self.encode(
                negative_prompt or ""
            )
            if self.delegate_2 is not None:
                negative_token_ids_2, _attn_mask_neg_2 = await self.encode(
                    negative_prompt_2 or negative_prompt or "",
                    use_secondary=True,
                )

        return (
            token_ids,
            attn_mask,
            token_ids_2,
            negative_token_ids,
            negative_token_ids_2,
        )

    @property
    def eos(self) -> int:
        """Returns the end-of-sequence token ID."""
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self,
        prompt: str,
        add_special_tokens: bool = True,
        *,
        use_secondary: bool = False,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Transforms the provided prompt into a token array."""
        delegate = self.delegate_2 if use_secondary else self.delegate
        max_sequence_length = (
            self.secondary_max_length if use_secondary else self.max_length
        )

        tokenizer_output: Any

        def _encode_fn(prompt_str: str) -> Any:
            assert delegate is not None
            return delegate(
                prompt_str,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=add_special_tokens,
            )

        # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
        # Add a standard (non-async) lock in the executor thread if needed.
        tokenizer_output = await run_with_default_executor(_encode_fn, prompt)

        if (
            max_sequence_length
            and len(tokenizer_output.input_ids) > max_sequence_length
        ):
            raise ValueError(
                f"Input string is larger than tokenizer's max length ({len(tokenizer_output.input_ids)} > {max_sequence_length})."
            )

        encoded_prompt = np.array(tokenizer_output.input_ids)
        attention_mask = np.array(tokenizer_output.attention_mask).astype(
            np.bool_
        )

        return encoded_prompt, attention_mask

    async def decode(
        self,
        encoded: tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]],
        **kwargs,
    ) -> str:
        """Decodes token arrays to text (not implemented for this tokenizer)."""
        raise NotImplementedError(
            "Decoding is not implemented for this tokenizer."
        )

    async def postprocess(
        self,
        pixel_data: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Post-process pixel data from model output (NCHW -> NHWC, normalized)."""
        pixel_data = (pixel_data * 0.5 + 0.5).clip(min=0.0, max=1.0)
        pixel_data = pixel_data.transpose(0, 2, 3, 1)
        return pixel_data

    async def new_context(self, request: OpenResponsesRequest) -> PixelContext:
        """Create a new PixelContext object, leveraging necessary information from OpenResponsesRequest."""
        # Extract prompt from request.body.input (must be a string)
        if isinstance(request.body.input, list):
            raise ValueError(
                "Pixel generation does not support message list input. "
                "Please provide a single string prompt via the 'input' field."
            )

        prompt = request.body.input
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        # Extract image provider options (required for pixel generation)
        if not request.body.provider_options:
            raise ValueError(
                "Pixel generation requires 'provider_options' to be set. "
                "Please provide a ProviderOptions object with ImageProviderOptions."
            )

        if not request.body.provider_options.image:
            raise ValueError(
                "Pixel generation requires 'provider_options.image' to be set. "
                "Please provide ImageProviderOptions for image generation parameters."
            )

        image_options = request.body.provider_options.image

        if (
            image_options.guidance_scale < 1.0
            or image_options.true_cfg_scale < 1.0
        ):
            logger.warning(
                f"Guidance scales < 1.0 detected (guidance_scale={image_options.guidance_scale}, "
                f"true_cfg_scale={image_options.true_cfg_scale}). This is mathematically possible"
                " but may produce lower quality or unexpected results."
            )

        if (
            image_options.true_cfg_scale > 1.0
            and image_options.negative_prompt is None
        ):
            logger.warning(
                f"true_cfg_scale={image_options.true_cfg_scale} is set, but no negative_prompt "
                "is provided. True classifier-free guidance requires a negative prompt; "
                "falling back to standard generation."
            )

        do_true_cfg = (
            image_options.true_cfg_scale > 1.0
            and image_options.negative_prompt is not None
        )

        # 1. Tokenize prompts
        (
            token_ids,
            attn_mask,
            token_ids_2,
            negative_token_ids,
            negative_token_ids_2,
        ) = await self._generate_tokens_ids(
            prompt,
            image_options.secondary_prompt,
            image_options.negative_prompt,
            image_options.secondary_negative_prompt,
            do_true_cfg,
        )

        token_buffer = TokenBuffer(
            array=token_ids.astype(np.int64, copy=False),
        )
        token_buffer_2 = None
        if token_ids_2 is not None:
            token_buffer_2 = TokenBuffer(
                array=token_ids_2.astype(np.int64, copy=False),
            )
        negative_token_buffer = None
        if negative_token_ids is not None:
            negative_token_buffer = TokenBuffer(
                array=negative_token_ids.astype(np.int64, copy=False),
            )
        negative_token_buffer_2 = None
        if negative_token_ids_2 is not None:
            negative_token_buffer_2 = TokenBuffer(
                array=negative_token_ids_2.astype(np.int64, copy=False),
            )

        # 3. Resolve image dimensions using cached static values
        height = (
            image_options.height
            or self._default_sample_size * self._vae_scale_factor
        )
        width = (
            image_options.width
            or self._default_sample_size * self._vae_scale_factor
        )

        latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        mu = self._calculate_shift(
            image_seq_len,
            self._base_image_seq_len,
            self._max_image_seq_len,
            self._base_shift,
            self._max_shift,
        )

        num_inference_steps = image_options.steps
        sigmas: npt.NDArray[np.float32] | None = (
            None
            if self._scheduler_use_flow_sigmas
            else np.linspace(
                1.0,
                1.0 / num_inference_steps,
                num_inference_steps,
                dtype=np.float32,
            )
        )
        timesteps, num_inference_steps = self._retrieve_timesteps(
            self._scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        # Z-Image uses inverted timestep normalization compared to other flow matching models
        if self._pipeline_class_name == "ZImagePipeline":
            timesteps = ((1000.0 - timesteps) / 1000.0).astype(np.float32)
        else:
            timesteps = (timesteps / 1000.0).astype(np.float32)
        num_warmup_steps: int = max(
            len(timesteps) - num_inference_steps * self._scheduler.order, 0
        )

        latents, latent_image_ids = self._prepare_latents(
            image_options.num_images,
            self._num_channels_latents,
            latent_height,
            latent_width,
            request.body.seed,
        )

        guidance: npt.NDArray[np.float32] | None = None
        if self._use_guidance_embeds:
            guidance = np.array(
                [image_options.guidance_scale], dtype=np.float32
            )

        # 5. Build the context
        context = PixelContext(
            request_id=request.request_id,
            tokens=token_buffer,
            mask=attn_mask,
            tokens_2=token_buffer_2,
            negative_tokens=negative_token_buffer,
            negative_tokens_2=negative_token_buffer_2,
            timesteps=timesteps,
            sigmas=self._scheduler.sigmas,
            latents=latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=image_options.guidance_scale,
            guidance=guidance,
            num_images_per_prompt=image_options.num_images,
            true_cfg_scale=image_options.true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            model_name=request.body.model,
        )

        for validator in self._context_validators:
            validator(context)

        return context
