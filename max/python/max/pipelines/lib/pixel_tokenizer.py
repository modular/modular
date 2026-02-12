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
from enum import Enum
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
    import PIL.Image
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


async def run_with_default_executor(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Runs a callable in the default thread pool executor.

    Args:
        fn: Callable to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The result of ``fn(*args, **kwargs)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)


class PipelineClassName(str, Enum):
    FLUX = "FluxPipeline"
    FLUX2 = "Flux2Pipeline"
    ZIMAGE = "ZImagePipeline"

    @classmethod
    def from_diffusers_config(
        cls, diffusers_config: dict[str, Any]
    ) -> PipelineClassName:
        """Resolve a PipelineClassName from a diffusers config dict."""
        raw = diffusers_config.get("_class_name")
        if raw is None:
            raise KeyError(
                "diffusers_config is missing required key '_class_name'."
            )
        try:
            return cls(raw)
        except ValueError as e:
            allowed = ", ".join([m.value for m in cls])
            raise ValueError(
                f"Unsupported _class_name={raw!r}. Allowed: {allowed}"
            ) from e


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
        self._pipeline_class_name = PipelineClassName.from_diffusers_config(
            self.diffusers_config
        )

        # Extract static config values once during initialization
        components = self.diffusers_config.get("components", {})
        vae_config = components.get("vae", {}).get("config_dict", {})
        transformer_config = components.get("transformer", {}).get(
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

        # Create scheduler
        scheduler_class_name = components.get("scheduler", {}).get(
            "class_name", None
        )
        scheduler_cfg = components.get("scheduler", {}).get("config_dict", {})
        scheduler_cfg["use_empirical_mu"] = (
            self._pipeline_class_name == PipelineClassName.FLUX2
        )
        self._scheduler = SchedulerFactory.create(
            class_name=scheduler_class_name,
            config_dict=scheduler_cfg,
        )

        self._max_pixel_size = None
        if self._pipeline_class_name == PipelineClassName.FLUX2:
            self._max_pixel_size = 1024 * 1024

    def _prepare_latent_image_ids(
        self, height: int, width: int, batch_size: int = 1
    ) -> npt.NDArray[np.float32]:
        if self._pipeline_class_name == PipelineClassName.FLUX2:
            # Create 4D coordinates using numpy (T=0, H, W, L=0)
            t_coords, h_coords, w_coords, l_coords = np.meshgrid(
                np.array([0]),  # T dimension
                np.arange(height),  # H dimension
                np.arange(width),  # W dimension
                np.array([0]),  # L dimension
                indexing="ij",
            )
            latent_image_ids = np.stack(
                [t_coords, h_coords, w_coords, l_coords], axis=-1
            )
            latent_image_ids = latent_image_ids.reshape(-1, 4)

            latent_image_ids = np.tile(
                latent_image_ids[np.newaxis, :, :], (batch_size, 1, 1)
            )
            return latent_image_ids
        else:
            latent_image_ids = np.zeros((height, width, 3))
            latent_image_ids[..., 1] = (
                latent_image_ids[..., 1] + np.arange(height)[:, None]
            )
            latent_image_ids[..., 2] = (
                latent_image_ids[..., 2] + np.arange(width)[None, :]
            )
            return latent_image_ids.reshape(
                -1, latent_image_ids.shape[-1]
            ).astype(np.float32)

    def _randn_tensor(
        self,
        shape: tuple[int, ...],
        seed: int | None,
    ) -> npt.NDArray[np.float32]:
        rng = np.random.RandomState(seed)
        return rng.standard_normal(shape).astype(np.float32)

    def _preprocess_input_image(
        self,
        image: PIL.Image.Image | npt.NDArray[np.uint8],
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> PIL.Image.Image:
        """Preprocess input image for image-to-image generation.

        This method preprocesses images for condition-based image-to-image generation.
        Matching diffusers behavior: resizes large images, ensures dimensions are multiples
        of vae_scale_factor * 2, and optionally resizes to target dimensions.

        Note: This is a simplified version compared to pipeline_flux2.py which uses
        image_processor.preprocess. This tokenizer-level preprocessing is sufficient
        for the Max framework's condition-based approach.

        Args:
            image: PIL Image or numpy array (uint8) to preprocess.
            target_height: Target height for the image. If None, uses image's height.
            target_width: Target width for the image. If None, uses image's width.

        Returns:
            Preprocessed PIL Image with adjusted dimensions.
        """
        import PIL.Image

        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image.astype(np.uint8))

        image_width, image_height = image.size
        multiple_of = self._vae_scale_factor * 2

        if self._max_pixel_size is not None:
            if image_width * image_height > self._max_pixel_size:
                scale = (
                    self._max_pixel_size / (image_width * image_height)
                ) ** 0.5
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)
                image = image.resize(
                    (new_width, new_height), PIL.Image.Resampling.LANCZOS
                )
                image_width, image_height = image.size

        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of

        if target_height is not None:
            image_height = (target_height // multiple_of) * multiple_of
        if target_width is not None:
            image_width = (target_width // multiple_of) * multiple_of

        if image.size != (image_width, image_height):
            image = image.resize(
                (image_width, image_height), PIL.Image.Resampling.LANCZOS
            )

        return image

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
            latent_height // 2, latent_width // 2, batch_size
        )

        return latents, latent_image_ids

    async def _generate_tokens_ids(
        self,
        prompt: str,
        prompt_2: str | None = None,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
        do_true_cfg: bool = False,
        images: list[PIL.Image.Image] | None = None,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.int64] | None,
    ]:
        """Tokenize prompt(s) with encoder model(s).

        Args:
            prompt: Primary prompt to tokenize.
            prompt_2: Secondary prompt (optional).
            negative_prompt: Negative prompt (optional).
            negative_prompt_2: Secondary negative prompt (optional).
            do_true_cfg: Whether to use true classifier-free guidance.
            images: Optional list of images for image-to-image generation (Flux2 only).

        Returns:
            Tuple of (token_ids, attn_mask, token_ids_2, negative_token_ids, negative_token_ids_2).
            token_ids_2 and negative_token_ids_2 are None if no secondary tokenizer is configured.
        """
        token_ids, attn_mask = await self.encode(prompt, images=images)

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
        images: list[PIL.Image.Image] | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Transforms the provided prompt into a token array."""
        delegate = self.delegate_2 if use_secondary else self.delegate
        max_sequence_length = (
            self.secondary_max_length if use_secondary else self.max_length
        )

        tokenizer_output: Any

        # Check if this is Flux2 pipeline (uses Mistral3Tokenizer with chat_template)
        # Flux2 requires apply_chat_template for proper tokenization

        def _encode_fn(prompt_str: str) -> Any:
            assert delegate is not None

            # For Flux2, use apply_chat_template with format_input
            if self._pipeline_class_name == PipelineClassName.FLUX2:
                from max.pipelines.architectures.flux2.system_messages import (
                    SYSTEM_MESSAGE,
                    format_input,
                )

                messages_batch = format_input(
                    prompts=[prompt_str],
                    system_message=SYSTEM_MESSAGE,
                    images=None,
                )

                return delegate.apply_chat_template(
                    messages_batch[0],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_sequence_length,
                    return_length=False,
                    return_overflowing_tokens=False,
                )
            else:
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

        # Extract input_ids and attention_mask
        if isinstance(tokenizer_output, dict):
            # apply_chat_template returns a dict
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output.get("attention_mask", None)
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)

            # Extract real tokens only (using attention mask) for Flux2
            if self._pipeline_class_name == PipelineClassName.FLUX2:
                # Filter to keep only real tokens (where mask == 1)
                real_token_ids = [
                    token_id
                    for token_id, mask in zip(
                        input_ids[0], attention_mask[0], strict=False
                    )
                    if mask == 1
                ]
                input_ids = [real_token_ids]
                attention_mask = [[1] * len(real_token_ids)]
        else:
            # Standard tokenizer output
            input_ids = tokenizer_output.input_ids
            attention_mask = tokenizer_output.attention_mask

        if max_sequence_length and len(input_ids) > max_sequence_length:
            raise ValueError(
                f"Input string is larger than tokenizer's max length ({len(input_ids)} > {max_sequence_length})."
            )

        encoded_prompt = np.array(input_ids)
        attention_mask_array = np.array(attention_mask).astype(np.bool_)

        return encoded_prompt, attention_mask_array

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
        output: Any,
    ) -> Any:
        """Post-process pipeline output.

        Accepts either a raw numpy array or a GenerationOutput.
        For raw numpy arrays, denormalizes from [-1, 1] to [0, 1].
        For GenerationOutput, returns as-is (denormalization is handled
        in the pipeline variant before encoding to OutputImageContent).
        """
        from max.interfaces.generation import GenerationOutput

        if isinstance(output, GenerationOutput):
            return output

        # Raw numpy path
        pixel_data = (output * 0.5 + 0.5).clip(min=0.0, max=1.0)
        return pixel_data

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> PixelContext:
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
        import PIL.Image

        # 1. Tokenize prompts
        # Convert input_image to list format for _generate_tokens_ids
        images_for_tokenization: list[PIL.Image.Image] | None = None
        if input_image is not None:
            input_img: PIL.Image.Image
            if isinstance(input_image, np.ndarray):
                input_img = PIL.Image.fromarray(input_image.astype(np.uint8))
            else:
                input_img = input_image
            images_for_tokenization = [input_img]

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
            images=images_for_tokenization,
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

        default_sample_size = self._default_sample_size
        vae_scale_factor = self._vae_scale_factor

        height = image_options.height or default_sample_size * vae_scale_factor
        width = image_options.width or default_sample_size * vae_scale_factor

        # 2. Preprocess input image if provided
        preprocessed_image = None
        if input_image is not None:
            preprocessed_image = self._preprocess_input_image(
                input_image, height, width
            )
            height = preprocessed_image.height
            width = preprocessed_image.width

        # 3. Resolve image dimensions using cached static values
        latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        num_inference_steps = image_options.steps
        timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
            image_seq_len, num_inference_steps
        )

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

        # 5. Build the context
        context = PixelContext(
            request_id=request.request_id,
            tokens=token_buffer,
            mask=attn_mask,
            tokens_2=token_buffer_2,
            negative_tokens=negative_token_buffer,
            negative_tokens_2=negative_token_buffer_2,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=image_options.guidance_scale,
            num_images_per_prompt=image_options.num_images,
            true_cfg_scale=image_options.true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            model_name=request.body.model,
            input_image=preprocessed_image,
        )

        for validator in self._context_validators:
            validator(context)

        return context
