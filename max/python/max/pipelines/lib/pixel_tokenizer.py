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
import base64
import logging
import math
import threading
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.interfaces import (
    PipelineTokenizer,
    TokenBuffer,
)
from max.interfaces.generation import GenerationOutput
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    InputImageContent,
    InputTextContent,
)
from max.pipelines.core import PixelContext
from transformers import AutoTokenizer

from .diffusion_schedulers import SchedulerFactory
from .qwen_image_processor import Qwen2_5VLPromptImageProcessor

if TYPE_CHECKING:
    import PIL.Image
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

QWEN_EDIT_PROMPT_IMAGE_SIZE = 384 * 384
QWEN_EDIT_VAE_IMAGE_SIZE = 1024 * 1024
QWEN_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN_EDIT_IMAGE_PROMPT_TEMPLATE = (
    "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
)


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


class LockedTokenizer:
    """Serialize access to tokenizer interfaces that may race across threads."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped tokenizer under the shared lock."""
        with self._lock:
            return self._delegate(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the wrapped tokenizer chat template under the shared lock."""
        with self._lock:
            return self._delegate.apply_chat_template(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Encode with the wrapped tokenizer under the shared lock."""
        with self._lock:
            return self._delegate.encode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class PipelineClassName(str, Enum):
    """Known pipeline class names for image generation models."""

    FLUX = "FluxPipeline"
    FLUX2 = "Flux2Pipeline"
    FLUX2_KLEIN = "Flux2KleinPipeline"
    ZIMAGE = "ZImagePipeline"
    QWENIMAGE = "QwenImagePipeline"
    QWENIMAGE_EDIT = "QwenImageEditPipeline"
    QWENIMAGE_EDIT_PLUS = "QwenImageEditPlusPipeline"

    @property
    def is_qwen_image_family(self) -> bool:
        """Returns whether the pipeline belongs to the Qwen image family."""
        return self in {
            PipelineClassName.QWENIMAGE,
            PipelineClassName.QWENIMAGE_EDIT,
            PipelineClassName.QWENIMAGE_EDIT_PLUS,
        }

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
        if raw in {
            cls.QWENIMAGE_EDIT.value,
            cls.QWENIMAGE_EDIT_PLUS.value,
        }:
            return cls.QWENIMAGE
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
        default_num_inference_steps: int = 50,
        context_validators: list[Callable[[PixelContext], None]] | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self._default_num_inference_steps = default_num_inference_steps

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
        self.delegate: LockedTokenizer
        self.delegate_2: LockedTokenizer | None

        try:
            self.delegate = LockedTokenizer(
                AutoTokenizer.from_pretrained(
                    model_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    model_max_length=self.max_length,
                    subfolder=subfolder,
                )
            )

            if subfolder_2 is not None:
                self.delegate_2 = LockedTokenizer(
                    AutoTokenizer.from_pretrained(
                        model_path,
                        revision=revision,
                        trust_remote_code=trust_remote_code,
                        model_max_length=self.secondary_max_length,
                        subfolder=subfolder_2,
                    )
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
        self._raw_pipeline_class_name = self.diffusers_config.get("_class_name")
        self._is_qwen_image_edit_family = self._raw_pipeline_class_name in {
            PipelineClassName.QWENIMAGE_EDIT.value,
            PipelineClassName.QWENIMAGE_EDIT_PLUS.value,
        }
        self._qwen_edit_image_token_id: int | None = None
        self._qwen_edit_image_processor: (
            Qwen2_5VLPromptImageProcessor | None
        ) = None
        if self._is_qwen_image_edit_family:
            self._qwen_edit_image_token_id = (
                self.delegate.convert_tokens_to_ids("<|image_pad|>")
            )
            text_encoder_config = (
                self.diffusers_config.get("components", {})
                .get("text_encoder", {})
                .get("config_dict", {})
            )
            vision_config = text_encoder_config.get("vision_config", {})
            self._qwen_edit_image_processor = Qwen2_5VLPromptImageProcessor(
                patch_size=vision_config.get("patch_size", 14),
                temporal_patch_size=vision_config.get("temporal_patch_size", 2),
                merge_size=vision_config.get("spatial_merge_size", 2),
            )

        # Preserve tokenizer attention masks so downstream text encoders can
        # derive additive attention bias directly from tokenizer semantics.

        # Extract static config values once during initialization
        components = self.diffusers_config.get("components", {})
        vae_config = components.get("vae", {}).get("config_dict", {})
        transformer_config = components.get("transformer", {}).get(
            "config_dict", {}
        )

        # Compute static VAE scale factor
        block_out_channels = vae_config.get("block_out_channels", None)
        if block_out_channels:
            self._vae_scale_factor = 2 ** (len(block_out_channels) - 1)
        elif self._pipeline_class_name.is_qwen_image_family:
            self._vae_scale_factor = 8
        else:
            self._vae_scale_factor = 8

        # Store static model dimensions
        self._default_sample_size = 128
        if self._pipeline_class_name == PipelineClassName.ZIMAGE:
            self._num_channels_latents = transformer_config["in_channels"]
        else:
            self._num_channels_latents = transformer_config["in_channels"] // 4

        # Create scheduler
        scheduler_class_name = components.get("scheduler", {}).get(
            "class_name", None
        )
        scheduler_cfg = components.get("scheduler", {}).get("config_dict", {})
        scheduler_cfg["use_empirical_mu"] = self._pipeline_class_name in (
            PipelineClassName.FLUX2,
            PipelineClassName.FLUX2_KLEIN,
        )
        self._scheduler = SchedulerFactory.create(
            class_name=scheduler_class_name,
            config_dict=scheduler_cfg,
        )
        self._scheduler_shift = float(scheduler_cfg.get("shift", 1.0))

        self._max_pixel_size = None
        if self._pipeline_class_name in (
            PipelineClassName.FLUX2,
            PipelineClassName.FLUX2_KLEIN,
        ):
            self._max_pixel_size = 1024 * 1024

    def _prepare_latent_image_ids(
        self, height: int, width: int, batch_size: int = 1
    ) -> npt.NDArray[np.float32]:
        if self._pipeline_class_name in (
            PipelineClassName.FLUX2,
            PipelineClassName.FLUX2_KLEIN,
        ):
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
        elif self._pipeline_class_name.is_qwen_image_family:
            t_coords = np.zeros((height, width), dtype=np.int64)
            h_centered = np.arange(height, dtype=np.int64) - (
                height - height // 2
            )
            w_centered = np.arange(width, dtype=np.int64) - (width - width // 2)
            h_coords, w_coords = np.meshgrid(
                h_centered, w_centered, indexing="ij"
            )
            latent_image_ids = np.stack([t_coords, h_coords, w_coords], axis=-1)
            latent_image_ids = latent_image_ids.reshape(-1, 3)
            latent_image_ids = np.tile(
                latent_image_ids[np.newaxis, :, :], (batch_size, 1, 1)
            )
            return latent_image_ids.astype(np.float32, copy=False)
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

    @staticmethod
    def _resize_with_center_crop(
        image: PIL.Image.Image, target_width: int, target_height: int
    ) -> PIL.Image.Image:
        ratio = target_width / target_height
        src_ratio = image.width / image.height

        src_w = (
            target_width
            if ratio > src_ratio
            else image.width * target_height // image.height
        )
        src_h = (
            target_height
            if ratio <= src_ratio
            else image.height * target_width // image.width
        )

        resized = image.resize(
            (src_w, src_h), resample=PIL.Image.Resampling.LANCZOS
        )
        canvas = PIL.Image.new("RGB", (target_width, target_height))
        canvas.paste(
            resized,
            box=(
                target_width // 2 - src_w // 2,
                target_height // 2 - src_h // 2,
            ),
        )
        return canvas

    def _preprocess_input_image(
        self,
        image: PIL.Image.Image | npt.NDArray[np.uint8],
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        preserve_aspect_ratio: bool = True,
    ) -> PIL.Image.Image:
        """Preprocess input image for image-to-image generation.

        Matches the shared image-to-image behavior:
        - cap image area when needed
        - floor dimensions to multiples of vae_scale_factor * 2
        - apply aspect-ratio preserving center-crop resize to the floored size

        Args:
            image: PIL Image or numpy array (uint8) to preprocess.
            target_height: Optional requested output height before latent prep.
            target_width: Optional requested output width before latent prep.
            preserve_aspect_ratio: Whether to keep the source aspect ratio when
                resizing. When false, resize directly to the target dimensions.

        Returns:
            Preprocessed PIL Image with adjusted dimensions.
        """
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image.astype(np.uint8))

        if image.mode != "RGB":
            image = image.convert("RGB")

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

        image_width = max(
            (image_width // multiple_of) * multiple_of, multiple_of
        )
        image_height = max(
            (image_height // multiple_of) * multiple_of, multiple_of
        )

        if target_width is not None:
            image_width = max(
                (int(target_width) // multiple_of) * multiple_of, multiple_of
            )
        if target_height is not None:
            image_height = max(
                (int(target_height) // multiple_of) * multiple_of, multiple_of
            )

        if image.size != (image_width, image_height):
            if preserve_aspect_ratio:
                image = self._resize_with_center_crop(
                    image, image_width, image_height
                )
            else:
                image = image.resize(
                    (image_width, image_height),
                    resample=PIL.Image.Resampling.LANCZOS,
                )

        if target_width is None:
            image_width = max(
                (image_width // multiple_of) * multiple_of, multiple_of
            )
        else:
            image_width = max(
                (target_width // multiple_of) * multiple_of, multiple_of
            )

        if image.size != (image_width, image_height):
            if target_height is None and target_width is None:
                image = self._resize_with_center_crop(
                    image, image_width, image_height
                )
            else:
                image = image.resize(
                    (image_width, image_height),
                    PIL.Image.Resampling.LANCZOS,
                )

        return image

    @staticmethod
    def _resize_image_to_area(
        image: npt.NDArray[np.uint8], target_area: int
    ) -> npt.NDArray[np.uint8]:
        width = math.sqrt(target_area * (image.shape[1] / image.shape[0]))
        height = width / (image.shape[1] / image.shape[0])
        resized_width = round(width / 32) * 32
        resized_height = round(height / 32) * 32
        if (image.shape[1], image.shape[0]) == (resized_width, resized_height):
            return image

        pil_image = PIL.Image.fromarray(image)
        resized = pil_image.resize(
            (resized_width, resized_height), PIL.Image.Resampling.LANCZOS
        )
        return np.asarray(resized)

    def _prepare_qwen_edit_condition_images(
        self,
        input_images: list[npt.NDArray[np.uint8]] | None,
    ) -> tuple[
        list[npt.NDArray[np.uint8]] | None, list[npt.NDArray[np.uint8]] | None
    ]:
        if not input_images:
            return None, None

        prompt_images = [
            self._resize_image_to_area(image, QWEN_EDIT_PROMPT_IMAGE_SIZE)
            for image in input_images
        ]
        vae_condition_images = [
            self._resize_image_to_area(image, QWEN_EDIT_VAE_IMAGE_SIZE)
            for image in input_images
        ]
        return prompt_images, vae_condition_images

    def _prepare_qwen_edit_tokens(
        self,
        prompt: str,
        prompt_images: list[npt.NDArray[np.uint8]] | None,
    ) -> npt.NDArray[np.int64]:
        if not prompt_images:
            raise ValueError(
                "prompt_images are required for qwen edit tokenization"
            )
        if self._qwen_edit_image_processor is None:
            raise ValueError("qwen edit image processor is not initialized")
        if self._qwen_edit_image_token_id is None:
            raise ValueError("qwen edit image token id is not initialized")

        from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
            fetch_image,
        )

        processed_images = [
            fetch_image({"image": PIL.Image.fromarray(image).convert("RGB")})
            for image in prompt_images
        ]
        processed = self._qwen_edit_image_processor(
            images=processed_images,
            return_tensors="np",
        )
        processed_dict = (
            processed[0] if isinstance(processed, tuple) else processed
        )
        image_grid_thw = np.asarray(
            processed_dict["image_grid_thw"], dtype=np.int64
        )

        vision_prefix = "".join(
            QWEN_EDIT_IMAGE_PROMPT_TEMPLATE.format(index + 1)
            for index in range(len(prompt_images))
        )
        formatted_prompt = QWEN_EDIT_PROMPT_TEMPLATE.format(
            vision_prefix + prompt
        )
        encoded = self.delegate(
            formatted_prompt,
            padding=False,
            return_tensors="np",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"][0].astype(np.int64, copy=False)

        merge_len = self._qwen_edit_image_processor.merge_size**2
        image_token_indices = np.where(
            input_ids == self._qwen_edit_image_token_id
        )[0]
        if len(image_token_indices) < len(image_grid_thw):
            raise ValueError(
                "not enough qwen edit image placeholder tokens were generated"
            )

        for index, grid_thw in enumerate(reversed(image_grid_thw)):
            token_index = image_token_indices[-(index + 1)]
            t, h, w = grid_thw
            num_image_tokens = int((t * h * w) // merge_len)
            input_ids = np.insert(
                input_ids,
                token_index,
                [self._qwen_edit_image_token_id] * (num_image_tokens - 1),
            )

        return input_ids.astype(np.int64, copy=False)

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
            if self._pipeline_class_name == PipelineClassName.FLUX2:
                # Import lazily to avoid a lib <-> architectures cycle.
                from max.pipelines.architectures.flux2_modulev3.system_messages import (
                    SYSTEM_MESSAGE,
                    format_input,
                )

                messages_batch = format_input(
                    prompts=[prompt_str],
                    system_message=SYSTEM_MESSAGE,
                    images=None,
                )

                # Validate prompt length before truncation.
                # apply_chat_template with truncation=True silently
                # drops tokens; error early instead.
                precheck = delegate.apply_chat_template(
                    messages_batch[0],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    truncation=False,
                )
                precheck_ids = precheck["input_ids"]
                precheck_len = (
                    len(precheck_ids[0])
                    if precheck_ids and isinstance(precheck_ids[0], list)
                    else len(precheck_ids)
                )
                if max_sequence_length and precheck_len > max_sequence_length:
                    raise ValueError(
                        f"Prompt is too long for this model's text"
                        f" encoder: {precheck_len} tokens exceeds"
                        f" the maximum of {max_sequence_length}"
                        " tokens. Please shorten your prompt."
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
            elif self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
                # Import lazily to avoid a lib <-> architectures cycle.
                from max.pipelines.architectures.flux2_modulev3.system_messages import (
                    format_input_klein,
                )

                messages_batch = format_input_klein(
                    prompts=[prompt_str],
                    images=None,
                )
                kwargs = dict(
                    add_generation_prompt=True,
                    tokenize=False,
                )
                try:
                    prompt_text = delegate.apply_chat_template(
                        messages_batch[0],
                        enable_thinking=False,
                        **kwargs,
                    )
                except TypeError:
                    prompt_text = delegate.apply_chat_template(
                        messages_batch[0],
                        **kwargs,
                    )
                # Validate prompt length before truncation.
                raw_ids = delegate.encode(
                    prompt_text,
                    add_special_tokens=add_special_tokens,
                )
                if max_sequence_length and len(raw_ids) > max_sequence_length:
                    raise ValueError(
                        f"Prompt is too long for this model's text"
                        f" encoder: {len(raw_ids)} tokens exceeds"
                        f" the maximum of {max_sequence_length}"
                        " tokens. Please shorten your prompt."
                    )

                return delegate(
                    prompt_text,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                    return_attention_mask=True,
                )
            elif self._pipeline_class_name.is_qwen_image_family:
                # QwenImage wraps prompts in a Qwen2 chat template.
                template = (
                    "<|im_start|>system\n"
                    "Describe the image by detailing the color, shape, "
                    "size, texture, quantity, text, spatial relationships "
                    "of the objects and background:<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                formatted = template.format(prompt_str)
                return delegate(
                    formatted,
                    padding=False,
                    max_length=max_sequence_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                )
            elif self._pipeline_class_name == PipelineClassName.ZIMAGE:
                # For Z-Image, use Qwen chat-template formatting.
                messages = [{"role": "user", "content": prompt_str}]
                if not hasattr(delegate, "apply_chat_template"):
                    raise ValueError(
                        "Z-Image requires tokenizer.apply_chat_template, "
                        "but the loaded tokenizer does not provide it."
                    )
                return delegate.apply_chat_template(
                    messages,
                    enable_thinking=True,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_sequence_length,
                    return_length=False,
                    return_overflowing_tokens=False,
                )
            else:
                # Validate prompt length before truncation.
                # The tokenizer's truncation=True silently drops
                # tokens beyond max_sequence_length; error early
                # instead.
                raw_ids = delegate.encode(
                    prompt_str,
                    add_special_tokens=add_special_tokens,
                )
                if max_sequence_length and len(raw_ids) > max_sequence_length:
                    raise ValueError(
                        f"Prompt is too long for this model's text"
                        f" encoder: {len(raw_ids)} tokens exceeds"
                        f" the maximum of {max_sequence_length}"
                        " tokens. Please shorten your prompt."
                    )
                return delegate(
                    prompt_str,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                )

        tokenizer_output = await run_with_default_executor(_encode_fn, prompt)

        # Extract input_ids and attention_mask from both dict-like and object-like
        # tokenizer outputs (e.g. BatchEncoding).
        input_ids: Any
        attention_mask: Any | None
        if hasattr(tokenizer_output, "__getitem__") and (
            hasattr(tokenizer_output, "keys")
            and "input_ids" in tokenizer_output
        ):
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output.get("attention_mask", None)
        elif hasattr(tokenizer_output, "input_ids"):
            input_ids = tokenizer_output.input_ids
            attention_mask = getattr(tokenizer_output, "attention_mask", None)
        else:
            raise ValueError(
                "Tokenizer output does not contain `input_ids`; cannot build PixelContext."
            )

        input_ids_array = np.asarray(input_ids, dtype=np.int64)
        if input_ids_array.ndim == 1:
            input_ids_array = input_ids_array[None, :]

        if attention_mask is None:
            attention_mask_array = np.ones_like(input_ids_array, dtype=np.bool_)
        else:
            attention_mask_array = np.asarray(attention_mask, dtype=np.bool_)
            if attention_mask_array.ndim == 1:
                attention_mask_array = attention_mask_array[None, :]

        if input_ids_array.shape != attention_mask_array.shape:
            raise ValueError(
                "Tokenizer produced mismatched `input_ids` and `attention_mask` shapes: "
                f"{input_ids_array.shape} vs {attention_mask_array.shape}."
            )

        # Flux2 text encoder path currently does not consume an explicit
        # attention mask. Strip padded tokens here and keep a dense mask.
        # FLUX2_KLEIN/Z-Image consume attention_mask directly.
        if self._pipeline_class_name == PipelineClassName.FLUX2:
            token_row = input_ids_array[0]
            mask_row = attention_mask_array[0]
            real_token_ids = token_row[mask_row]
            if real_token_ids.size == 0:
                raise ValueError(
                    f"{self._pipeline_class_name.value} tokenization produced "
                    "an empty effective prompt after attention masking."
                )
            input_ids_array = np.expand_dims(
                real_token_ids.astype(np.int64, copy=False), axis=0
            )
            attention_mask_array = np.ones_like(input_ids_array, dtype=np.bool_)

        if (
            max_sequence_length is not None
            and input_ids_array.shape[1] > max_sequence_length
        ):
            raise ValueError(
                "Input string is larger than tokenizer's max length "
                f"({input_ids_array.shape[1]} > {max_sequence_length})."
            )

        encoded_prompt = input_ids_array[0].astype(np.int64, copy=False)
        attention_mask_flat = attention_mask_array[0].astype(
            np.bool_, copy=False
        )

        return encoded_prompt, attention_mask_flat

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
        if isinstance(output, GenerationOutput):
            return output

        # Raw numpy path
        pixel_data = (output * 0.5 + 0.5).clip(min=0.0, max=1.0)
        return pixel_data

    @staticmethod
    def _retrieve_prompt(request: OpenResponsesRequest) -> str:
        """Retrieve the text prompt from an OpenResponsesRequest.

        Supports three input formats:
        1. input is a string - use directly as prompt
        2. input is a list of messages where first message content is a string - use as prompt
        3. input is a list of messages where first message content is a list - extract InputTextContent.text

        Args:
            request: The OpenResponsesRequest to extract the prompt from.

        Returns:
            The extracted text prompt.

        Raises:
            ValueError: If no valid prompt can be extracted from the request.
        """
        # Case 1: input is a string
        if isinstance(request.body.input, str):
            return request.body.input

        # Cases 2 & 3: input is a list of messages
        if isinstance(request.body.input, list):
            if not request.body.input:
                raise ValueError("Input message list cannot be empty.")

            first_message = request.body.input[0]

            # Case 2: message.content is a string
            if isinstance(first_message.content, str):
                return first_message.content

            # Case 3: message.content is a list
            if isinstance(first_message.content, list):
                # Extract text from all InputTextContent items
                text_parts = [
                    item.text
                    for item in first_message.content
                    if isinstance(item, InputTextContent)
                ]
                if not text_parts:
                    raise ValueError(
                        "No text content found in message. Please include at least one "
                        "InputTextContent item with a text prompt."
                    )
                return " ".join(text_parts)

            raise ValueError(
                f"Unexpected message content type: {type(first_message.content).__name__}"
            )

        raise ValueError(
            f"Input must be a string or list of messages, got {type(request.body.input).__name__}"
        )

    @staticmethod
    def _retrieve_images(
        request: OpenResponsesRequest,
    ) -> list[PIL.Image.Image]:
        """Retrieve all input images from an OpenResponsesRequest.

        Extracts all InputImageContent items from the first message's content
        list and converts data URIs to PIL Images.

        Args:
            request: The OpenResponsesRequest to extract images from.

        Returns:
            List of PIL Images (empty if none found).
        """
        # Only check list inputs
        if not isinstance(request.body.input, list):
            return []

        if not request.body.input:
            return []

        first_message = request.body.input[0]

        # Only check list content
        if not isinstance(first_message.content, list):
            return []

        images: list[PIL.Image.Image] = []
        for item in first_message.content:
            image_type = getattr(item, "type", None)
            image_url = getattr(item, "image_url", None)
            if (
                isinstance(item, InputImageContent)
                or image_type == "input_image"
            ) and isinstance(image_url, str):
                if image_url.startswith("data:"):
                    _, base64_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(base64_data)
                    images.append(PIL.Image.open(BytesIO(image_bytes)))

        return images

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> PixelContext:
        """Create a new PixelContext object, leveraging necessary information from OpenResponsesRequest."""
        # Extract prompt from request using the helper method
        prompt = self._retrieve_prompt(request)
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        # Extract input images from request content (takes precedence over input_image parameter)
        input_images_list = self._retrieve_images(request)
        if not input_images_list and input_image is not None:
            input_images_list = [input_image]

        # Extract image provider options (always available via defaults)
        image_options = request.body.provider_options.image
        if image_options is None:
            raise ValueError(
                "Image provider options are required for pixel generation. "
                "This should not happen as defaults are applied at request creation."
            )

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

        do_zimage_cfg = (
            self._pipeline_class_name == PipelineClassName.ZIMAGE
            and image_options.guidance_scale > 0.0
        )
        if self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
            is_distilled_klein = bool(
                self.diffusers_config.get("is_distilled", False)
            )
            do_true_cfg = (
                image_options.guidance_scale > 1.0 and not is_distilled_klein
            )
        else:
            do_true_cfg = (
                image_options.true_cfg_scale > 1.0
                and image_options.negative_prompt is not None
            )
        import PIL.Image

        # 1. Tokenize prompts
        # Convert input images to list format for _generate_tokens_ids
        images_for_tokenization: list[PIL.Image.Image] | None = None
        if input_images_list:
            images_for_tokenization = []
            for img in input_images_list:
                if isinstance(img, np.ndarray):
                    images_for_tokenization.append(
                        PIL.Image.fromarray(img.astype(np.uint8))
                    )
                else:
                    images_for_tokenization.append(img)

        (
            token_ids,
            attn_mask,
            token_ids_2,
            negative_token_ids,
            negative_attn_mask,
            negative_token_ids_2,
        ) = await self._generate_tokens_ids(
            prompt,
            image_options.secondary_prompt,
            image_options.negative_prompt,
            image_options.secondary_negative_prompt,
            do_true_cfg or do_zimage_cfg,
            images=images_for_tokenization,
        )

        default_sample_size = self._default_sample_size
        vae_scale_factor = self._vae_scale_factor

        requested_height = image_options.height
        requested_width = image_options.width

        height = (
            requested_height
            if requested_height is not None
            else default_sample_size * vae_scale_factor
        )
        width = (
            requested_width
            if requested_width is not None
            else default_sample_size * vae_scale_factor
        )

        # 2. Preprocess input images if provided
        preprocessed_image_arrays: list[npt.NDArray[np.uint8]] | None = None
        if input_images_list:
            if self._is_qwen_image_edit_family:
                target_height = None
                target_width = None
            else:
                target_height = (
                    requested_height if requested_height is not None else None
                )
                target_width = (
                    requested_width if requested_width is not None else None
                )
            preprocessed_image_arrays = []
            for img in input_images_list:
                preprocessed_image = self._preprocess_input_image(
                    img, target_height, target_width
                )
                if (
                    not preprocessed_image_arrays
                    and not self._is_qwen_image_edit_family
                ):
                    height = preprocessed_image.height
                    width = preprocessed_image.width
                preprocessed_image_arrays.append(
                    np.array(preprocessed_image, dtype=np.uint8).copy()
                )

        prompt_images: list[npt.NDArray[np.uint8]] | None = None
        vae_condition_images: list[npt.NDArray[np.uint8]] | None = None
        if self._is_qwen_image_edit_family:
            prompt_images, vae_condition_images = (
                self._prepare_qwen_edit_condition_images(
                    preprocessed_image_arrays
                )
            )
            if prompt_images:
                token_ids = self._prepare_qwen_edit_tokens(
                    prompt, prompt_images
                )
                attn_mask = np.ones_like(token_ids, dtype=np.bool_)
                if do_true_cfg and image_options.negative_prompt is not None:
                    negative_token_ids = self._prepare_qwen_edit_tokens(
                        image_options.negative_prompt,
                        prompt_images,
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

        # 2. Preprocess input image if provided
        preprocessed_image_array = None
        if input_image is not None:
            preprocessed_image = self._preprocess_input_image(
                input_image,
                target_height=image_options.height,
                target_width=image_options.width,
                preserve_aspect_ratio=(
                    self._pipeline_class_name != PipelineClassName.ZIMAGE
                ),
            )
            height = image_options.height or preprocessed_image.height
            width = image_options.width or preprocessed_image.width
            preprocessed_image_array = np.array(
                preprocessed_image, dtype=np.uint8
            ).copy()
        else:
            height = (
                image_options.height or default_sample_size * vae_scale_factor
            )
            width = (
                image_options.width or default_sample_size * vae_scale_factor
            )

        # 3. Resolve image dimensions using cached static values
        latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        num_inference_steps = (
            image_options.steps
            if "steps" in image_options.model_fields_set
            else self._default_num_inference_steps
        )
        sigma_min = (
            0.0
            if self._pipeline_class_name == PipelineClassName.ZIMAGE
            else None
        )
        timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
            image_seq_len, num_inference_steps, sigma_min=sigma_min
        )
        if (
            self._pipeline_class_name == PipelineClassName.ZIMAGE
            and self._scheduler_shift != 1.0
        ):
            # Match diffusers FlowMatchEulerDiscreteScheduler static shift behavior.
            # Z-Image scheduler config uses shift=6.0.
            shifted_timesteps = (
                self._scheduler_shift
                * timesteps
                / (1.0 + (self._scheduler_shift - 1.0) * timesteps)
            ).astype(np.float32)
            timesteps = shifted_timesteps
            sigmas = np.append(shifted_timesteps, np.float32(0.0))

        # Z-Image img2img follows diffusers strength behavior by starting
        # denoising from a later timestep.
        if (
            self._pipeline_class_name == PipelineClassName.ZIMAGE
            and input_image is not None
        ):
            init_timestep = min(
                num_inference_steps * image_options.strength,
                float(num_inference_steps),
            )
            t_start = int(max(num_inference_steps - init_timestep, 0.0))
            timesteps = timesteps[t_start:]
            sigmas = sigmas[t_start:]
            num_inference_steps = int(timesteps.shape[0])

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
            negative_mask=negative_attn_mask,
            negative_tokens_2=negative_token_buffer_2,
            explicit_negative_prompt=image_options.negative_prompt is not None,
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
            strength=image_options.strength,
            cfg_normalization=image_options.cfg_normalization,
            cfg_truncation=image_options.cfg_truncation,
            num_warmup_steps=num_warmup_steps,
            model_name=request.body.model,
            input_image=preprocessed_image_arrays[0]
            if preprocessed_image_arrays
            else None,
            input_images=preprocessed_image_arrays,
            prompt_images=prompt_images,
            vae_condition_images=vae_condition_images,
            output_format=image_options.output_format,
            residual_threshold=image_options.residual_threshold,
        )

        for validator in self._context_validators:
            validator(context)

        return context
