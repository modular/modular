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
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    InputImageContent,
    InputTextContent,
)
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
    FLUX2_KLEIN = "Flux2KleinPipeline"
    ZIMAGE = "ZImagePipeline"
    LTX2 = "LTX2Pipeline"

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

            if "gemma" in type(self.delegate).__name__.lower():
                # Gemma expects left padding for chat-style prompts
                self.delegate.padding_side = "left"
                if self.delegate.pad_token is None:
                    self.delegate.pad_token = self.delegate.eos_token

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
        self._vae_spatial_compression_ratio = (
            2 ** (len(block_out_channels) - 1) if block_out_channels else 8
        )

        # Store static model dimensions
        self._default_sample_size = 128
        self._num_channels_latents = transformer_config["in_channels"] // 4

        if self._pipeline_class_name == PipelineClassName.LTX2:
            self._num_channels_latents = transformer_config["in_channels"]
            self._causal_offset = 1
            self._vae_temporal_compression_ratio = vae_config[
                "temporal_compression_ratio"
            ]
            # LTX-2 uses CausalVideoAutoencoder with no standard block_out_channels;
            # read spatial/temporal compression ratios directly from the VAE config.
            self._vae_spatial_compression_ratio = vae_config[
                "spatial_compression_ratio"
            ]
            self._patch_size = transformer_config["patch_size"]
            self._patch_size_t = transformer_config["patch_size_t"]
            audio_vae_config = components.get("audio_vae", {}).get(
                "config_dict", {}
            )
            self._audio_sampling_rate = audio_vae_config["sample_rate"]
            self._audio_hop_length = audio_vae_config["mel_hop_length"]
            self._mel_bins = audio_vae_config["mel_bins"]
            # LATENT_DOWNSAMPLE_FACTOR = 4
            self._mel_compression_ratio = 4

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

        self._max_pixel_size = None
        if self._pipeline_class_name in (
            PipelineClassName.FLUX2,
            PipelineClassName.FLUX2_KLEIN,
        ):
            self._max_pixel_size = 1024 * 1024

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
    ) -> npt.NDArray[np.float32]:
        # 1. 1-D grids for each spatial/temporal dimension.
        grid_f = np.arange(0, num_frames, self._patch_size_t, dtype=np.float32)
        grid_h = np.arange(0, height, self._patch_size, dtype=np.float32)
        grid_w = np.arange(0, width, self._patch_size, dtype=np.float32)

        # 2. Broadcast to 3-D grid [N_F, N_H, N_W] then stack → [3, N_F, N_H, N_W].
        grid = np.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = np.stack(grid, axis=0)  # [3, N_F, N_H, N_W]

        # 2. Get the patch boundaries with respect to the latent video grid
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = np.array(patch_size, dtype=grid.dtype)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        # Combine the start (grid) and end (patch_ends) coordinates along new trailing dimension
        latent_coords = np.stack(
            [grid, patch_ends], axis=-1
        )  # [3, N_F, N_H, N_W, 2]
        # Reshape to (batch_size, 3, num_patches, 2)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = np.expand_dims(latent_coords, axis=0).repeat(
            batch_size, axis=0
        )

        # 3. Calculate the pixel space patch boundaries from the latent boundaries.
        scale_tensor = np.array(
            (
                self._vae_temporal_compression_ratio,
                self._vae_spatial_compression_ratio,
                self._vae_spatial_compression_ratio,
            ),
            dtype=latent_coords.dtype,
        )
        # Broadcast the VAE scale factors such that they are compatible with latent_coords's shape
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1  # This is the (frame, height, width) dim
        # Apply per-axis scaling to convert latent coordinates to pixel space coordinates
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # As the VAE temporal stride for the first frame is 1 instead of self.vae_scale_factors[0], we need to shift
        # and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...]
            + self._causal_offset
            - self._vae_temporal_compression_ratio
        ).clip(min=0)

        # Scale the temporal coordinates by the video FPS
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def _prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        shift: int = 0,
    ) -> npt.NDArray[np.float32]:
        # 1. Generate coordinates in the frame (time) dimension.
        # Always compute rope in fp32
        grid_f = np.arange(
            start=shift,
            stop=num_frames + shift,
            step=self._patch_size_t,
            dtype=np.float32,
        )

        # 2. Calculate start timstamps in seconds with respect to the original spectrogram grid
        audio_scale_factor = self._vae_temporal_compression_ratio
        # Scale back to mel spectrogram space
        grid_start_mel = grid_f * audio_scale_factor
        # Handle first frame causal offset, ensuring non-negative timestamps
        grid_start_mel = (
            grid_start_mel + self._causal_offset - audio_scale_factor
        ).clip(min=0)
        # Convert mel bins back into seconds
        grid_start_s = (
            grid_start_mel * self._audio_hop_length / self._audio_sampling_rate
        )

        # 3. Calculate start timstamps in seconds with respect to the original spectrogram grid
        grid_end_mel = (grid_f + self._patch_size_t) * audio_scale_factor
        grid_end_mel = (
            grid_end_mel + self._causal_offset - audio_scale_factor
        ).clip(min=0)
        grid_end_s = (
            grid_end_mel * self._audio_hop_length / self._audio_sampling_rate
        )

        audio_coords = np.stack(
            [grid_start_s, grid_end_s], axis=-1
        )  # [num_patches, 2]
        audio_coords = np.expand_dims(audio_coords, axis=0).repeat(
            batch_size, axis=0
        )  # [batch_size, num_patches, 2]
        audio_coords = np.expand_dims(
            audio_coords, axis=1
        )  # [batch_size, 1, num_patches, 2]
        return audio_coords

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
    ) -> PIL.Image.Image:
        """Preprocess input image for image-to-image generation.

        Matches diffusers FLUX2 behavior:
        - cap image area when needed
        - floor dimensions to multiples of vae_scale_factor * 2
        - apply aspect-ratio preserving center-crop resize to the floored size

        Args:
            image: PIL Image or numpy array (uint8) to preprocess.

        Returns:
            Preprocessed PIL Image with adjusted dimensions.
        """
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image.astype(np.uint8))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_width, image_height = image.size
        multiple_of = self._vae_spatial_compression_ratio * 2

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

        if image.size != (image_width, image_height):
            image = self._resize_with_center_crop(
                image, image_width, image_height
            )

        return image

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        seed: int | None,
        num_frames: int | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        shape = (batch_size, num_channels_latents, latent_height, latent_width)
        if num_frames is not None:
            num_latent_frames = (
                num_frames - 1
            ) // self._vae_temporal_compression_ratio + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                latent_height,
                latent_width,
            )
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
        npt.NDArray[np.bool_] | None,
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.bool_] | None,
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
            Tuple of (
                token_ids,
                attn_mask,
                token_ids_2,
                attn_mask_2,
                negative_token_ids,
                negative_attn_mask,
                negative_token_ids_2,
            ).
            token_ids_2 and negative_token_ids_2 are None if no secondary tokenizer is configured.
        """
        token_ids, attn_mask = await self.encode(prompt, images=images)

        token_ids_2: npt.NDArray[np.int64] | None = None
        attn_mask_2: npt.NDArray[np.bool_] | None = None
        if self.delegate_2 is not None:
            token_ids_2, attn_mask_2 = await self.encode(
                prompt_2 or prompt,
                use_secondary=True,
            )

        negative_token_ids: npt.NDArray[np.int64] | None = None
        negative_attn_mask: npt.NDArray[np.bool_] | None = None
        negative_token_ids_2: npt.NDArray[np.int64] | None = None
        attn_mask_neg: npt.NDArray[np.bool_] | None = None
        if do_true_cfg:
            negative_token_ids, negative_attn_mask = await self.encode(
                negative_prompt or ""
            )
            if self.delegate_2 is not None:
                negative_token_ids_2, _negative_attn_mask_2 = await self.encode(
                    negative_prompt_2 or negative_prompt or "",
                    use_secondary=True,
                )

        return (
            token_ids,
            attn_mask,
            attn_mask_neg,
            token_ids_2,
            attn_mask_2,
            negative_token_ids,
            negative_attn_mask,
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

        def _encode_fn(prompt_str: str) -> Any:
            assert delegate is not None

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
            elif self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
                from max.pipelines.architectures.flux2.system_messages import (
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
                return delegate(
                    prompt_text,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                    return_attention_mask=True,
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

        # Extract input_ids and attention_mask.
        if isinstance(tokenizer_output, dict):
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output.get("attention_mask", None)
        else:
            input_ids = tokenizer_output.input_ids
            attention_mask = tokenizer_output.attention_mask

        input_ids_array = np.asarray(input_ids, dtype=np.int64)
        if attention_mask is None:
            attention_mask_array = np.ones_like(input_ids_array, dtype=np.bool_)
        else:
            attention_mask_array = np.asarray(attention_mask, dtype=np.bool_)

        # Tokenizers can return a batch dimension for a single prompt.
        if input_ids_array.ndim == 2:
            if input_ids_array.shape[0] != 1:
                raise ValueError(
                    "Expected one prompt during tokenization, got "
                    f"batch size {input_ids_array.shape[0]}."
                )
            input_ids_array = input_ids_array[0]
        elif input_ids_array.ndim != 1:
            raise ValueError(
                "Expected rank-1 or rank-2 input_ids, got "
                f"shape {input_ids_array.shape}."
            )

        if attention_mask_array.ndim == 2:
            if attention_mask_array.shape[0] != 1:
                raise ValueError(
                    "Expected one prompt attention_mask, got "
                    f"batch size {attention_mask_array.shape[0]}."
                )
            attention_mask_array = attention_mask_array[0]
        elif attention_mask_array.ndim != 1:
            raise ValueError(
                "Expected rank-1 or rank-2 attention_mask, got "
                f"shape {attention_mask_array.shape}."
            )

        if attention_mask_array.shape[0] != input_ids_array.shape[0]:
            raise ValueError(
                "input_ids and attention_mask must have the same sequence "
                f"length ({input_ids_array.shape[0]} != {attention_mask_array.shape[0]})."
            )

        # FLUX.2 uses compact token IDs; FLUX.2-Klein keeps full tokenizer output.
        if self._pipeline_class_name == PipelineClassName.FLUX2:
            input_ids_array = input_ids_array[attention_mask_array]
            attention_mask_array = np.ones(
                input_ids_array.shape[0], dtype=np.bool_
            )

        if (
            max_sequence_length
            and input_ids_array.shape[0] > max_sequence_length
        ):
            raise ValueError(
                "Input string is larger than tokenizer's max length "
                f"({input_ids_array.shape[0]} > {max_sequence_length})."
            )

        encoded_prompt = input_ids_array.astype(np.int64, copy=False)
        attention_mask_array = attention_mask_array.astype(np.bool_, copy=False)

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
    def _retrieve_image(
        request: OpenResponsesRequest,
    ) -> PIL.Image.Image | None:
        """Retrieve the input image from an OpenResponsesRequest.

        Extracts InputImageContent from the first message's content list and converts
        the data URI to a PIL Image.

        Args:
            request: The OpenResponsesRequest to extract the image from.

        Returns:
            PIL Image if found, None otherwise.
        """
        # Only check list inputs
        if not isinstance(request.body.input, list):
            return None

        if not request.body.input:
            return None

        first_message = request.body.input[0]

        # Only check list content
        if not isinstance(first_message.content, list):
            return None

        # Find first InputImageContent item
        for item in first_message.content:
            if isinstance(item, InputImageContent):
                # Parse data URI and convert to PIL Image
                image_url = item.image_url
                if image_url.startswith("data:"):
                    # Extract base64 data from data URI
                    # Format: data:image/png;base64,<base64_data>
                    _, base64_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(base64_data)
                    return PIL.Image.open(BytesIO(image_bytes))

        return None

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

        # Extract input image from request content (takes precedence over input_image parameter)
        input_image = self._retrieve_image(request) or input_image

        # Extract image provider options (always available via defaults)
        visual_options = getattr(
            request.body.provider_options,
            "video",
            request.body.provider_options.image,
        )
        if visual_options is None:
            raise ValueError(
                "Visual provider options are required for visual generation. "
                "This should not happen as defaults are applied at request creation."
            )

        if (
            visual_options.guidance_scale < 1.0
            or visual_options.true_cfg_scale < 1.0
        ):
            logger.warning(
                f"Guidance scales < 1.0 detected (guidance_scale={visual_options.guidance_scale}, "
                f"true_cfg_scale={visual_options.true_cfg_scale}). This is mathematically possible"
                " but may produce lower quality or unexpected results."
            )

        if (
            visual_options.true_cfg_scale > 1.0
            and visual_options.negative_prompt is None
        ):
            logger.warning(
                f"true_cfg_scale={visual_options.true_cfg_scale} is set, but no negative_prompt "
                "is provided. True classifier-free guidance requires a negative prompt; "
                "falling back to standard generation."
            )

        if self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
            is_distilled_klein = bool(
                self.diffusers_config.get("is_distilled", False)
            )
            # for non-distilled models, CFG is enabled
            # whenever guidance_scale > 1.0; negative prompt defaults to "".
            do_true_cfg = (
                visual_options.guidance_scale > 1.0 and not is_distilled_klein
            )
        else:
            do_true_cfg = (
                visual_options.true_cfg_scale > 1.0
                and visual_options.negative_prompt is not None
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
            attn_mask_neg,
            token_ids_2,
            _attn_mask_2,
            negative_token_ids,
            _negative_attn_mask,
            negative_token_ids_2,
        ) = await self._generate_tokens_ids(
            prompt,
            visual_options.secondary_prompt,
            visual_options.negative_prompt,
            visual_options.secondary_negative_prompt,
            do_true_cfg,
            images_for_tokenization,
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
        vae_spatial_compression_ratio = self._vae_spatial_compression_ratio

        # 2. Preprocess input image if provided
        preprocessed_image_array = None
        if input_image is not None:
            preprocessed_image = self._preprocess_input_image(input_image)
            height = visual_options.height or preprocessed_image.height
            width = visual_options.width or preprocessed_image.width
            preprocessed_image_array = np.array(
                preprocessed_image, dtype=np.uint8
            ).copy()
        else:
            height = (
                visual_options.height
                or default_sample_size * vae_spatial_compression_ratio
            )
            width = (
                visual_options.width
                or default_sample_size * vae_spatial_compression_ratio
            )

        # 3. Resolve image dimensions using cached static values
        latent_height = 2 * (
            int(height) // (self._vae_spatial_compression_ratio * 2)
        )
        latent_width = 2 * (
            int(width) // (self._vae_spatial_compression_ratio * 2)
        )
        visual_seq_len = (
            (latent_height // 2)
            * (latent_width // 2)
            * (
                (visual_options.num_frames - 1)
                // self._vae_temporal_compression_ratio
                + 1
                if getattr(visual_options, "num_frames", None) is not None
                else 1
            )
        )

        timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
            visual_seq_len, visual_options.steps
        )

        num_warmup_steps: int = max(
            len(timesteps) - visual_options.steps * self._scheduler.order, 0
        )

        latents, latent_image_ids = self._prepare_latents(
            visual_options.num_visuals,
            self._num_channels_latents,
            latent_height,
            latent_width,
            request.body.seed,
            getattr(visual_options, "num_frames", None),
        )

        extra_params: dict[str, npt.NDArray[Any]] = {}
        if self._pipeline_class_name == PipelineClassName.LTX2:
            latent_mel_bins = self._mel_bins // self._mel_compression_ratio
            duration_s = (
                visual_options.num_frames / visual_options.frames_per_second
            )
            audio_latents_per_second = (
                self._audio_sampling_rate
                / self._audio_hop_length
                / float(self._mel_compression_ratio)
            )
            audio_num_frames = round(duration_s * audio_latents_per_second)
            audio_shape = (
                visual_options.num_visuals,
                8,
                audio_num_frames,
                latent_mel_bins,
            )
            audio_latents = self._randn_tensor(audio_shape, request.body.seed)
            extra_params["audio_latents"] = audio_latents
            latent_num_frames = (
                visual_options.num_frames - 1
            ) // self._vae_temporal_compression_ratio + 1
            extra_params["latent_mel_bins"] = np.array(
                latent_mel_bins, dtype=np.int64
            )

            video_coords = self._prepare_video_coords(
                visual_options.num_visuals,
                latent_num_frames,
                latent_height,
                latent_width,
                visual_options.frames_per_second,
            )
            audio_coords = self._prepare_audio_coords(
                visual_options.num_visuals,
                audio_num_frames,
            )

            if visual_options.guidance_scale > 1.0:
                video_coords = np.concatenate([video_coords, video_coords])
                audio_coords = np.concatenate([audio_coords, audio_coords])
            extra_params["video_coords"] = video_coords
            extra_params["audio_coords"] = audio_coords

            valid_length = np.atleast_2d(
                np.array(attn_mask.sum(axis=-1), dtype=np.int32)
            )
            if visual_options.guidance_scale > 1.0:
                valid_length_neg = np.atleast_2d(
                    np.array(attn_mask_neg.sum(axis=-1), dtype=np.int32)
                )
                valid_length = np.concatenate([valid_length_neg, valid_length])
            extra_params["valid_length"] = valid_length

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
            num_inference_steps=visual_options.steps,
            guidance_scale=visual_options.guidance_scale,
            num_visuals_per_prompt=visual_options.num_visuals,
            num_frames=visual_options.num_frames,
            frame_rate=visual_options.frames_per_second,
            true_cfg_scale=visual_options.true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            model_name=request.body.model,
            input_image=preprocessed_image_array,  # Pass numpy array instead of PIL.Image
            extra_params=extra_params,
        )

        for validator in self._context_validators:
            validator(context)

        return context
