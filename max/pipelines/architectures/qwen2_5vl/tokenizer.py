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

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    ImageMetadata,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    get_seqlens,
    get_window_index,
)
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    fetch_image,
    process_vision_info,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import TextAndVisionTokenizer, max_tokens_to_generate
from max.pipelines.lib.config import PipelineConfig
from max.support.image import find_contiguous_ranges
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from .nn.data_processing import get_rope_index, mrope_pos_ids_3d

logger = logging.getLogger("max.pipelines")


def _convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    """Convert image to the specified mode."""
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


def _rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def qwen2_5vl_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
    """Preprocess image for Qwen2.5VL vision model.

    This function assumes the image has already been processed by fetch_image
    and is correctly sized. It only handles normalization and patch extraction.

    Args:
        image: PIL Image to preprocess (already resized by fetch_image)
        patch_size: Patch size for vision transformer (default 14)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2)

    Returns:
        Tuple of (pixel_values, image_grid_thw) where:
        - pixel_values: Flattened patch values as numpy array
        - image_grid_thw: Grid dimensions (temporal, height, width)
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Image is already correctly sized by fetch_image, no need to resize
    # Get actual dimensions
    width, height = image.size

    # Convert to numpy array and rescale to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Apply Standard ImageNet normalization (best match from testing)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Calculate grid dimensions based on actual image dimensions
    height_patches = height // patch_size
    width_patches = width // patch_size

    # Convert to numpy array
    patches = np.array([img_array])  # Shape: (n_images, height, width, 3)

    # Transpose to channel-first format
    patches = patches.transpose(
        0, 3, 1, 2
    )  # Shape: (n_images, 3, height, width)

    # Calculate dimensions
    channel = patches.shape[1]
    grid_h, grid_w = height_patches, width_patches

    # Handle temporal dimension
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(
            patches[-1][np.newaxis],
            temporal_patch_size - (patches.shape[0] % temporal_patch_size),
            axis=0,
        )
        patches = np.concatenate([patches, repeats], axis=0)

    # For images, grid_t should be 1 (single temporal group)
    grid_t = 1

    # Check if spatial merging is possible
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Spatial merging is not possible because grid_h {grid_h} % merge_size {merge_size} != 0 or grid_w {grid_w} % merge_size {merge_size} != 0"
        )
    else:
        # Now reshape with spatial merging
        patches = patches.reshape(
            grid_t,  # Temporal groups (1 for images)
            temporal_patch_size,  # Patches per temporal group (2)
            channel,  # RGB channels (3)
            grid_h // merge_size,  # Spatial groups in height (49)
            merge_size,  # Patches per spatial group (2)
            patch_size,  # Patch height (14)
            grid_w // merge_size,  # Spatial groups in width (73)
            merge_size,  # Patches per spatial group (2)
            patch_size,  # Patch width (14)
        )

        # Transpose following transformers library logic
        # This reorders dimensions to get the correct patch ordering
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        # Flatten patches
        # This preserves the patch ordering from the transpose
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

    # Create grid dimensions (temporal, height, width)
    image_grid_thw = (grid_t, grid_h, grid_w)

    return flatten_patches, image_grid_thw


class Qwen2_5VLImageProcessor:
    """Custom image processor for Qwen2.5VL that handles image processing without PyTorch dependencies.

    This processor mimics the interface of AutoImageProcessor but uses pure NumPy/PIL
    for image preprocessing.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ):
        """Initialize the custom image processor.

        Args:
            patch_size: Patch size for vision transformer
            temporal_patch_size: Temporal patch size
            merge_size: Spatial merge size (used for calculating image tokens)
        """
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.float32]]]:
        """Process images for Qwen2.5VL.

        Args:
            images: Single image or list of images to process
            return_tensors: Ignored (always returns numpy arrays)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary containing processed image data with keys:
            - pixel_values: Normalized pixel values as numpy array of shape (num_patches, patch_features)
            - image_grid_thw: Grid dimensions as numpy array of shape (num_images, 3) where each row is (temporal, height, width)
            List of pixel values for each image
        """
        # Handle single image vs list of images
        if isinstance(images, Image.Image):
            images = [images]

        # Process each image
        pixel_values_list: list[npt.NDArray[np.float32]] = []
        image_grid_thw_list: list[tuple[int, int, int]] = []

        for image in images:
            pixel_values, image_grid_thw_tuple = qwen2_5vl_image_preprocessing(
                image,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
            )
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw_tuple)

        # Stack results
        pixel_values = np.vstack(pixel_values_list)
        image_grid_thw_array: npt.NDArray[np.int32] = np.array(
            image_grid_thw_list, dtype=np.int32
        )

        return {
            "concatenated_pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw_array,
        }, pixel_values_list

    def preprocess(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.float32]]]:
        """Alias for __call__ to match transformers interface."""
        return self.__call__(images, return_tensors=return_tensors, **kwargs)


class Qwen2_5VLTokenizer(TextAndVisionTokenizer):
    """Qwen2.5VL-specific tokenizer that handles vision and text processing.

    This tokenizer uses separate AutoTokenizer and custom image processor
    to handle multimodal inputs for the Qwen2.5VL model.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        pipeline_config: PipelineConfig | None = None,
        **unused_kwargs,
    ):
        """Initialize the tokenizer with custom image processor instead of AutoProcessor."""
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Create encoding functions. Used by encode method in parent class.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )

        # Load config to get image processing parameters
        config = AutoConfig.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        # Extract vision config parameters
        vision_config = config.vision_config
        patch_size = getattr(vision_config, "patch_size", 14)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        self.spatial_merge_size = getattr(
            vision_config, "spatial_merge_size", 2
        )

        # NEW: Add these for window index calculation
        self.patch_size = patch_size
        self.window_size = getattr(vision_config, "window_size", 448)

        # Create custom image processor instead of AutoImageProcessor
        self.img_processor = Qwen2_5VLImageProcessor(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=self.spatial_merge_size,
        )

        # Initialize EOS token IDs
        self._default_eos_token_ids = set([self.eos])
        self._default_eos_token_ids = set([self.eos])

        if pipeline_config:
            huggingface_config = pipeline_config.model_config.huggingface_config
            if eos_token_id := getattr(
                huggingface_config, "eos_token_id", None
            ):
                if isinstance(eos_token_id, int):
                    self._default_eos_token_ids.add(eos_token_id)
                elif isinstance(eos_token_id, list):
                    self._default_eos_token_ids.update(eos_token_id)

            if image_token_id := getattr(
                pipeline_config.model_config.huggingface_config,
                "image_token_id",
                None,
            ):
                self.image_token_id = image_token_id
            else:
                raise ValueError(
                    "image_token_id not found in model_config config"
                )

            if video_token_id := getattr(
                pipeline_config.model_config.huggingface_config,
                "video_token_id",
                None,
            ):
                self.video_token_id = video_token_id

            if vision_start_token_id := getattr(
                pipeline_config.model_config.huggingface_config,
                "vision_start_token_id",
                None,
            ):
                self.vision_start_token_id = vision_start_token_id
            if vision_config := getattr(
                huggingface_config, "vision_config", None
            ):
                self.tokens_per_second = vision_config.tokens_per_second
            else:
                raise ValueError(
                    "vision_config must be provided in HuggingFace Config"
                )

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Apply chat template using tokenizer directly (not processor)."""
        # Use tokenizer directly instead of processor to avoid AutoProcessor dependency
        # TODO(E2EOPT-621): Wrap message content more generically.
        messages = self._wrap_str_message_content(messages)
        templated_message = self.delegate.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext for Qwen2.5VL processing.

        This method processes both text and vision inputs using the Qwen2.5VL
        processor and extracts the necessary components for model execution.
        """
        # Determine prompt
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
            if request.images:
                content = [
                    {"type": "text", "text": request.prompt},
                ] + [{"type": "image"} for _ in request.images]
                messages = [
                    TextGenerationRequestMessage(
                        role="user",
                        content=content,
                    )
                ]
                new_request = TextGenerationRequest(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    messages=messages,
                )
                assert new_request.messages is not None
                prompt = self.apply_chat_template(new_request.messages)
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Load and process images
        image_inputs = None
        if request.messages:
            # process_vision_info returns (image_inputs, video_inputs, placeholder_text)
            # Convert messages to the format expected by qwen_vl_utils
            # TextGenerationRequestMessage is a TypedDict, so it's already dict-like
            messages_data = [dict(msg) for msg in request.messages]
            image_inputs, _, _ = process_vision_info(
                messages_data
            )  # We ignore video_inputs for image-only use case
        else:
            # Fall back to using the loaded images
            if request.images:
                logger.info(
                    "Loading images from request.images rather than messages, not using process_vision_info"
                )
                image_inputs = [
                    fetch_image({"image": image_data})
                    for image_data in request.images
                ]

        # Step 1: Build chat text with tokenizer (not image processor)
        if isinstance(prompt, str):
            text = prompt
        else:
            # prompt is already processed tokens, convert back to text for processing
            text = self.delegate.decode(prompt, skip_special_tokens=True)

        # Step 2: Process images with custom image processor (if any)
        processed_images: dict[str, npt.NDArray[Any]] = {}
        pixel_values_list: list[npt.NDArray[np.float32]] = []

        image_grid_thw = None
        if image_inputs:
            processed_images, pixel_values_list = self.img_processor(
                images=image_inputs, return_tensors="pt"
            )

            # Step 3: Expand <|image_pad|> placeholders using image_grid_thw and merge_size**2
            if "image_grid_thw" in processed_images:
                grid = processed_images[
                    "image_grid_thw"
                ]  # List of (t, h, w) tuples
                merge_len = self.img_processor.merge_size**2

                # Expand placeholders for each image individually
                if "<|image_pad|>" in text:
                    for t, h, w in grid:
                        num_img_tokens = (t * h * w) // merge_len
                        # Replace first occurrence of <|image_pad|> with multiple <|image_pad|> tokens
                        # Use placeholder approach from example to avoid recursive replacement
                        placeholder_tokens = "<|placeholder|>" * num_img_tokens
                        text = text.replace(
                            "<|image_pad|>", placeholder_tokens, 1
                        )

                    # Convert all placeholders back to <|image_pad|> tokens
                    text = text.replace("<|placeholder|>", "<|image_pad|>")

        # Step 4: Tokenize the expanded text
        tokenizer_inputs = self.delegate(
            [text], padding=True, return_tensors=None
        )

        # Combine tokenizer and image processor outputs
        processed_inputs = {
            "input_ids": tokenizer_inputs["input_ids"],
            "attention_mask": tokenizer_inputs["attention_mask"],
        }

        # Add image processing results
        if processed_images:
            if "concatenated_pixel_values" in processed_images:
                processed_inputs["concatenated_pixel_values"] = (
                    processed_images["concatenated_pixel_values"]
                )
            if "image_grid_thw" in processed_images:
                processed_inputs["image_grid_thw"] = processed_images[
                    "image_grid_thw"
                ]

        if "input_ids" not in processed_inputs:
            raise ValueError("input_ids not generated by tokenizer")

        # Prepare extra model arguments
        extra_model_args = {}

        # Extract input_ids
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        if input_ids := processed_inputs.get("input_ids"):
            if isinstance(input_ids[0], int):
                seq = np.array(input_ids)
            else:
                seq = np.asarray(input_ids[0])

            image_token_indices = (
                (seq == self.image_token_id).nonzero()[0].astype(np.int32)
            )
            extra_model_args["image_token_indices"] = image_token_indices

        # Calculate max generation tokens
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        # Process vision inputs for Qwen2.5VL (image-only)
        attention_mask = None
        concatenated_pixel_values: npt.NDArray[Any] | None = None

        if image_inputs is not None:
            if "concatenated_pixel_values" in processed_inputs:
                concatenated_pixel_values = processed_inputs[
                    "concatenated_pixel_values"
                ]
                if not isinstance(concatenated_pixel_values, np.ndarray):
                    raise ValueError(
                        f"Expected concatenated_pixel_values to be a numpy array but got {type(concatenated_pixel_values)}"
                    )

            # Extract image_grid_thw if present (Qwen2.5VL specific)
            # Note: image_grid_thw is only used locally for computing other values, not passed to model
            if "image_grid_thw" in processed_inputs:
                image_grid_thw = processed_inputs["image_grid_thw"]
                # Handle numpy array from custom image processor
                if not isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = np.array(image_grid_thw)

                # Precompute vision_position_ids for this context
                vision_position_ids = mrope_pos_ids_3d(
                    grid_thw=image_grid_thw,
                    spatial_merge_size=self.spatial_merge_size,
                )
                extra_model_args["vision_position_ids"] = vision_position_ids

                # Precompute window index and cu_window_seqlens
                window_index, cu_window_seqlens = get_window_index(
                    grid_thw=image_grid_thw,
                    window_size=self.window_size,
                    spatial_merge_size=self.spatial_merge_size,
                    patch_size=self.patch_size,
                    spatial_merge_unit=self.spatial_merge_size**2,
                )
                extra_model_args["window_index"] = window_index
                # Note: cu_window_seqlens is only used locally, not passed to model

                # Precompute seqlens values
                (
                    cu_seqlens,
                    cu_window_seqlens_unique,
                    max_seqlen,
                    window_max_seqlen,
                ) = get_seqlens(
                    grid_thw=image_grid_thw,
                    cu_win_seqlens=cu_window_seqlens,
                )
                extra_model_args["cu_seqlens"] = cu_seqlens
                extra_model_args["cu_window_seqlens_unique"] = (
                    cu_window_seqlens_unique
                )
                extra_model_args["max_seqlen"] = np.array(
                    max_seqlen, dtype=np.int32
                )
                extra_model_args["window_max_seqlen"] = np.array(
                    window_max_seqlen, dtype=np.int32
                )

                # Precompute max_grid_size (max of height and width dimensions)
                max_grid_size = int(np.max(image_grid_thw[:, 1:]))
                extra_model_args["max_grid_size"] = np.array(
                    max_grid_size, dtype=np.int32
                )

            # Extract attention_mask for use in get_rope_index
            # Note: attention_mask is only used locally for get_rope_index, not passed to model
            if "attention_mask" in processed_inputs:
                attention_mask_raw = processed_inputs["attention_mask"]
                # Handle various formats from tokenizer
                if hasattr(attention_mask_raw, "numpy"):
                    attention_mask = attention_mask_raw.numpy()
                elif isinstance(attention_mask_raw, list):
                    attention_mask = np.array(attention_mask_raw)
                elif isinstance(attention_mask_raw, np.ndarray):
                    attention_mask = attention_mask_raw
                else:
                    attention_mask = np.array(attention_mask_raw)

        # Calculate Rope Delta and position ids
        temp_position_ids, rope_delta = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            input_ids=encoded_prompt.reshape(1, -1),
            image_grid_thw=image_grid_thw,
            # This is never calculated prior to this.
            video_grid_thw=None,
            # This is never calculated prior to this.
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )
        temp_position_ids = temp_position_ids.squeeze(1)
        extra_model_args["rope_delta"] = rope_delta
        extra_model_args["decoder_position_ids"] = temp_position_ids
        extra_model_args["concatenated_pixel_values"] = (
            concatenated_pixel_values
        )

        # Handle JSON schema if provided
        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        # Determine EOS token IDs
        if request.sampling_params.ignore_eos:
            eos_token_ids = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise ValueError(
                "encoded_prompt is greater than the max_length of the tokenizer"
            )

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, [self.image_token_id]
        )
        # Create and return context
        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            extra_model_args=extra_model_args,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                )
                for (start_idx, end_idx), pixels in zip(
                    start_and_end_idxs,
                    pixel_values_list,
                    strict=True,
                )
            ],
            vision_token_ids=[self.image_token_id],
        )
        return context
