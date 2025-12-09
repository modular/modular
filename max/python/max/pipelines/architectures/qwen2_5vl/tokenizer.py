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

import asyncio
import functools
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    float32_to_bfloat16_as_uint16,
    max_tokens_to_generate,
)
from max.pipelines.lib.config import PipelineConfig
from max.support.image import find_contiguous_ranges
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from .context import Qwen2_5VLTextAndVisionContext, VisionEncodingData
from .nn.data_processing import get_rope_index, mrope_pos_ids_3d

logger = logging.getLogger("max.pipelines")


# Pre-computed normalization constants for ImageNet
# These are computed as: scale = 1 / (255 * std), offset = -mean / std
# This allows simplified normalization: normalized = pixel * scale + offset
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_NORM_OFFSET = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


def qwen2_5vl_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple[npt.NDArray[np.uint16], tuple[int, int, int]]:
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

    # Get actual dimensions
    width, height = image.size

    # Calculate grid dimensions based on actual image dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Check if spatial merging is possible early
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Spatial merging is not possible because grid_h {grid_h} % merge_size {merge_size} != 0 or grid_w {grid_w} % merge_size {merge_size} != 0"
        )

    # Convert to numpy array (float32) with simplified normalization
    # This combines: (pixel / 255.0 - mean) / std = pixel * scale + offset
    # Using in-place operations to reduce memory allocations
    img_array = np.array(image, dtype=np.float32)
    np.multiply(img_array, _NORM_SCALE, out=img_array)
    np.add(img_array, _NORM_OFFSET, out=img_array)

    # For single images, temporal dimension is always 1 and we need to repeat
    # for temporal_patch_size.
    channel = 3
    grid_t = 1

    # Transpose to channel-first: (H, W, 3) -> (3, H, W)
    img_chw = img_array.transpose(2, 0, 1)

    # Add temporal dimension (single frame for images, will tile to temporal_patch_size at the end)
    patches = img_chw[np.newaxis]  # (1, 3, H, W)

    # Reshape with spatial merging
    # Input shape: (1, channel, height, width) - single temporal frame
    patches = patches.reshape(
        grid_t,  # Temporal groups (1 for images)
        1,  # Single frame, will tile at the end
        channel,  # RGB channels (3)
        grid_h // merge_size,  # Spatial groups in height
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch height (14)
        grid_w // merge_size,  # Spatial groups in width
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch width (14)
    )

    # Transpose following transformers library logic
    # This reorders dimensions to get the correct patch ordering
    # Output shape: (grid_t, gh//m, gw//m, m, m, channel, 1, ps, ps)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    # Tile for temporal dimension: images have 1 frame but model expects
    # temporal_patch_size frames, so we replicate the single frame.
    num_patches = grid_t * grid_h * grid_w
    # Reshape to expose temporal dimension: (num_patches, channel, 1, patch_size^2)
    patches_4d = patches.reshape(
        num_patches, channel, 1, patch_size * patch_size
    )
    # Tile to (num_patches, channel, temporal_patch_size, patch_size^2)
    patches_tiled = np.tile(patches_4d, (1, 1, temporal_patch_size, 1))
    # Flatten to final shape: (num_patches, channel * temporal_patch_size * patch_size^2)
    flatten_patches = patches_tiled.reshape(
        num_patches,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    flatten_patches_uint16 = float32_to_bfloat16_as_uint16(flatten_patches)

    # Create grid dimensions (temporal, height, width)
    image_grid_thw = (grid_t, grid_h, grid_w)

    return flatten_patches_uint16, image_grid_thw


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
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
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
        pixel_values_list: list[npt.NDArray[np.uint16]] = []
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
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
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
        self.executor: ThreadPoolExecutor | None = None

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Apply chat template using tokenizer directly (not processor)."""
        templated_message = self.delegate.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Qwen2_5VLTextAndVisionContext:
        """Create a new Qwen2_5VLTextAndVisionContext for Qwen2.5VL processing.

        This method processes both text and vision inputs using the Qwen2.5VL
        processor and extracts the necessary components for model execution.
        """
        if self.executor is None:
            # lazy init the executor because the tokenizer gets pickled
            # when launching the model worker, and the executor is not pickle-safe
            self.executor = ThreadPoolExecutor(max_workers=2)

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self.new_context_blocking, request
        )

    def new_context_blocking(
        self, request: TextGenerationRequest
    ) -> Qwen2_5VLTextAndVisionContext:
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
        pixel_values_list: list[npt.NDArray[np.uint16]] = []

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
        else:
            image_token_indices = np.array([], dtype=np.int32)

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
        vision_data: VisionEncodingData | None = None

        # Extract attention_mask for use in get_rope_index
        # This should be extracted regardless of whether images are present
        # since the tokenizer always provides attention_mask
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

        if image_inputs is not None:
            concatenated_pixel_values: npt.NDArray[Any] | None = None
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

                # Precompute window index and cu_window_seqlens
                window_index, cu_window_seqlens = get_window_index(
                    grid_thw=image_grid_thw,
                    window_size=self.window_size,
                    spatial_merge_size=self.spatial_merge_size,
                    patch_size=self.patch_size,
                    spatial_merge_unit=self.spatial_merge_size**2,
                )
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
                max_seqlen_arr = np.array(max_seqlen, dtype=np.uint32)
                window_max_seqlen_arr = np.array(
                    window_max_seqlen, dtype=np.uint32
                )

                # Precompute max_grid_size (max of height and width dimensions)
                max_grid_size = np.array(
                    int(np.max(image_grid_thw[:, 1:])), dtype=np.int32
                )

                # Create VisionEncodingData with all vision-specific fields
                if concatenated_pixel_values is not None:
                    vision_data = VisionEncodingData(
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                        vision_position_ids=vision_position_ids,
                        window_index=window_index,
                        max_grid_size=max_grid_size,
                        cu_seqlens=cu_seqlens,
                        cu_window_seqlens_unique=cu_window_seqlens_unique,
                        max_seqlen=max_seqlen_arr,
                        window_max_seqlen=window_max_seqlen_arr,
                        concatenated_pixel_values=concatenated_pixel_values,
                    )

        # Calculate Rope Delta and position ids
        decoder_position_ids, rope_delta_array = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            input_ids=encoded_prompt.reshape(1, -1),
            image_grid_thw=vision_data.image_grid_thw
            if vision_data is not None
            else None,
            # This is never calculated prior to this.
            video_grid_thw=None,
            # This is never calculated prior to this.
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )
        decoder_position_ids = decoder_position_ids.squeeze(1)
        rope_delta = int(rope_delta_array.item())

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

        if pixel_values_list:
            start_and_end_idxs = find_contiguous_ranges(
                encoded_prompt, [self.image_token_id]
            )
            images = [
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixel_values,
                )
                for (start_idx, end_idx), pixel_values in zip(
                    start_and_end_idxs, pixel_values_list, strict=True
                )
            ]
        else:
            images = []

        # Create and return context
        context = Qwen2_5VLTextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            images=images,
            vision_token_ids=[self.image_token_id],
            # Qwen2.5VL-specific fields
            spatial_merge_size=self.spatial_merge_size,
            rope_delta=rope_delta,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            image_token_indices=image_token_indices,
            decoder_position_ids=decoder_position_ids,
            vision_data=vision_data,
        )
        return context
